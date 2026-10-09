// DeBERTa-v2/v3 encoder: disentangled attention over a shared relative-position
// embedding.
//
// Layout notes that matter and are easy to get wrong:
//
//  * POST-LN, like BERT. The norm comes after the residual add, not before
//    (contrast ModernBERT / NeoBERT which are pre-LN). There is no attn_norm
//    before attention at all -- the norm IS attention.output.LayerNorm.
//  * No RoPE, no absolute position embeddings (position_biased_input=false),
//    no token_type embeddings. Position enters only through the relative
//    embedding, so this is order-sensitive without any rotary machinery.
//  * `rel_embd` is MODEL level. HF builds it once on the encoder and hands the
//    same tensor to all 24 layers. Each layer derives its own pos_q / pos_k by
//    applying that layer's q/k projections to it -- so `share_att_key` means
//    there are no separate pos_* weights to load.
//  * The relative term is a *dot product*, not an additive per-dim bias like
//    T5's pos_bias. T5's [n_tok, n_embd, n_kv] intermediate and DeBERTa's have
//    the same shape and the same get_rows/reshape/permute dance, but DeBERTa's
//    then gets contracted against the query, per head, with a sum over that
//    head's n_embd_head dims only.

#include "models.h"

void llama_model_deberta_v3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_eps);

    // Disentangled attention constants. Both are required -- a DeBERTa graph
    // with either at zero cannot bucket relative positions, so fail at load
    // rather than producing plausible garbage.
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, hparams.n_rel_pos_bkts);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_POS_MAX,       hparams.n_rel_pos_max);

    if (hparams.n_rel_pos_bkts == 0) {
        throw std::runtime_error(
            "deberta: missing '<arch>.attention.relative_buckets_count' -- "
            "disentangled attention cannot be built without position_buckets");
    }
    if (hparams.n_rel_pos_max == 0) {
        throw std::runtime_error(
            "deberta: missing '<arch>.attention.relative_pos_max' -- "
            "cannot resolve the relative-position bucket curve");
    }

    // The relative embedding table has 2 * position_buckets rows (c2p indexes
    // the upper half, p2c the lower). Check it against the file rather than
    // trusting the hparam, so a mis-converted file is caught at load.
    hparams.n_rel_pos_rows = 2 * hparams.n_rel_pos_bkts;

    hparams.llm_ffn_op = LLM_FFN_GELU;
    std::string hidden_act;
    if (ml.get_key(LLM_KV_HIDDEN_ACT, hidden_act, false)) {
        hparams.llm_ffn_op = llm_ffn_op_type_from_string(hidden_act, LLM_FFN_GELU);
    }

    switch (hparams.n_layer()) {
        case 12: type = LLM_TYPE_137M; break;
        case 24: type = LLM_TYPE_410M; break; // deberta-v3-large (435M encoder)
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_deberta_v3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    tok_embd   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    tok_norm   = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);
    tok_norm_b = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "bias",   0), {n_embd}, 0);

    // Relative position embedding, shared by every layer.
    rel_embd = create_tensor(tn(LLM_TENSOR_ENC_ATTN_REL_EMB, "weight"),
                             {n_embd, hparams.n_rel_pos_rows}, 0);

    // Encoder final norm. ALSO normalises rel_embd: HF assigns self.LayerNorm
    // once inside DebertaV2Encoder and uses it for both norm_rel_ebd and the
    // output, so these tensors are read twice rather than stored twice.
    output_norm   = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output_norm_b = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "bias"),   {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd_gqa, n_embd_gqa, 0);

        layer.wo   = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "bias",   i), {n_embd}, 0);

        // post-attention norm (applied to attn_out + residual)
        layer.attn_out_norm   = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.attn_out_norm_b = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM, "bias",   i), {n_embd}, 0);

        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up_b = create_tensor(tn(LLM_TENSOR_FFN_UP,   "bias",   i), {n_ff}, 0);
        layer.ffn_down   = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_down_b = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "bias",   i), {n_embd}, 0);

        layer.layer_out_norm   = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, 0);
        layer.layer_out_norm_b = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "bias",   i), {n_embd}, 0);
    }

    // GLiNER decision head: Linear(n_embd, head_dim) -> ReLU -> Linear(head_dim, 1).
    // A plain DeBERTa file has no head_dim and keeps the per-token hidden state.
    uint32_t head_dim = 0;
    if (ml.get_key(LLM_KV_DECISION_HEAD_DIM, head_dim, false) && head_dim > 0) {
        hparams.n_embd_out_impl = 1;
        cls       = create_tensor(tn(LLM_TENSOR_DECISION_HEAD_HIDDEN, "weight"), {n_embd, head_dim}, 0);
        cls_b     = create_tensor(tn(LLM_TENSOR_DECISION_HEAD_HIDDEN, "bias"),   {head_dim}, 0);
        cls_out   = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT,  "weight"), {head_dim}, 0);
        cls_out_b = create_tensor(tn(LLM_TENSOR_DECISION_SCORER_OUT,  "bias"),   {1}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_deberta_v3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// ---------------------------------------------------------------------------
// relative term
// ---------------------------------------------------------------------------

// out[i, j, h] = sum_c pos[h, c, idx[i, j]] * x4[h, c, <axis>]
//
//   pos     [n_embd, n_rows]  layer-projected relative embedding; ne0 is the
//                            linear output feature, ordered h*n_embd_head + d
//   idx     I32 [n_row, n_col]  gather indices; n_row = query, n_col = key
//   x4      [n_embd_head, n_head, n_row, 1]  x indexed by the QUERY (c2p)
//          [n_embd_head, n_head, 1, n_col]  x indexed by the KEY  (p2c)
//   out     [n_col, n_row, n_head]  one scalar per (head, query, key)
//
// x4 arrives already split into (head, head-local dim) so that ne0 is
// head-local: ggml_sum_rows reduces ne0, and reducing the full n_embd is the
// bug this function exists to prevent. The reference contracts head h's 64
// dims against head h's slice of the projected relative embedding; summing all
// n_embd dims and broadcasting across the head axis instead averages sixteen
// heads' position terms and applies that average to all sixteen, cancelling
// most of the per-head signal. Measured end-to-end against the reference:
// cos 0.924 collapsed vs 1.000000 per-head (~/work/deberta_rel_probe.py).
//
// The caller supplies the 4D shape because the two terms contract against
// different axes: c2p pairs the QUERY with the position, p2c pairs the KEY.
static ggml_tensor * build_deberta_rel_dot(
        ggml_context * ctx,
        ggml_tensor  * pos,   // [n_embd, n_rows]
        ggml_tensor  * idx,   // I32 [n_row, n_col]
        ggml_tensor  * x4) {  // [n_embd_head, n_head, n_row|1, n_col|1]
    const int64_t n_embd_head = x4->ne[0];
    const int64_t n_head      = x4->ne[1];
    const int64_t n_row       = idx->ne[0];
    const int64_t n_col       = idx->ne[1];
    const int64_t n_embd      = pos->ne[0];

    GGML_ASSERT(pos->ne[0] == n_embd_head * n_head);
    GGML_ASSERT(x4->ne[2] == n_row || x4->ne[2] == 1);
    GGML_ASSERT(x4->ne[3] == n_col || x4->ne[3] == 1);

    // ggml_get_rows(a, b) yields [a->ne[0], len(b)] with result[c, j] = a[b[j], c]:
    // the gathered axis is dim 0. That is what we want -- the embedding axis must
    // be dim 0 because it is the axis contracted over. Do NOT permute pos to put
    // the bucket axis first; that yields [n_rows, n_idx] and the reshape below
    // then fails the element count.
    ggml_tensor * pos_c = ggml_cont(ctx, pos);   // mul_mat output is a view

    ggml_tensor * idx_1d = ggml_reshape_1d(ctx, idx, n_row * n_col);
    ggml_tensor * g = ggml_get_rows(ctx, pos_c, idx_1d);   // [n_embd, n_row*n_col]

    // Already [n_embd, n_row, n_col] -- no permute.
    //
    // idx is [n_row, n_col] with ne0 = KEY. llm_graph_input_pos_bucket_deberta::
    // set_input writes idx[j*n_tokens + i] with j the key (pos[j]) and i the
    // query (pos[i]); a ggml element (ne0, ne1) sits at ne0 + N*ne1, so the key
    // lands on ne0 and the query on ne1. n_row is therefore the KEY count and
    // n_col the QUERY count -- the names are misleading, the arithmetic is not.
    //
    // reshape_1d enumerates that key-major: t = key * n_row + query. get_rows
    // returns t on ne1, so ne1 is the fast axis of the reshape below and must
    // carry the key -- hence reshape_3d(g, n_embd, n_row, n_col) puts the key
    // on ne1 and the query on ne2.
    //
    // An earlier version reshaped to (n_embd, n_col, n_row) and then permuted
    // (0,2,1,3); those two operations CANCEL and together they TRANSPOSE the
    // (query, key) grid, so the bias indexed [c, key, query] was read as
    // [c, query, key]. Verified with explicit integers in
    // ~/work/getrows_probe.py: the permuted form yields [[3,4,5],[2,3,4],
    // [1,2,3]] where the correct answer is [[3,2,1],[4,3,2],[5,4,3]].
    //
    // Do not "fix" this by swapping the reshape arguments. Commit c01ca2c4c
    // did exactly that, reasoning that ne0 was the query; its own new comment
    // then claimed reshape_3d(g, n_embd, n_col, n_row) yields
    // [n_embd, key, query] while n_col holds queries, so the fix transposed
    // the grid exactly as the permute had. It was reverted in 286977a0c.
    // The GGML_ASSERTs cannot catch this either: pos_c2p is square, so
    // n_row == n_col and both spellings satisfy them.
    g = ggml_reshape_3d(ctx, g, n_embd, n_row, n_col);

    // Split ne0 into (n_embd_head, n_head): ne0 must be the head-LOCAL dim,
    // because that is the axis ggml_sum_rows collapses, and ne1 carries the head
    // so the contraction stays per-head.
    //
    // `pos` ne0 is a linear layer's output feature, ordered head-major, so the
    // feature index is c = h * n_embd_head + d with d the faster position within
    // each head. ggml's reshape also puts the new ne0 on the fastest axis, so
    // reshape_4d(..., n_embd_head, n_head, ...) yields A[d, h] sourced from
    // index d + n_embd_head*h, which is the same feature (h, d) that the
    // contraction pairs up. The two conventions agree precisely because both
    // make d the fast axis.
    //
    // Reversing the pair -- reshape_4d(..., n_head, n_embd_head, ...) -- puts
    // n_embd_head on ne2, where sum_rows cannot collapse it, and the bias
    // arrives as [n_kv, n_q, n_embd_head] instead of [n_kv, n_q, n_head].
    // ggml_add then rejects it against KQ. Measured with a temporary shape
    // print: kq_b=[4,4,64,1] against KQ=[4,4,16,1].
    g = ggml_reshape_4d(ctx, g, n_embd_head, n_head, n_row, n_col);

    // ggml_mul returns a tensor shaped like its FIRST argument and requires the
    // second to divide it, so the full [n_embd_head, n_head, n_row, n_col] term
    // must lead and x4 is reshaped to match. The broadcast is per-dimension
    // modulo (ggml_can_repeat), so x4's size-1 axes expand.
    ggml_tensor * s = ggml_sum_rows(ctx, ggml_mul(ctx, g, x4));  // [1, n_head, n_row, n_col]

    // build_attn_mha's KQ is mul_mat(k, q) -> [n_kv, n_tok_q, n_head, 1]: ne0 is
    // the KEY axis, ne1 the query, ne2 the head. So hand back [n_col, n_row,
    // n_head, 1]. Getting this wrong is SILENT in self-attention, where
    // n_row == n_col makes ggml_can_repeat pass either way and the bias is added
    // TRANSPOSED. T5's build_pos_bias is the in-tree precedent for this axis
    // order; per-head vs per-position is a separate axis question and T5 gets
    // that right too (its attn_rel_b is [n_head, n_rel_attn_bkts]).
    //
    // NOTE ggml_permute's axis arguments are SCATTER: ggml_permute(a, a0..a3)
    // assigns ne[a_i] = a->ne[i], i.e. each source axis names its own
    // destination slot. src is (1, n_head, n_row, n_col) and we want
    // (n_col, n_row, n_head, 1), so the call is (3, 2, 1, 0). Reading the args
    // as a gather gives the identity and silently leaves the bias transposed.
    return ggml_cont(ctx, ggml_permute(ctx, s, 3, 2, 1, 0));
}

// ---------------------------------------------------------------------------
// graph
// ---------------------------------------------------------------------------

llama_model_deberta_v3::graph::graph(const llama_model & model, const llm_graph_params & params)
        : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(hparams.n_rel_pos_bkts > 0);
    GGML_ASSERT(hparams.n_rel_pos_max   > 0);
    GGML_ASSERT(model.rel_embd);

    // Host-filled relative-position gather index. ONE grid serves both terms --
    // see the note on the p2c call site. Hold the owning pointer for the whole
    // constructor: it is moved into res only at the end. (Returning a tensor out
    // of a builder that has already moved its input into res is a null deref --
    // do not reintroduce that shape.)
    auto inp_rel = std::make_unique<llm_graph_input_pos_bucket_deberta>(hparams);

    inp_rel->pos_c2p = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    ggml_set_input(inp_rel->pos_c2p);

    ggml_tensor * const rel_idx = inp_rel->pos_c2p;

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    inpL = build_norm(inpL, model.tok_norm, model.tok_norm_b, LLM_NORM, 0);
    cb(inpL, "inp_norm", 0);

    // norm_rel_ebd == "layer_norm". The encoder's single LayerNorm exists for
    // exactly this -- it is created inside DebertaV2Encoder.__init__ to
    // normalise the relative-position embedding and is applied to nothing else.
    // There is no final output norm; do not add one (see the graph tail).
    //
    // Cast to F32 first: the CUDA norm kernel asserts an F32 source, and unlike
    // the LayerNorm weight vectors -- which the quantizer leaves alone -- this is
    // a 512x1024 matrix it will happily convert. At Q8_0 that made the model
    // unloadable (ggml-cuda/norm.cu:441 GGML_ASSERT(src0->type == GGML_TYPE_F32)
    // failed). It is 2 MB, 0.4 % of the q8_0 model, so quantising it buys
    // nothing. The cast is a no-op when the tensor is already F32.
    ggml_tensor * rel_src = model.rel_embd;
    if (rel_src->type != GGML_TYPE_F32) {
        rel_src = ggml_cast(ctx0, rel_src, GGML_TYPE_F32);
        cb(rel_src, "rel_embd_f32", -1);
    }

    ggml_tensor * rel_e = build_norm(rel_src,
                                     model.output_norm, model.output_norm_b,
                                     LLM_NORM, -1);
    cb(rel_e, "rel_e", -1);

    auto * inp_attn = build_attn_inp_no_cache();

    // No out_ids gather. build_inp_out_ids() always returns a tensor (its
    // nullptr shortcut is commented out upstream); it is an identity ONLY when
    // n_outputs == n_tokens, and otherwise selects just the output-marked
    // tokens, collapsing [n_embd, n_tokens] to a single row that the server
    // then broadcasts to every slot. With --pooling none we need every row, so
    // skip the optimisation rather than depend on that invariant holding.
    ggml_tensor * inp_out_ids = nullptr;

    // scale_factor = 1 + (c2p) + (p2c) for pos_att_type ["p2c", "c2p"], and all
    // three terms are divided by the same sqrt(head_dim * scale_factor). That
    // shared scale is why this fits build_attn's single kq_scale.
    const float kq_scale = 1.0f / std::sqrt(3.0f * (float) n_embd_head);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * cur = inpL;

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // share_att_key: the position stream reuses the content q/k
            // projections, so there are no separate pos_q / pos_k WEIGHTS.
            // The biases, however, are still the q/k biases and they DO apply.
            ggml_tensor * wq = model.layers[il].wq;
            ggml_tensor * wk = model.layers[il].wk;

            // The reference computes pos_key_layer = key_proj(rel_embeddings)
            // and pos_query_layer = query_proj(rel_embeddings). Both are
            // nn.Linear(..., bias=True), so the bias must be added. Omitting it
            // here is silent: the shapes are all still valid and the bias term
            // is simply wrong, which makes the relative term uncorrelated with
            // the reference rather than merely imprecise. Found by dumping the
            // graph's own pos_k tensor (examples/rel-dump) and comparing
            // against the reference: the values were unrelated at every head,
            // with no fixed ratio and no head permutation that reconciled them.
            ggml_tensor * pos_q = ggml_mul_mat(ctx0, wq, rel_e);   // [n_embd, n_rows]
            ggml_tensor * pos_k = ggml_mul_mat(ctx0, wk, rel_e);   // [n_embd, n_rows]
            if (model.layers[il].wq_b) {
                pos_q = ggml_add(ctx0, pos_q,
                        ggml_repeat(ctx0, model.layers[il].wq_b, pos_q));
            }
            if (model.layers[il].wk_b) {
                pos_k = ggml_add(ctx0, pos_k,
                        ggml_repeat(ctx0, model.layers[il].wk_b, pos_k));
            }
            cb(pos_q, "pos_q", il);
            cb(pos_k, "pos_k", il);

            // c2p contracts the QUERY against the position, so q is broadcast
            // along the key axis. build_qkv already returns Qcur as
            // [n_embd_head, n_head, n_tokens], which is exactly the order the
            // helper wants, so this reshape is free.
            ggml_tensor * q4 = ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_tokens, 1);
            ggml_tensor * c2p = build_deberta_rel_dot(ctx0, pos_k, rel_idx, q4);
            cb(c2p, "rel_c2p", il);

            // p2c contracts the KEY against the position, so k is broadcast
            // along the query axis instead: [n_embd_head, n_head, 1, n_tokens].
            //
            // It reuses rel_idx, NOT clamp(-bucket + span). In the source the
            // gather indexes a key-major tensor [bh, key, bucket] with an array
            // whose leading axis is the query, so the negation and the transpose
            // cancel and the effective index is c2p_pos[query, key]. Confirmed
            // against the reference: transcribing the p2c expression literally
            // and the shared-grid form are bit-identical, and both reproduce
            // HF's rel_att exactly (~/work/deberta_rel_probe.py).
            // build_qkv already returns [n_embd_head, n_head, n_tokens], which is exactly the order the
            // helper wants, so this is a free reshape -- no swap.
            ggml_tensor * k4 = ggml_reshape_4d(ctx0, Kcur, n_embd_head, n_head, 1, n_tokens);
            ggml_tensor * p2c_g = build_deberta_rel_dot(ctx0, pos_q, rel_idx, k4);
            cb(p2c_g, "rel_p2c", il);

            ggml_tensor * kq_b = ggml_add(ctx0, c2p, p2c_g);
            cb(kq_b, "kq_rel_bias", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, model.layers[il].wo_b, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, kq_b, nullptr, nullptr, kq_scale, il);
            cb(cur, "kqv_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        // post-LN: norm(attn_out + residual)
        cur = ggml_add(ctx0, cur, inpL);
        cur = build_norm(cur,
                model.layers[il].attn_out_norm, model.layers[il].attn_out_norm_b,
                LLM_NORM, il);
        cb(cur, "attn_out_norm", il);

        ggml_tensor * ffn_inp = cur;

        cur = build_ffn(cur,
                model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL,
                NULL, NULL, NULL,
                model.layers[il].ffn_down, model.layers[il].ffn_down_b, NULL, NULL,
                LLM_FFN_GELU, LLM_FFN_SEQ, il);
        cb(cur, "ffn_out", il);

        // post-LN: norm(ffn_out + residual)
        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_norm(cur,
                model.layers[il].layer_out_norm, model.layers[il].layer_out_norm_b,
                LLM_NORM, il);
        cb(cur, "ffn_out_norm", il);

        inpL = cur;
    }

    // NO final norm. DebertaV2Encoder.forward returns `output_states` straight
    // after the layer loop; the encoder's single LayerNorm exists only to
    // implement norm_rel_ebd and is applied to rel_embd, never to the hidden
    // states. `output_norm` in the GGUF is therefore the rel-embedding norm,
    // not an output norm, and applying it here scaled the result (a 1-token
    // input, where attention is a no-op, came back with norm 10.41 against
    // torch's 26.82).
    ggml_tensor * cur = inpL;

    // GLiNER: one logit per token. The server reads it at each [L] marker.
    // ReLU, not GELU. The encoder FFN is GELU; this head is not.
    if (model.cls && model.cls_out) {
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, model.cls, cur), model.cls_b);
        cur = ggml_relu(ctx0, cur);
        ggml_tensor * scorer = ggml_reshape_2d(ctx0, model.cls_out, model.cls_out->ne[0], 1);
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, scorer, cur), model.cls_out_b);
        cb(cur, "decision_scores", -1);
    } else {
        cb(cur, "output", -1);
    }

    res->t_embd = cur;

    // Hand ownership of the relative-position index input to the graph context.
    res->add_input(std::move(inp_rel));

    ggml_build_forward_expand(gf, cur);
}
