#include "llama.h"
#include "common.h"

#include "../tools/mtmd/mtmd.h"
#include "../tools/server/server-decision.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// LiquidAI/d1-2B: load the GGUF and its mmproj, then answer one noul question.
// Not part of ctest. The weights are local files, passed as arguments or the defaults below.

static const char * k_model_default  = "/Users/home/Project/Wally-Work-Space/models/liquid/d1-2B-GGUF/d1-2B-Q4_K_M.gguf";
static const char * k_mmproj_default = "/Users/home/Project/Wally-Work-Space/models/liquid/d1-2B-GGUF/mmproj-d1-2B-Q8_0.gguf";

static void fail(const std::string & msg) {
    fprintf(stderr, "test-lfm3-d1: %s\n", msg.c_str());
    exit(1);
}

int main(int argc, char ** argv) {
    const char * model_path  = argc > 1 ? argv[1] : k_model_default;
    const char * mmproj_path = argc > 2 ? argv[2] : k_mmproj_default;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fail(std::string("failed to load ") + model_path);
    }
    if (common_get_decision_type(model) != COMMON_DECISION_TYPE_LFM3_D1) {
        fail("decision type is not lfm3-d1");
    }

    mtmd_context * mctx = mtmd_init_from_file(mmproj_path, model, mtmd_context_params_default());
    if (mctx == nullptr) {
        fail(std::string("failed to load ") + mmproj_path);
    }
    if (!mtmd_support_vision(mctx)) {
        fail("mmproj has no vision");
    }

    server_decision_context decision;
    try {
        decision.init(model);
    } catch (const std::exception & e) {
        fail(e.what());
    }
    if (decision.type != COMMON_DECISION_TYPE_LFM3_D1) {
        fail("server did not take the model as lfm3-d1");
    }

    const json body = {
        {"state", "The capital of France is Paris."},
        {"questions", {{"capital", {
            {"type", "noul"},
            {"instructions", "The capital of France is Paris."},
        }}}},
    };

    std::vector<server_decision_question> questions;
    try {
        questions = decision.parse_questions(body);
    } catch (const std::exception & e) {
        fail(e.what());
    }
    if (questions.size() != 1) {
        fail("expected one question");
    }

    server_task task(SERVER_TASK_TYPE_DECISION);
    const mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    try {
        decision.fill_task(body.at("state"), questions, questions[0], 0, {}, nullptr, init_opt, task);
    } catch (const std::exception & e) {
        fail(e.what());
    }
    if (task.tokens.size() < 2 || task.decision.labels.empty() || task.decision.label_groups.size() != 2) {
        fail("noul prompt did not produce two label groups");
    }

    const int32_t n_tokens = (int32_t) task.tokens.size();
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx    = n_tokens;
    cparams.n_batch  = n_tokens;
    cparams.n_ubatch = n_tokens;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fail("failed to create context");
    }
    if (llama_get_memory(ctx) != nullptr || llama_get_causal_attn(ctx)) {
        fail("lfm3-d1 context must be non-causal and keep no memory");
    }

    common_batch batch(ctx);
    for (size_t i = 0; i < task.tokens.size(); i++) {
        batch.add(task.tokens[i], (llama_pos) i, 0, i + 1 == task.tokens.size());
    }
    const int32_t rc = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
    if (rc != 0) {
        fail("encode failed: " + std::to_string(rc));
    }

    const float * logits = llama_get_logits_ith(ctx, (int32_t) task.tokens.size() - 1);
    if (logits == nullptr) {
        fail("no logits");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<float> scores;
    size_t i_label = 0;
    for (const int32_t n : task.decision.label_groups) {
        float best = -INFINITY;
        for (int32_t j = 0; j < n; j++) {
            const llama_token id = task.decision.labels[i_label++];
            if (id < 0 || id >= n_vocab) {
                fail("label token out of range");
            }
            best = std::max(best, logits[id]);
        }
        if (!std::isfinite(best)) {
            fail("label logit is not finite");
        }
        scores.push_back(best);
    }

    json answer;
    try {
        answer = decision.format_answer(questions[0], {scores});
    } catch (const std::exception & e) {
        fail(e.what());
    }
    if (!answer.contains("noul") || !answer.at("noul").is_number()) {
        fail("noul answer missing");
    }
    const double p_true = answer.at("noul").get<double>();
    fprintf(stderr, "test-lfm3-d1: noul P(true) = %.4f\n", p_true);
    if (!(p_true > 0.5)) {
        fail("expected the true statement to score above 0.5");
    }

    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
