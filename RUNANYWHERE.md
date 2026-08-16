# RunAnywhere llama.cpp maintenance notes

This fork stays rooted in the canonical `ggml-org/llama.cpp` repository. The
RunAnywhere integration branch carries only model/runtime compatibility that is
not yet available upstream:

- DeepGrove Maple model architecture and ternary expert support, ported from
  `deepgrove-ai/llama.cpp`.
- Prism-compatible `Q2_0` blocks (`QK2_0 = 128`) so existing PrismML ternary
  GGUF artifacts continue to load. Canonical llama.cpp currently assigns the
  same GGUF type ID to a 64-value block, so this is intentionally a wire-format
  compatibility choice and must be re-checked on every upstream rebase.

## Current backend status

- CPU: Maple, Bonsai `Q1_0`, and Prism `Q2_0` are supported.
- Metal: Bonsai `Q1_0` and Prism `Q2_0` are supported.
- Maple must use CPU execution. Its expert path requires a ternary
  `mul_mm_id` Metal pipeline that the DeepGrove patch does not provide.
- CUDA, Vulkan, SYCL, and WebGPU retain canonical llama.cpp's 64-value `Q2_0`
  kernels and must not be advertised for Prism `Q2_0` until their kernels are
  ported and tested. This does not affect `Q1_0` artifacts.

## Rebase and release checklist

1. Rebase the integration branch onto a pinned canonical llama.cpp revision.
2. Re-apply the small Maple and Prism compatibility commits; do not merge a
   third-party fork wholesale.
3. Build `llama-cli` on macOS with Metal and on Linux with the CPU backend.
4. Run real inference with the pinned Maple preview, Bonsai `Q1_0`, and Prism
   `Q2_0` GGUF fixtures recorded in the RunAnywhere SDK runtime plan.
5. Keep Maple CPU-only until a real Metal expert pipeline passes the same model
   smoke test.
6. Tag the exact tested commit and pin the SDK to that immutable tag or SHA.

The upstream remote for maintenance should always be
`https://github.com/ggml-org/llama.cpp.git`.
