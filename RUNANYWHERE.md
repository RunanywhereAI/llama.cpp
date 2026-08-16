# RunAnywhere llama.cpp maintenance notes

This is a direct fork of `ggml-org/llama.cpp`. The RunAnywhere integration
branch carries the DeepGrove Maple model architecture and ternary expert graph
from `deepgrove-ai/llama.cpp`; it does not wholesale-track that fork.

## Backend status

- CPU: Maple Preview, PrismML Bonsai `Q1_0`, and canonical `Q2_0` are supported.
- Metal: Bonsai `Q1_0` and canonical `Q2_0` are supported.
- Maple must use CPU execution. Its expert path requires a ternary
  `mul_mm_id` Metal pipeline that the current DeepGrove patch does not provide.
- Prism's older `Q2_0` artifacts used a non-canonical 128-value block under the
  same GGUF type ID. Do not change the global type to accommodate them. Use the
  newer `Q2_0_g64` artifacts Prism publishes for canonical llama.cpp instead;
  this preserves CUDA, Vulkan, Metal, SYCL, WebGPU, and conversion compatibility.

## Rebase and release checklist

1. Rebase the integration branch onto a pinned canonical llama.cpp revision.
2. Re-apply the two small Maple commits and resolve only their architecture
   registration/model graph surface.
3. Build `llama-cli` on macOS with Metal and on Linux with the CPU backend.
4. Run real inference with the pinned Maple Preview, Bonsai `Q1_0`, and Prism
   `Q2_0_g64` GGUF fixtures recorded in the RunAnywhere SDK runtime plan.
5. Keep Maple CPU-only until a real GPU expert pipeline passes the same model
   smoke test.
6. Tag the exact tested commit and pin the SDK to that immutable tag or SHA.

The upstream remote for maintenance should always be
`https://github.com/ggml-org/llama.cpp.git`.
