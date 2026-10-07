# llama.phi

A fork of llama.cpp (branch `phi`, from upstream c479922ac, 2026-10-06)
whose server keeps prompt processing and token generation apart: two
engines in flight at once on one model, so a new request's prompt never
stalls the tokens another request is receiving. Everything else is
upstream, unchanged; `tools/phi-server` and `tools/phi-twoctx` are the
fork's own, and the fork tracks upstream by rebasing them.

## Why

On the GPU rack (four RTX 3080 20 GB, four Xeon Phi 3120 cards, 251 GB)
with Qwen3.8 Flash Next at UD-Q6_K_XL, prompt processing is compute
bound (every token through every weight; the GPUs and the cards) and
generation is bandwidth bound (one token, the active weights streamed).
Upstream llama-server runs both through one context: every prompt
chunk sits in the same decode call as the live tokens, and the devices
take turns on each call. Decoupling them is a server design, not a
kernel: a prefill context and a decode context on the same model, each
on its own thread.

## Design (`tools/phi-server`)

- **One model, two contexts.** The prefill context takes prompts in
  chunks (`-ub`), its KV cache in host memory; the decode context runs
  continuous batching over every sequence whose prompt is done. The
  cards' backend (Intel-Phi-AVX512, `PHI_GGML_PP_ONLY=1`) serves the
  prefill context's host blocks; the GPUs serve both.
- **Handoff.** A finished prompt's state goes from the prefill context
  to a free decode slot with `llama_state_seq_get_data` and
  `llama_state_seq_set_data` (for Flash Next about 24 KB a token at
  Q8 K and V plus the recurrent state, so a 10k-token prompt moves in
  under a second).
- **Threads.** The prefill thread owns the prefill context; the decode
  thread owns the decode context; a request queue between them and a
  response channel out. The ggml backends are entered from two threads
  at once: CUDA on its own streams, the cards' backend serialized at
  the multiply, so a decode step slips between a prompt's multiplies.
  `tools/phi-twoctx` measured whether that holds (`docs/phi/results.md`).
- **Protocol.** `/completion`, `/v1/completions` and
  `/v1/chat/completions` as upstream's server answers them (streaming
  included), plus `/health` and `/slots`; the pieces of
  `tools/server` are reused where they are headers, never edited.

## Build

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target phi-server phi-twoctx
```

Run through the cards: `Intel Phi AVX-512/scripts/phi-ggml.sh build/bin/phi-server ...`.
