# llama.phi

A fork of llama.cpp (branch `phi`, from upstream c479922ac, 2026-10-06)
whose server keeps prompt processing and token generation apart: two
engines in flight at once on one model, so a new request's prompt never
stalls the tokens another request is receiving. Everything else is
upstream, unchanged; the fork's own files are `tools/server/phi-prefill.*`,
one hook in `tools/server/server-context.cpp`, `tools/phi-twoctx`,
`scripts/phi-serve.sh`, `scripts/phi-smoke.sh` and `docs/phi/`. The fork
tracks upstream by rebasing them.

## Why

On the GPU rack (four RTX 3080 20 GB, four Xeon Phi 3120 cards, 251 GB)
with Qwen3.8 Flash Next at UD-Q6_K_XL, prompt processing is compute
bound (every token through every weight) and generation is bandwidth
bound (one token, the active weights streamed). Upstream llama-server
runs both through one context on one placement: every prompt chunk
sits in the same decode call as the live tokens, and the devices take
turns on each call. Decoupling them is a server design, not a kernel.

## How (`tools/server/phi-prefill.h`)

llama-server stays whole and is the decode side. Beside it, when
`PHI_PREFILL=1`, a **prefill engine** loads the same model file a second
time with its own placement and its own context, on its own thread.
A completion task whose prompt is long enough and not already held by a
slot or the prompt cache goes to the engine instead of a slot: the task
is deferred, the engine computes the prompt in big batches and exports
the sequence's state (`llama_state_seq_get_data_ext`), and the state is
put into the server's own prompt cache as that prompt's entry. The
deferred task then takes a slot, the slot restores the entry
(`server_prompt_cache::load`, upstream code), finds the whole prompt in
place, recomputes the last token for its logits and generates. Nothing
in the response path changed: every endpoint, streaming, tool parsing,
the Anthropic and OpenAI formats are upstream's.

The decode context's KV layout is mirrored by the engine (it copies the
server's `common_params` and changes only placement and batch sizes),
because a state imports only into the same layout: one KV stream
(`--kv-unified`), the same K and V types.

Environment, read once at start (each unset one keeps the server's
value): `PHI_PREFILL_TS` (tensor split, `a/b/c/d`), `PHI_PREFILL_OT`
(tensor buffer overrides, as `-ot`), `PHI_PREFILL_NGL`,
`PHI_PREFILL_BATCH`, `PHI_PREFILL_UBATCH`, `PHI_PREFILL_THREADS`,
`PHI_PREFILL_KV_HOST` (1: the engine's KV in host memory),
`PHI_PREFILL_MIN` (prompts shorter than this stay with the server, 64).
`--cache-ram` must be on (the states live in the prompt cache; `-1` for
no limit).

## The rack's placement (`scripts/phi-serve.sh`)

Decode: the whole model on the CPU (`-ngl 0`, 32 threads): 7.6 tok/s
measured against 8.0 with the GPUs, which the prefill model now fills.
Prefill: experts of blocks 0 to 23 on the four GPUs (`6/6/6/30`, room
for 2048-token compute buffers), blocks 24 to 47 and the per-layer
embeddings in host memory, batch 4096, ubatch 2048: 440 tok/s measured
on a 4096-token prompt (560 with the Q4_K_XL file and 30 blocks in
VRAM). The numbers and every command are in `docs/phi/results.md`.

```
scripts/phi-serve.sh              # port 8001, two slots
scripts/phi-smoke.sh              # a 6000-token prompt and a short request at once
```

## Build

```
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server
```
