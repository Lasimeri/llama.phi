# llama.phi: measurements

## 2026-10-06: two contexts at once, and the handoff (`tools/phi-twoctx`)

Rack: EPYC 7742, four RTX 3080 20 GB, four Xeon Phi 3120 (the cards'
backend from Intel-Phi-AVX512 with `PHI_GGML_PP_ONLY=1`, so the cards
took the prompt's host-block multiplies and never a generation step).
Model: Qwen3.8 Flash Next UD-Q6_K_XL, experts of blocks 0 to 27 on the
GPUs (`tensor_split` 7/7/7/28), blocks 28 to 47 and the per-layer
embeddings in host memory, both contexts' KV in host memory, Q8 K and V,
one stream (`kv_unified`), 12 threads each. Prompt: 6034 tokens of
code, in 512-token chunks; generation greedy.

```
twoctx MODEL calib-code.txt 64 512      (twoctx2.log, twoctx2.out)
```

| | alone | together |
| --- | --- | --- |
| context A, prefill, tok/s | 172.8 | 168.6 |
| context B, generation, tok/s | 11.00 | 8.28 |

Together: A's whole prefill ran on one thread while B generated 64
tokens on another, on the same model and the same GPUs. The prefill
lost 2 percent; the generation kept 75 percent of its rate while a full
prompt ran beside it (the first run of the pair, before the one-stream
fix, kept 87: 9.26 against 10.67). B's text was identical alone and
together.

Handoff: A's finished sequence exported with `llama_state_seq_get_data`
(216,804,420 bytes for 6034 tokens: 35.9 KB a token, Q8 K and V plus
the recurrent state) and imported into B as a second sequence with
`llama_state_seq_set_data` in 0.081 s; B then continued the prompt at
9.52 tok/s with a correct continuation. The import fails with
`n_stream mismatch` unless both contexts have the same stream count:
`kv_unified = true` on both gives one stream whatever `n_seq_max` is.

What this settles: one process, one model, a prefill context and a
decode context on their own threads, is sound on this build (llama.cpp
c479922ac) with CUDA and the cards' backend entered from two threads.
The server design in `README.md` stands. What the next measurements
decide is the placement of each context's model: the prefill one for
the GPUs at large batches, the decode one for the CPU and the cards.

## 2026-10-06, later: the placement of each side (llama-bench, stock build)

Prompts on the GPUs alone, batch 4096, ubatch 2048, KV in host memory,
op offload on, the per-layer embeddings memory-mapped:

| prefill model | blocks with experts in VRAM | pp4096 tok/s |
| --- | --- | --- |
| UD-Q6_K_XL, `-ts 6/6/6/30` | 0 to 23 | 440.1, 453.0 |
| UD-Q4_K_XL, `-ts 8/8/8/24` | 0 to 29 | 560.5 |

(With 28 Q6 blocks or 34 Q4 blocks the 2048-token compute buffers did
not fit: `cudaMalloc failed`.) Generation on the CPU, no GPU:

| decode side | tg64 tok/s |
| --- | --- |
| CPU, 32 threads | 7.62 |
| CPU, 64 threads | 7.25 |
| CPU, 12 threads, plus the four cards (row share, `PHI_GGML_PP_ONLY=0`) | 7.12 |
| (the shared placement of the morning, GPUs and host) | 7.6 to 8.0 |

So the decode side loses 5 percent by leaving the GPUs, and the GPUs
then take the prompts five times faster. The cards do not help the
decode on this host (the row share costs more than it saves at one
token).

`-sm row` (llama.cpp's own tensor parallelism) does not exist in this
version for CUDA: `device CUDA0 does not support split buffers`. Its
micro-batch pipelining across the GPUs is off whenever the KV cache is
in host memory or any tensor override is set (`llama-context.cpp`,
`pipeline_parallel`), which this placement needs; the fork adds
`LLAMA_PHI_PIPELINE=1` to ask for it anyway, measured below.

## 2026-10-06, later: the server (`scripts/phi-smoke.sh`, the fork)

Decode on the CPU (32 threads, two slots, 131072 cells, Q8 K and V, one
KV stream), the prefill engine on the GPUs with the Q6 placement above.
A 6034-token prompt with 32 tokens to generate, and two seconds into
it a short request generating 48 tokens.

| | short request | long request | prefill engine |
| --- | --- | --- | --- |
| server reading its own prompts (first run, the state import failed) | 48 tokens at 6.9 tok/s, in flight during the prompt | 104.5 s wall: prompt 6034 tokens at 74.5 tok/s on the CPU, then 32 at 7.2 | 6034 tokens in 19.8 s (305 tok/s), unused |
| engine's state restored (last token left to the slot) | 48 tokens at 6.5 tok/s, in flight during the prompt | **23.8 s wall: prompt 1 token (181 ms), then 32 at 7.1** | 6034 tokens in 19.1 s (315 tok/s) |

Three defects met on the way, each fixed in the fork:
`tensor_buft_overrides` must end with a null entry (the conversion
asserts); the server's loop sleeps until a new task arrives, so a
finished prefill has to wake it (`pop_deferred_task`); and the slot
rewinds one position to recompute the last token's logits, which a
recurrent layer cannot do (the slot then redid the prompt), so the
engine prefills all but the last token.

Also found: llama-server never calls `ggml_backend_load_all`, so
`GGML_BACKEND_PATH` (the cards' backend) was ignored by it while
llama-bench honoured it; the fork adds the call.
