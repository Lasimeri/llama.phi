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
