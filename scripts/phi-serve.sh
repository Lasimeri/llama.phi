#!/bin/bash
# phi-serve.sh: llama.phi on the rack, two servers (docs/phi/README.md).
#
#   the prefill server   llama-server on the four GPUs, big batches, port
#                        PHI_PREFILL_PORT (8002), reached only by the decode
#                        server's engine; it saves each finished prompt's
#                        state to PHI_PREFILL_DIR
#   the decode server    llama-server on the CPU and the four Xeon Phis (the
#                        AVX-512 backend, generation steps only), port
#                        PHI_PORT (8001): the one clients talk to. Its
#                        engine (PHI_PREFILL=1) hands every long prompt to
#                        the prefill server and restores the state into its
#                        prompt cache, so a slot generates at once.
#
#   phi-serve.sh                start both in the foreground; Ctrl-C stops both
#   phi-serve.sh -np 4 ...      options after the script name go to the decode server
#   PHI_CARDS=0 phi-serve.sh    decode on the CPU alone (32 threads), no cards
#
# Logs: $PHI_LOG/phi-prefill.log and phi-decode.log (default /mnt/raid5/phi/bench).
# Every number behind a default is in docs/phi/results.md.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
srv=$root/build/bin/llama-server
[ -x "$srv" ] || { echo "$0: $srv not built (cmake --build build -j)" >&2; exit 1; }
M=${PHI_MODEL:-/mnt/raid0/Qwen3.8-Flash-Next-GGUF/UD-Q6_K_XL/Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf}
[ -f "$M" ] || { echo "$0: no model at $M (PHI_MODEL)" >&2; exit 1; }
avx=${PHI_AVX512:-/mnt/raid5/phi/Intel Phi AVX-512}
log=${PHI_LOG:-/mnt/raid5/phi/bench}
dir=${PHI_PREFILL_DIR:-/mnt/raid0/phi-prefill}
pport=${PHI_PREFILL_PORT:-8002}
port=${PHI_PORT:-8001}
ctx=${PHI_CTX:-131072}
cards=${PHI_CARDS:-1}
mkdir -p "$dir" "$log" || exit 1
rm -f "$dir"/phi-prefill-*.bin

# Both contexts must share one KV layout: the state saved by one imports
# only into the other's (one stream, Q8 K and V, flash attention).
common=(-m "$M" -c "$ctx" --kv-unified -fa on -ctk q8_0 -ctv q8_0 --jinja)

# The prefill server: experts of blocks 0..23 on the GPUs (room left for
# 2048-token compute buffers), the rest and the per-layer embeddings in host
# memory, its KV in host memory, one slot (the engine's), states saved to $dir.
PHI_PREFILL_TS=${PHI_PREFILL_TS:-6/6/6/30}
PHI_PREFILL_OT=${PHI_PREFILL_OT:-'blk\.(2[4-9]|3[0-9]|4[0-7])\.ffn_(up|gate|down)_exps\.weight=CPU,per_layer_token_embd\.weight=CPU'}
PHI_PREFILL_NGL=${PHI_PREFILL_NGL:-999}
PHI_PREFILL_BATCH=${PHI_PREFILL_BATCH:-4096}
PHI_PREFILL_UBATCH=${PHI_PREFILL_UBATCH:-2048}
PHI_PREFILL_THREADS=${PHI_PREFILL_THREADS:-24}
# Its KV: in host memory (-nkvo, PHI_PREFILL_NKVO=1) or on the GPUs (0): the
# attention of every prompt chunk runs where the KV is (docs/phi/results.md,
# 2026-10-08, the rate deep in context).
PHI_PREFILL_NKVO=${PHI_PREFILL_NKVO:-1}
nkvo=()
[ "$PHI_PREFILL_NKVO" != 0 ] && nkvo=(-nkvo)
if curl -sf "http://127.0.0.1:$pport/health" > /dev/null; then
    # A prefill server from an earlier start keeps its model loaded; it is
    # reused (and left running at exit), so the decode side can be restarted alone.
    echo "phi-serve: a prefill server already answers on port $pport; reused, not stopped at exit"
else
    echo "phi-serve: prefill server on the GPUs, port $pport, log $log/phi-prefill.log"
    env -u GGML_BACKEND_PATH "$srv" "${common[@]}" -np 1 --slot-save-path "$dir/" \
        -ngl "$PHI_PREFILL_NGL" -ts "$PHI_PREFILL_TS" -ot "$PHI_PREFILL_OT" \
        -b "$PHI_PREFILL_BATCH" -ub "$PHI_PREFILL_UBATCH" -t "$PHI_PREFILL_THREADS" -tb "$PHI_PREFILL_THREADS" "${nkvo[@]}" \
        --host 127.0.0.1 --port "$pport" > "$log/phi-prefill.log" 2>&1 &
    pre=$!
    trap 'kill $pre 2>/dev/null; wait $pre 2>/dev/null' EXIT
    until curl -sf "http://127.0.0.1:$pport/health" > /dev/null; do
        kill -0 $pre 2>/dev/null || { echo "phi-serve: the prefill server died; tail of its log:" >&2; tail -5 "$log/phi-prefill.log" >&2; exit 1; }
        sleep 2
    done
    echo "phi-serve: prefill server up"
fi

# The decode server: the model on the CPU (-ngl 0), one KV stream, the engine
# in client form, the prompt cache capped (it carries the prefilled states
# and every slot's state saved before it loads one: 22 KiB a token, 2.1 GiB
# at 99k tokens; the oldest entry goes when the cap is reached, and an entry
# that is a prefix of a newer one goes at once).
export PHI_PREFILL=1
export PHI_PREFILL_URL="http://127.0.0.1:$pport"
export PHI_PREFILL_DIR="$dir"
export PHI_PREFILL_MIN=${PHI_PREFILL_MIN:-64}
slots=${PHI_SLOTS:-2}
cache=${PHI_CACHE_RAM:-24576}
decode=("$srv" "${common[@]}" -ngl 0 -np "$slots" --cache-ram "$cache" --host 0.0.0.0 --port "$port")
if [ "$cards" != 0 ]; then
    # The cards take the generation steps' multiplies (PHI_GGML_TG_ONLY=K: up
    # to K tokens, one per slot generating at once), the host 12 threads
    # beside them (the backend's rule: more oversubscribes).
    t=${PHI_DECODE_THREADS:-12}
    export PHI_GGML_TG_ONLY=${PHI_GGML_TG_ONLY:-$slots}
    echo "phi-serve: decode server on the CPU ($t threads) and the cards, port $port, log $log/phi-decode.log"
    "$avx/scripts/phi-ggml.sh" "${decode[@]}" -t "$t" -tb "$t" "$@" 2>&1 | tee "$log/phi-decode.log"
else
    t=${PHI_DECODE_THREADS:-32}
    echo "phi-serve: decode server on the CPU ($t threads), no cards, port $port, log $log/phi-decode.log"
    "${decode[@]}" -t "$t" -tb "$t" "$@" 2>&1 | tee "$log/phi-decode.log"
fi
