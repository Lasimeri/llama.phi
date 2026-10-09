#!/bin/bash
# phi-serve.sh: llama.phi on the rack, two servers (docs/phi/README.md).
#
#   the prefill server   llama-server on the four GPUs, big batches, port
#                        PHI_PREFILL_PORT (8002), reached only by the decode
#                        server's engine; it saves each finished prompt's
#                        state to PHI_PREFILL_DIR
#   the decode server    llama-server on the CPU and the Xeon Phis (the
#                        AVX-512 backend, generation steps only), port
#                        PHI_PORT (8001): the one clients talk to. Its
#                        engine (PHI_PREFILL=1) hands every long prompt to
#                        the prefill server and restores the state into its
#                        prompt cache, so a slot generates at once.
#
#   phi-serve.sh                start both in the foreground; Ctrl-C stops both
#   phi-serve.sh -np 4 ...      options after the script name go to the decode server
#   phi-serve.sh plan [MODEL]   print the prefill model's placement for MODEL
#                               (default PHI_MODEL) and start nothing
#   PHI_CARDS=0 phi-serve.sh    decode on the CPU alone (32 threads), no cards
#   PHI_ENGINE=one phi-serve.sh one server instead of the two: prompts and generation in one
#                               batch stream (GPUs, CPU, cards), the placement the same
#
# The prefill model's placement follows the model file: scripts/phi-gguf.c
# reads the shards' headers for the bytes of every block's experts and
# other weights, nvidia-smi gives each GPU's free memory, and the plan fills
# the GPUs in order with whole blocks (experts and all), the last GPU with
# the remaining blocks' non-expert weights, the output and as many blocks'
# experts as still fit; the experts of the blocks left over stay in host
# memory (-ot). PHI_PREFILL_TS and PHI_PREFILL_OT set by hand override it.
# Headroom, per GPU: PHI_PREFILL_RESERVE_MIB (2355: the prefill server's
# compute buffers and CUDA context, measured 2238 MiB on a GPU holding 5
# blocks at ubatch 2048), one block's experts more on GPU 0 (the host
# blocks' experts are staged there for each batch: 4908 MiB measured
# there), the KV cache's share when it is on the GPUs (PHI_PREFILL_NKVO=0:
# the context's K and V, q8_0, over the attention blocks the GPU holds),
# PHI_PREFILL_MARGIN_MIB (1024) on top, and the decode server's CUDA
# context (PHI_DECODE_GPU0_MIB 1700, PHI_DECODE_GPU_MIB 450) when it is not
# running yet.
#
# Logs: $PHI_LOG/phi-prefill.log and phi-decode.log (default /mnt/raid5/phi/bench).
# Every number behind a default is in docs/phi/results.md.
set -u
root=$(cd "$(dirname "$0")/.." && pwd)
srv=$root/build/bin/llama-server
plan_only=0
if [ "${1:-}" = plan ]; then
    plan_only=1
    shift
    [ -n "${1:-}" ] && { PHI_MODEL=$1; shift; }
fi
[ "$plan_only" = 1 ] || [ -x "$srv" ] || { echo "$0: $srv not built (cmake --build build -j)" >&2; exit 1; }
M=${PHI_MODEL:-/mnt/raid0/Qwen3.8-Flash-Next-GGUF/Q8_0/Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf}
[ -f "$M" ] || { echo "$0: no model at $M (PHI_MODEL)" >&2; exit 1; }
avx=${PHI_AVX512:-/mnt/raid5/phi/Intel Phi AVX-512}
log=${PHI_LOG:-/mnt/raid5/phi/bench}
dir=${PHI_PREFILL_DIR:-/mnt/raid0/phi-prefill}
pport=${PHI_PREFILL_PORT:-8002}
port=${PHI_PORT:-8001}
ctx=${PHI_CTX:-131072}
cards=${PHI_CARDS:-1}

# Both contexts must share one KV layout: the state saved by one imports
# only into the other's (one stream, Q8 K and V, flash attention).
common=(-m "$M" -c "$ctx" --kv-unified -fa on -ctk q8_0 -ctv q8_0 --jinja)

# The prefill server: its KV on the GPUs (PHI_PREFILL_NKVO=0, the default)
# or in host memory (1, -nkvo). On the GPUs the attention of every prompt
# chunk runs there too: 232 against 211 tok/s over 49152 tokens, and the
# decode server's generation beside a prefill keeps 93 percent of its rate
# instead of 51 (with -nkvo the prefill's 24 host threads run the attention
# over host memory, against the generation's threads and bus;
# docs/phi/results.md 2026-10-08). The cost: 136 MiB of GPU memory per
# attention block for the 131072-token context, which the plan below counts.
PHI_PREFILL_NGL=${PHI_PREFILL_NGL:-999}
PHI_PREFILL_BATCH=${PHI_PREFILL_BATCH:-4096}
PHI_PREFILL_UBATCH=${PHI_PREFILL_UBATCH:-2048}
PHI_PREFILL_THREADS=${PHI_PREFILL_THREADS:-24}
PHI_PREFILL_NKVO=${PHI_PREFILL_NKVO:-0}
nkvo=()
[ "$PHI_PREFILL_NKVO" != 0 ] && nkvo=(-nkvo)

# The placement: the model's bytes by block (phi-gguf, built from
# scripts/phi-gguf.c on first use), the GPUs' free memory (what a running
# prefill server holds counts as free: the plan is for its replacement).
plan() {
    local gguf=$root/build/bin/phi-gguf
    if [ ! -x "$gguf" ] || [ "$gguf" -ot "$root/scripts/phi-gguf.c" ]; then
        mkdir -p "$root/build/bin"
        ${CC:-cc} -O2 -o "$gguf" "$root/scripts/phi-gguf.c" || return 1
    fi
    local shards=("$M")
    case "$M" in
        *-of-*.gguf) shards=("${M%-*-of-*.gguf}"-*-of-*.gguf) ;;
    esac
    local gpus
    gpus=$(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || { echo "phi-serve: no nvidia-smi: set PHI_PREFILL_TS and PHI_PREFILL_OT" >&2; return 1; }
    local held=""
    local ppid
    ppid=$(pgrep -f "llama-server.*--port $pport" | head -1)
    if [ -n "$ppid" ]; then
        held=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v p="$ppid" '$1 == p {print $2 "," $3}')
        held=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | tr -d ' ' | awk -F, -v h="$held" 'BEGIN {n = split(h, a, "\n"); for (i = 1; i <= n; i++) {split(a[i], b, ","); u[b[1]] = b[2]}} {print $1 "," (u[$2] + 0)}')
    fi
    # The decode server sees no GPU (below) unless PHI_DECODE_CUDA=1; then it
    # opens a CUDA context on every GPU even at -ngl 0 (about 1700 MiB on GPU
    # 0, 450 on the others, measured 2026-10-08 with GPU 0 left 883 MiB free;
    # more with a larger model's compute buffers), reserved when it is not yet
    # running (once it runs, nvidia-smi already counts it).
    # PHI_DECODE_GPU0_MIB, PHI_DECODE_GPU_MIB.
    local dres0=0 dres=0
    if [ "${PHI_DECODE_CUDA:-0}" = 1 ] && ! pgrep -f "llama-server.*--port $port( |$)" > /dev/null; then
        dres0=${PHI_DECODE_GPU0_MIB:-1700}; dres=${PHI_DECODE_GPU_MIB:-450}
    fi
    "$gguf" "${shards[@]}" | awk -v gpus="$gpus" -v held="$held" -v dres0="$dres0" -v dres="$dres" -v ctx="$ctx" -v kv_on_gpu="$([ "$PHI_PREFILL_NKVO" = 0 ] && echo 1 || echo 0)" \
        -v reserve="${PHI_PREFILL_RESERVE_MIB:-2355}" -v margin="${PHI_PREFILL_MARGIN_MIB:-1024}" -v model="$M" '
    function mib(b) { return b / 1048576 }
    /^blocks/ { nb = $2 }
    /^block / { E[$2] = $4; D[$2] = $6; A[$2] = $8 }
    /^kv_width/ { kvw = $2 }
    /^other/ { if ($2 == "output.weight") out = $3; else if ($2 ~ /per_layer_token_embd/) ple = $3; else if ($2 ~ /^token_embd/) emb = $3 }
    END {
        ng = split(gpus, g, "\n")
        nh = split(held, h, "\n")
        for (i = 1; i <= nh; i++) { split(h[i], t, ","); keep[t[1]] = t[2] }
        for (i = 1; i <= ng; i++) { split(g[i], t, ","); idx[i] = t[1]; free[i] = t[2] - t[3] + keep[t[1]] - (t[1] == 0 ? dres0 : dres) }
        # the KV cache of one attention block, the whole context, q8_0 K and V
        kv_block = kv_on_gpu ? mib(ctx * 2 * kvw * 34 / 32) : 0
        # every block equal in practice; the largest decides
        e = 0; d = 0; for (b = 0; b < nb; b++) { if (E[b] > e) e = E[b]; if (D[b] > d) d = D[b] }
        e_mib = mib(e); d_mib = mib(d)
        # GPUs 0 .. n-2: whole blocks, in order
        b = 0
        for (i = 1; i < ng; i++) {
            cap = free[i] - reserve - margin - (i == 1 ? e_mib : 0)
            n = 0
            while (b + n < nb && (n + 1) * (e_mib + d_mib) + kv_block * attn_count(b, n + 1) <= cap) n++
            nfull[i] = n; first[i] = b; b += n
        }
        # the last GPU: the rest of the dense weights and the output, then experts
        rest = nb - b
        cap = free[ng] - reserve - margin - rest * d_mib - mib(out) - kv_block * attn_count(b, rest)
        k = 0
        while (b + k < nb && (k + 1) * e_mib <= cap) k++
        nfull[ng] = rest; first[ng] = b; kexp = k
        host_from = b + k
        ts = ""; for (i = 1; i <= ng; i++) ts = ts (i > 1 ? "/" : "") nfull[i]
        pat = ""
        for (x = host_from; x < nb; x++) pat = pat (x > host_from ? "|" : "") x
        ot = (host_from < nb ? "blk\\.(" pat ")\\.ffn_(up|gate|down)_exps\\.weight=CPU," : "") "per_layer_token_embd\\.weight=CPU"
        printf "phi-serve: placement for %s\n", model > "/dev/stderr"
        printf "phi-serve:   %d blocks; a block: experts %.3f GB, other weights %.3f GB; output %.3f GB; per-layer embeddings %.2f GB (host)\n", nb, e / 1e9, d / 1e9, out / 1e9, ple / 1e9 > "/dev/stderr"
        for (i = 1; i <= ng; i++) {
            if (i < ng) { kx = nfull[i]; what = sprintf("blocks %d to %d whole", first[i], first[i] + nfull[i] - 1) }
            else { kx = kexp; what = sprintf("blocks %d to %d without experts, the output, the experts of blocks %d to %d", first[i], nb - 1, first[i], first[i] + kexp - 1) }
            if (i == ng && kexp == 0) what = sprintf("blocks %d to %d without experts, the output", first[i], nb - 1)
            use = kx * e_mib + nfull[i] * d_mib + (i == ng ? mib(out) : 0) + reserve + (i == 1 ? e_mib : 0) + kv_block * attn_count(first[i], nfull[i])
            printf "phi-serve:   GPU %s: %5.0f MiB free, %5.0f planned (%s)\n", idx[i], free[i], use, what > "/dev/stderr"
        }
        onhost = (nb - host_from) * e
        printf "phi-serve:   host: the experts of blocks %d to %d, %.1f GB, and the per-layer embeddings, %.1f GB\n", host_from, nb - 1, onhost / 1e9, ple / 1e9 > "/dev/stderr"
        printf "phi-serve:   -ts %s -ot %s\n", ts, ot > "/dev/stderr"
        printf "%s\n%s\n", ts, ot
    }
    function attn_count(from, n,    c, x) { c = 0; for (x = from; x < from + n; x++) c += A[x]; return c }'
}
if [ -z "${PHI_PREFILL_TS:-}" ] || [ -z "${PHI_PREFILL_OT:-}" ]; then
    planned=$(plan) || { echo "$0: the placement could not be planned; set PHI_PREFILL_TS and PHI_PREFILL_OT" >&2; exit 1; }
    PHI_PREFILL_TS=${PHI_PREFILL_TS:-$(echo "$planned" | sed -n 1p)}
    PHI_PREFILL_OT=${PHI_PREFILL_OT:-$(echo "$planned" | sed -n 2p)}
else
    echo "phi-serve: placement from the environment: -ts $PHI_PREFILL_TS -ot $PHI_PREFILL_OT"
fi
[ "$plan_only" = 1 ] && exit 0

# One engine (PHI_ENGINE=one): a single server on PHI_PORT instead of the
# two, the prefill server's placement (the GPUs hold the blocks the plan
# gives them, every block's attention and the KV; the experts past them
# stay in host memory) and llama-server's continuous batching, so a
# prompt's chunks and the slots' generated tokens share its batches and
# neither waits for a hand-off. With the cards (PHI_CARDS not 0) the
# backend takes generation steps alone (PHI_GGML_TG_ONLY, a step of up to
# that many tokens), the host-memory weights' rows split across every
# card (no PHI_GGML_EXPERTS: each multiply divided card by card); prompt
# chunks stay with the GPUs and the host. -t is the generation's threads
# (12 beside the cards, 32 without), -tb the prompt's (PHI_PREFILL_THREADS).
if [ "${PHI_ENGINE:-two}" = one ]; then
    mkdir -p "$log" || exit 1
    slots=${PHI_SLOTS:-2}
    cache=${PHI_CACHE_RAM:-24576}
    one=("$srv" "${common[@]}" -np "$slots" --cache-ram "$cache" \
        -ngl "$PHI_PREFILL_NGL" -ts "$PHI_PREFILL_TS" -ot "$PHI_PREFILL_OT" \
        -b "$PHI_PREFILL_BATCH" -ub "$PHI_PREFILL_UBATCH" "${nkvo[@]}" \
        --host 0.0.0.0 --port "$port")
    if [ "$cards" != 0 ]; then
        t=${PHI_DECODE_THREADS:-12}
        export PHI_GGML_TG_ONLY=${PHI_GGML_TG_ONLY:-$slots}
        echo "phi-serve: one engine: prompts on the GPUs, generation on the GPUs, the CPU ($t threads) and the cards, port $port, log $log/phi-engine.log"
        exec "$avx/scripts/phi-ggml.sh" "${one[@]}" -t "$t" -tb "$PHI_PREFILL_THREADS" "$@" > "$log/phi-engine.log" 2>&1
    fi
    t=${PHI_DECODE_THREADS:-32}
    echo "phi-serve: one engine: prompts on the GPUs, generation on the GPUs and the CPU ($t threads), port $port, log $log/phi-engine.log"
    exec env -u GGML_BACKEND_PATH "${one[@]}" -t "$t" -tb "$PHI_PREFILL_THREADS" "$@" > "$log/phi-engine.log" 2>&1
fi

mkdir -p "$dir" "$log" || exit 1
rm -f "$dir"/phi-prefill-*.bin

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
# Its weights and KV are on the host (-ngl 0) and every prompt of
# PHI_PREFILL_MIN tokens or more is read by the prefill server, so the GPUs
# do nothing for it: they are hidden from it (no CUDA context, no compute
# buffer on them; with BF16 its 2070 MiB buffer on GPU 0 did not fit beside
# the prefill server, 2026-10-08). PHI_DECODE_CUDA=1 shows them again.
hide=(env CUDA_VISIBLE_DEVICES=)
[ "${PHI_DECODE_CUDA:-0}" = 1 ] && hide=()
if [ "$cards" != 0 ]; then
    # The cards take the generation steps' multiplies (PHI_GGML_TG_ONLY=K: up
    # to K tokens, one per slot generating at once), the host 12 threads
    # beside them (the backend's rule: more oversubscribes).
    t=${PHI_DECODE_THREADS:-12}
    export PHI_GGML_TG_ONLY=${PHI_GGML_TG_ONLY:-$slots}
    echo "phi-serve: decode server on the CPU ($t threads) and the cards, port $port, log $log/phi-decode.log"
    "${hide[@]}" "$avx/scripts/phi-ggml.sh" "${decode[@]}" -t "$t" -tb "$t" "$@" 2>&1 | tee "$log/phi-decode.log"
else
    t=${PHI_DECODE_THREADS:-32}
    echo "phi-serve: decode server on the CPU ($t threads), no cards, port $port, log $log/phi-decode.log"
    "${hide[@]}" "${decode[@]}" -t "$t" -tb "$t" "$@" 2>&1 | tee "$log/phi-decode.log"
fi
