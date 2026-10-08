// phi-gguf.c: the bytes of a GGUF model by block, from the shards' headers
// alone (no tensor data is read), for scripts/phi-serve.sh's placement of
// the prefill model. Prints, one line each:
//   blocks N                       the number of blk.* layers
//   block I experts E dense D attn A   bytes of blk.I's expert tensors
//                                  (ffn_{up,gate,down}_exps) and of its
//                                  others; A = 1 when the block has a KV
//                                  cache (an attn_k weight)
//   kv_width W                     the K projection's width (the KV cache
//                                  holds W numbers for K and W for V a token
//                                  in every attention block)
//   other NAME BYTES               every tensor outside the blocks
//   total BYTES
// Usage: phi-gguf SHARD...   (every shard of the model, any order)
// Build: cc -O2 -o phi-gguf phi-gguf.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_BLOCKS 512
#define MAX_OTHER  64

static uint64_t experts[MAX_BLOCKS], dense[MAX_BLOCKS];
static int attn[MAX_BLOCKS];
static uint64_t kv_width;
static int n_blocks;
static char other_name[MAX_OTHER][128];
static uint64_t other_bytes[MAX_OTHER];
static int n_other;

// ggml type -> (bytes of one block, elements in one block)
static int type_size(uint32_t t, uint64_t * blck, uint64_t * bytes) {
    switch (t) {
        case 0:  *blck = 1;   *bytes = 4;   return 1;   // F32
        case 1:  *blck = 1;   *bytes = 2;   return 1;   // F16
        case 2:  *blck = 32;  *bytes = 18;  return 1;   // Q4_0
        case 3:  *blck = 32;  *bytes = 20;  return 1;   // Q4_1
        case 6:  *blck = 32;  *bytes = 22;  return 1;   // Q5_0
        case 7:  *blck = 32;  *bytes = 24;  return 1;   // Q5_1
        case 8:  *blck = 32;  *bytes = 34;  return 1;   // Q8_0
        case 9:  *blck = 32;  *bytes = 36;  return 1;   // Q8_1
        case 10: *blck = 256; *bytes = 84;  return 1;   // Q2_K
        case 11: *blck = 256; *bytes = 110; return 1;   // Q3_K
        case 12: *blck = 256; *bytes = 144; return 1;   // Q4_K
        case 13: *blck = 256; *bytes = 176; return 1;   // Q5_K
        case 14: *blck = 256; *bytes = 210; return 1;   // Q6_K
        case 15: *blck = 256; *bytes = 292; return 1;   // Q8_K
        case 16: *blck = 256; *bytes = 66;  return 1;   // IQ2_XXS
        case 17: *blck = 256; *bytes = 74;  return 1;   // IQ2_XS
        case 18: *blck = 256; *bytes = 98;  return 1;   // IQ3_XXS
        case 19: *blck = 256; *bytes = 50;  return 1;   // IQ1_S
        case 20: *blck = 32;  *bytes = 18;  return 1;   // IQ4_NL
        case 21: *blck = 256; *bytes = 110; return 1;   // IQ3_S
        case 22: *blck = 256; *bytes = 82;  return 1;   // IQ2_S
        case 23: *blck = 256; *bytes = 136; return 1;   // IQ4_XS
        case 24: *blck = 1;   *bytes = 1;   return 1;   // I8
        case 25: *blck = 1;   *bytes = 2;   return 1;   // I16
        case 26: *blck = 1;   *bytes = 4;   return 1;   // I32
        case 27: *blck = 1;   *bytes = 8;   return 1;   // I64
        case 28: *blck = 1;   *bytes = 8;   return 1;   // F64
        case 29: *blck = 256; *bytes = 56;  return 1;   // IQ1_M
        case 30: *blck = 1;   *bytes = 2;   return 1;   // BF16
        case 34: *blck = 256; *bytes = 54;  return 1;   // TQ1_0
        case 35: *blck = 256; *bytes = 66;  return 1;   // TQ2_0
        case 39: *blck = 32;  *bytes = 17;  return 1;   // MXFP4
        default: return 0;
    }
}

static int rd(FILE * f, void * p, size_t n) {
    return fread(p, 1, n, f) == n;
}

static int skip_string(FILE * f) {
    uint64_t n;
    if (!rd(f, &n, 8)) return 0;
    return fseek(f, (long) n, SEEK_CUR) == 0;
}

static int read_string(FILE * f, char * out, size_t cap) {
    uint64_t n;
    if (!rd(f, &n, 8)) return 0;
    if (n >= cap) {
        if (fseek(f, (long) n, SEEK_CUR) != 0) return 0;
        out[0] = 0;
        return 1;
    }
    if (!rd(f, out, n)) return 0;
    out[n] = 0;
    return 1;
}

static int skip_value(FILE * f, uint32_t type);

static int skip_array(FILE * f) {
    uint32_t type;
    uint64_t n;
    if (!rd(f, &type, 4) || !rd(f, &n, 8)) return 0;
    uint64_t fixed = 0;
    switch (type) {
        case 0: case 1: case 7: fixed = 1; break;
        case 2: case 3: fixed = 2; break;
        case 4: case 5: case 6: fixed = 4; break;
        case 10: case 11: case 12: fixed = 8; break;
    }
    if (fixed) {
        return fseek(f, (long) (fixed * n), SEEK_CUR) == 0;
    }
    for (uint64_t i = 0; i < n; i++) {
        if (!skip_value(f, type)) return 0;
    }
    return 1;
}

static int skip_value(FILE * f, uint32_t type) {
    switch (type) {
        case 0: case 1: case 7: return fseek(f, 1, SEEK_CUR) == 0;
        case 2: case 3:         return fseek(f, 2, SEEK_CUR) == 0;
        case 4: case 5: case 6: return fseek(f, 4, SEEK_CUR) == 0;
        case 10: case 11: case 12: return fseek(f, 8, SEEK_CUR) == 0;
        case 8: return skip_string(f);
        case 9: return skip_array(f);
        default: return 0;
    }
}

static void add_other(const char * name, uint64_t bytes) {
    for (int i = 0; i < n_other; i++) {
        if (strcmp(other_name[i], name) == 0) {
            other_bytes[i] += bytes;
            return;
        }
    }
    if (n_other < MAX_OTHER) {
        snprintf(other_name[n_other], sizeof(other_name[0]), "%s", name);
        other_bytes[n_other] = bytes;
        n_other++;
    }
}

static int read_shard(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "phi-gguf: cannot open %s\n", path);
        return 0;
    }
    char magic[4];
    uint32_t version;
    uint64_t n_tensors, n_kv;
    if (!rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !rd(f, &version, 4) || !rd(f, &n_tensors, 8) || !rd(f, &n_kv, 8)) {
        fprintf(stderr, "phi-gguf: %s is not a GGUF file\n", path);
        fclose(f);
        return 0;
    }
    if (version < 2) {
        fprintf(stderr, "phi-gguf: %s: GGUF version %u is too old\n", path, version);
        fclose(f);
        return 0;
    }
    for (uint64_t i = 0; i < n_kv; i++) {
        uint32_t type;
        if (!skip_string(f) || !rd(f, &type, 4) || !skip_value(f, type)) {
            fprintf(stderr, "phi-gguf: %s: bad metadata entry %llu\n", path, (unsigned long long) i);
            fclose(f);
            return 0;
        }
    }
    for (uint64_t i = 0; i < n_tensors; i++) {
        char name[256];
        uint32_t n_dims, type;
        uint64_t dims[8], offset, nel = 1, blck, bytes;
        if (!read_string(f, name, sizeof(name)) || !rd(f, &n_dims, 4) || n_dims > 8) {
            fprintf(stderr, "phi-gguf: %s: bad tensor entry %llu\n", path, (unsigned long long) i);
            fclose(f);
            return 0;
        }
        for (uint32_t d = 0; d < n_dims; d++) {
            if (!rd(f, &dims[d], 8)) { fclose(f); return 0; }
            nel *= dims[d];
        }
        if (!rd(f, &type, 4) || !rd(f, &offset, 8)) { fclose(f); return 0; }
        if (!type_size(type, &blck, &bytes)) {
            fprintf(stderr, "phi-gguf: %s: tensor %s has an unknown type %u\n", path, name, type);
            fclose(f);
            return 0;
        }
        const uint64_t size = nel / blck * bytes;
        if (strncmp(name, "blk.", 4) == 0) {
            char * end;
            const long b = strtol(name + 4, &end, 10);
            if (*end == '.' && b >= 0 && b < MAX_BLOCKS) {
                if (b + 1 > n_blocks) n_blocks = (int) b + 1;
                if (strstr(end, "_exps.weight")) {
                    experts[b] += size;
                } else {
                    dense[b] += size;
                }
                if (strcmp(end, ".attn_k.weight") == 0) {
                    attn[b] = 1;
                    if (n_dims >= 2) kv_width = dims[1];
                }
                continue;
            }
        }
        add_other(name, size);
    }
    fclose(f);
    return 1;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: phi-gguf SHARD...\n");
        return 2;
    }
    for (int i = 1; i < argc; i++) {
        if (!read_shard(argv[i])) return 1;
    }
    uint64_t total = 0;
    printf("blocks %d\n", n_blocks);
    for (int b = 0; b < n_blocks; b++) {
        printf("block %d experts %llu dense %llu attn %d\n", b, (unsigned long long) experts[b], (unsigned long long) dense[b], attn[b]);
        total += experts[b] + dense[b];
    }
    printf("kv_width %llu\n", (unsigned long long) kv_width);
    for (int i = 0; i < n_other; i++) {
        printf("other %s %llu\n", other_name[i], (unsigned long long) other_bytes[i]);
        total += other_bytes[i];
    }
    printf("total %llu\n", (unsigned long long) total);
    return 0;
}
