#include "../../includes/common.h"
#include "../llama/llama.h"
#include "cache.h"

void go_llama_kv_cache_seq_rm(struct go_llama_state * state, int seq_id, int p0, int p1) {
    llama_kv_cache_seq_rm(state->ctx, seq_id, p0, p1);
}
void go_llama_kv_cache_seq_add(struct go_llama_state * state, int seq_id, int p0, int p1, int delta) {
    llama_kv_cache_seq_add(state->ctx, seq_id, p0, p1, delta);
}
void go_llama_kv_cache_seq_div(struct go_llama_state * state, int seq_id, int p0, int p1, int d) {
    llama_kv_cache_seq_div(state->ctx, seq_id, p0, p1, d);
}