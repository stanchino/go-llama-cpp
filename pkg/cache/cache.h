#ifndef BINDING_CACHE_H
#define BINDING_CACHE_H
#include "../llama/llama.h"
#ifdef __cplusplus
extern "C" {
#endif
void go_llama_kv_cache_seq_rm(struct go_llama_state * state, int seq_id, int p0, int p1);
void go_llama_kv_cache_seq_add(struct go_llama_state * state, int seq_id, int p0, int p1, int delta);
void go_llama_kv_cache_seq_div(struct go_llama_state * state, int seq_id, int p0, int p1, int d);
#ifdef __cplusplus
}
#endif
#endif //BINDING_CACHE_H
