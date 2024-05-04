#ifndef BINDING_SAMPLING_H
#define BINDING_SAMPLING_H
#include "../llama/llama.h"
#ifdef __cplusplus
extern "C" {
#endif
void go_llama_sampling_init(struct go_llama_state * state);
void go_llama_sampling_reset(struct go_llama_state * state);
int go_llama_sampling_sample(struct go_llama_state * state);
void go_llama_sampling_accept(struct go_llama_state * state, int id, bool apply_grammar);
tokens_list * go_llama_sampling_prev(struct go_llama_state * state);
#ifdef __cplusplus
}
#endif
#endif //BINDING_SAMPLING_H
