#include "../../includes/common.h"
#include "../llama/llama.h"
#include "sampling.h"

void go_llama_sampling_init(struct go_llama_state * state) {
    llama_numa_init(state->params->numa);
    state->ctx_sampling = llama_sampling_init(state->params->sparams);
}
int go_llama_sampling_sample(struct go_llama_state * state) {
    return llama_sampling_sample(state->ctx_sampling, state->ctx, state->ctx_guidance);
}
void go_llama_sampling_accept(struct go_llama_state * state, int id, bool apply_grammar) {
    llama_sampling_accept(state->ctx_sampling, state->ctx, id, apply_grammar);
}
tokens_list * go_llama_sampling_prev(struct go_llama_state * state) {
    auto result = new(tokens_list);
    result->size =  state->ctx_sampling->prev.size();
    result->tokens =  state->ctx_sampling->prev.data();
    return result;
}
void go_llama_sampling_reset(struct go_llama_state * state) {
    llama_sampling_reset(state->ctx_sampling);
}