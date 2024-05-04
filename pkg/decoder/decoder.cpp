#include "../../includes/common.h"
#include "../llama/llama.h"
#include "decoder.h"

int go_llama_decode_batch(struct go_llama_state * state, tokens_list tokens, int i, int n_eval, int n_past) {
    if (llama_decode(state->ctx, llama_batch_get_one(&tokens.tokens[i], n_eval, n_past, 0))) {
        LOG("%s : failed to eval\n", __func__);
        return 1;
    }
    return 0;
    // printf("eval: %s, i: %i, n_eval: %i, n_past: %i\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, embd).c_str(), i, n_eval, n_past);
    //LOG("\neval: %s, i: %i, n_eval: %i, n_past: %i\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, embd).c_str(), i, n_eval, n_past);
}
