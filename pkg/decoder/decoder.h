#ifndef BINDING_PREDICTOR_H
#define BINDING_PREDICTOR_H
#include "../llama/llama.h"
#ifdef __cplusplus
extern "C" {
#endif
int go_llama_decode_batch(struct go_llama_state * state, tokens_list tokens, int i, int n_eval, int n_past);
#ifdef __cplusplus
}
#endif
#endif //BINDING_PREDICTOR_H
