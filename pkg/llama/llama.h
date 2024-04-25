#ifndef BINDING_LLAMA_H
#define BINDING_LLAMA_H
#ifdef __cplusplus
#include "../options/options.h"
#include "llama.h"
extern "C" {
#endif
typedef struct go_llama_state {
    struct llama_context * ctx;
    struct llama_context * ctx_guidance;
    struct llama_sampling_context * ctx_sampling;
    struct llama_model * model;
    struct gpt_params * llama_params;
    struct go_llama_params * params;
} go_llama_state;
struct go_llama_state * go_llama_init(void * params_ptr);
void go_llama_free(struct go_llama_state * state);
//int go_llama_set_adapters(char ** adapters, int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_LLAMA_H
