#ifndef BINDING_LLAMA_H
#define BINDING_LLAMA_H
#ifdef __cplusplus
#include "../options/options.h"
extern "C" {
#endif
typedef struct go_llama_context_params {
   struct llama_context_params * params;
} go_llama_context_params;
struct go_llama_state {
    struct llama_context * ctx;
    struct llama_model * model;
    struct go_llama_context_params ctx_params;
};
void* go_llama_init(void * params_ptr);
void go_llama_free(void * state_ptr);
int go_llama_set_adapters(char ** adapters, int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_LLAMA_H
