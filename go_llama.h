#ifndef BINDING_LLAMA_H
#define BINDING_LLAMA_H
#ifdef __cplusplus
#include <stdbool.h>
#include "common.h"
extern "C" {
#endif
struct go_llama_state {
    struct llama_context * ctx;
    struct llama_model * model;
    struct llama_context_params ctx_params;
};
typedef struct go_llama_params {
    char * model;
    bool use_mmap;
} go_llama_params;
void* go_llama_init(void * params_ptr);
void go_llama_free(void * state_ptr);
int go_llama_set_adapters(char ** adapters, int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_MAIN_H
