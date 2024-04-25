#ifndef BINDING_OPTIONS_H
#define BINDING_OPTIONS_H
#include "../util/util.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct go_llama_params {
    char * model;
    bool use_mmap;
    bool interactive;
    bool interactive_first;
    bool display_prompt;
    int n_ctx;
    char * input_prefix;
    char * input_suffix;
    charArray * anti_prompts;
    charArray * eot_prompts;
} go_llama_params;
void * go_llama_params_to_gpt_params(go_llama_params p);
#ifdef __cplusplus
}
#endif
#endif //BINDING_OPTIONS_H
