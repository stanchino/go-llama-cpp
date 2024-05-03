#ifndef BINDING_OPTIONS_H
#define BINDING_OPTIONS_H

#include "../util/util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct go_llama_params {
    char *model;
    char *prompt;
    float rope_freq_base;
    float rope_freq_scale;
    bool input_prefix_bos;
    bool use_mmap;
    bool interactive;
    bool interactive_first;
    bool display_prompt;
    int n_ctx;
    int n_predict;
    int grp_attn_n;
    int grp_attn_w;
    int n_keep;
    int n_batch;
    char *input_prefix;
    char *input_suffix;
} go_llama_params;
void *go_llama_params_to_gpt_params(go_llama_params p);
#ifdef __cplusplus
}
#endif
#endif //BINDING_OPTIONS_H
