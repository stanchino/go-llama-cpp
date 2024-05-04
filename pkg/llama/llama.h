#ifndef BINDING_LLAMA_H
#define BINDING_LLAMA_H
#ifdef __cplusplus
extern "C" {
#endif
typedef int go_llama_token;
typedef struct go_llama_state {
    struct llama_context *ctx;
    struct llama_context *ctx_guidance;
    struct llama_sampling_context *ctx_sampling;
    struct llama_model *model;
    struct gpt_params *params;
} go_llama_state;
typedef struct tokens_list {
    unsigned long size;
    go_llama_token *tokens;
} tokens_list;
struct go_llama_state *go_llama_init(void *params_ptr);
void go_llama_free(struct go_llama_state *state);
#ifdef __cplusplus
}
#endif
#endif //BINDING_LLAMA_H
