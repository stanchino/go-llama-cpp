#ifndef BINDING_LLAMA_H
#define BINDING_LLAMA_H
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef int go_llama_token;
typedef struct tokens_list {
    unsigned long size;
    go_llama_token *tokens;
} tokens_list;
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
typedef struct go_llama_state {
    struct llama_context *ctx;
    struct llama_context *ctx_guidance;
    struct llama_sampling_context *ctx_sampling;
    struct llama_model *model;
    go_llama_params * params;
} go_llama_state;
typedef struct piece {
    char *data;
    size_t size;
} piece;
// cache
void go_llama_kv_cache_seq_rm(struct go_llama_state * state, int seq_id, int p0, int p1);
void go_llama_kv_cache_seq_add(struct go_llama_state * state, int seq_id, int p0, int p1, int delta);
void go_llama_kv_cache_seq_div(struct go_llama_state * state, int seq_id, int p0, int p1, int d);
void go_llama_kv_cache_clear(struct go_llama_state * state);
// decoding
int go_llama_decode_batch(struct go_llama_state * state, tokens_list tokens, int i, int n_eval, int n_past);
// sampling
void go_llama_sampling_init(struct go_llama_state * state);
void go_llama_sampling_reset(struct go_llama_state * state);
int go_llama_sampling_sample(struct go_llama_state * state);
void go_llama_sampling_accept(struct go_llama_state * state, int id, bool apply_grammar);
void go_llama_sampling_info(struct go_llama_state *state);
tokens_list * go_llama_sampling_prev(struct go_llama_state * state);
// context warmup
void go_llama_warmup(struct go_llama_state * state);
// state
void go_llama_backend_init();
struct llama_model * go_llama_load_model_from_file(go_llama_params * params_ptr);
struct llama_context * go_llama_new_context_with_model(struct llama_model * model, go_llama_params * params_ptr);
struct go_llama_state * go_llama_init_state(struct llama_model * model, struct llama_context * ctx, go_llama_params * params_ptr);
void go_llama_system_info(struct go_llama_state *state);
void go_llama_free(struct go_llama_state *state);
// tokenize
bool go_llama_token_is_eog(struct go_llama_state *state, int id);
go_llama_token go_llama_token_eos(struct go_llama_state *state);
go_llama_token go_llama_token_bos(struct go_llama_state *state);
bool go_llama_should_add_bos_token(struct go_llama_state *state);
struct tokens_list go_llama_tokenize(struct go_llama_state *state, const char *prompt, bool add_special, bool parse_special);
piece go_llama_token_to_piece(struct go_llama_state *state, const go_llama_token *tokens, unsigned int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_LLAMA_H
