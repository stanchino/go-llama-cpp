#include "../../includes/common.h"
#include "llama.h"
#include <sys/types.h>
#include <sys/mman.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

// state
gpt_params go_llama_params_to_gpt_params(go_llama_params * params_ptr) {
    gpt_params params;
    params.model = params_ptr->model;
    params.use_mmap = params_ptr->use_mmap;
    params.interactive = params_ptr->interactive;
    params.interactive_first = params_ptr->interactive_first;
    params.input_prefix_bos = params_ptr->input_prefix_bos;
    params.input_prefix = params_ptr->input_prefix;
    params.input_suffix = params_ptr->input_suffix;
    params.display_prompt = params_ptr->display_prompt;
    params.prompt = params_ptr->prompt;
    params.n_predict = params_ptr->n_predict;
    params.n_batch = params_ptr->n_batch;
    params.grp_attn_n = params_ptr->grp_attn_n;
    params.grp_attn_w = params_ptr->grp_attn_w;
    params.n_keep = params_ptr->n_keep;
    if (params_ptr->n_ctx > 0) {
        params.n_ctx = params_ptr->n_ctx;
    }
    if (params_ptr->rope_freq_base > 0) {
        params.rope_freq_base = params_ptr->rope_freq_base;
    }
    if (params_ptr->rope_freq_scale > 0) {
        params.rope_freq_scale = params_ptr->rope_freq_scale;
    }
    return params;
}

void go_llama_backend_init() {
    llama_backend_init();
}
llama_model * go_llama_load_model_from_file(go_llama_params * params_ptr) {
    auto params = go_llama_params_to_gpt_params(params_ptr);
    printf("model: %s\n", params.model.c_str());
    auto mparams = llama_model_params_from_gpt_params(params);
    return llama_load_model_from_file(params.model.c_str(), mparams);
}
llama_context * go_llama_new_context_with_model(llama_model * model, go_llama_params * params_ptr) {
    auto params = go_llama_params_to_gpt_params(params_ptr);
    auto cparams = llama_context_params_from_gpt_params(params);
    return llama_new_context_with_model(model, cparams);
}
struct go_llama_state * go_llama_init_state(struct llama_model * model, struct llama_context * ctx, go_llama_params *params_ptr) {
    auto state = new(go_llama_state);
    state->model = model;
    state->ctx = ctx;
    state->params = params_ptr;
    state->ctx_guidance = nullptr;
    state->ctx_sampling = nullptr;
    return state;
}
void go_llama_free(struct go_llama_state *state) {
    llama_print_timings(state->ctx);
    llama_free(state->ctx);
    llama_free_model(state->model);
    llama_backend_free();
    if (state->ctx_guidance) { llama_free(state->ctx_guidance); }
    if (state->ctx_sampling) llama_sampling_free(state->ctx_sampling);
}
// context warmup
void go_llama_warmup(struct go_llama_state * state) {
    std::vector<llama_token> tmp = { llama_token_bos(state->model), llama_token_eos(state->model), };
    llama_decode(state->ctx, llama_batch_get_one(tmp.data(), std::min((int) tmp.size(), state->params->n_batch), 0, 0));
    llama_kv_cache_clear(state->ctx);
    llama_synchronize(state->ctx);
    llama_reset_timings(state->ctx);
}
// cache
void go_llama_kv_cache_seq_rm(struct go_llama_state * state, int seq_id, int p0, int p1) {
    llama_kv_cache_seq_rm(state->ctx, seq_id, p0, p1);
}
void go_llama_kv_cache_seq_add(struct go_llama_state * state, int seq_id, int p0, int p1, int delta) {
    llama_kv_cache_seq_add(state->ctx, seq_id, p0, p1, delta);
}
void go_llama_kv_cache_seq_div(struct go_llama_state * state, int seq_id, int p0, int p1, int d) {
    llama_kv_cache_seq_div(state->ctx, seq_id, p0, p1, d);
}
void go_llama_kv_cache_clear(struct go_llama_state * state) {
    llama_kv_cache_clear(state->ctx);
}
// decoding
int go_llama_decode_batch(struct go_llama_state * state, tokens_list tokens, int i, int n_eval, int n_past) {
    if (llama_decode(state->ctx, llama_batch_get_one(&tokens.tokens[i], n_eval, n_past, 0))) {
        LOG("%s : failed to eval\n", __func__);
        return 1;
    }
    return 0;
}
// sampling
void go_llama_sampling_init(struct go_llama_state * state) {
    auto params = go_llama_params_to_gpt_params(state->params);
    llama_numa_init(params.numa);
    state->ctx_sampling = llama_sampling_init(params.sparams);
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
// tokenize
bool go_llama_token_is_eog(struct go_llama_state *state, int id) {
    return llama_token_is_eog(state->model, id);
}
struct tokens_list
go_llama_tokenize(struct go_llama_state *state, const char *prompt, bool add_special, bool parse_special) {
    std::vector<llama_token> tokens = llama_tokenize(state->ctx, prompt, add_special, parse_special);
    unsigned int size = tokens.size();
    struct tokens_list list = {
            tokens.size(),
            (go_llama_token *) malloc(sizeof(llama_token) * size)
    };
    for (unsigned int i = 0; i < size; i++) {
        list.tokens[i] = tokens[i];
    }
    return list;
}

piece go_llama_token_to_piece(struct go_llama_state *state, const llama_token *tokens, unsigned int size) {
    std::string token_str;
    for (unsigned int i = 0; i < size; i++) {
        token_str.append(llama_token_to_piece(state->ctx, tokens[i]));
    }
    char *data = (char *) malloc(token_str.size() );
    strncpy(data, token_str.c_str(), token_str.size());
    return piece{
            data,
        token_str.size(),
    };
}
go_llama_token go_llama_token_eos(struct go_llama_state *state) {
    return llama_token_eos(state->model);
}
go_llama_token go_llama_token_bos(struct go_llama_state *state) {
    return llama_token_bos(state->model);
}
bool go_llama_should_add_bos_token(struct go_llama_state *state) {
    return llama_should_add_bos_token(state->model);
}
void go_llama_system_info(struct go_llama_state *state) {
    auto params = go_llama_params_to_gpt_params(state->params);
    LOG("%s\n", get_system_info(params).c_str());
}
void go_llama_sampling_info(struct go_llama_state *state) {
    auto params = go_llama_params_to_gpt_params(state->params);
    LOG("sampling: \n%s\n", llama_sampling_print(params.sparams).c_str());
    LOG("sampling order: \n%s\n", llama_sampling_order_print(params.sparams).c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", params.n_ctx, params.n_batch, params.n_predict, params.n_keep);
}