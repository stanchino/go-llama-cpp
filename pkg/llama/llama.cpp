#include "../../includes/common.h"
#include "llama.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
/*
struct go_llama_state *go_llama_init(void *params_ptr) {
    auto params = (gpt_params *) params_ptr;
    auto mparams = llama_model_params_from_gpt_params(*params);
    llama_model *model;

    llama_backend_init();
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    // TODO: Add support for HF and URL loading
    model = llama_load_model_from_file(params->model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params->model.c_str());
        return nullptr;
    }
    auto cparams = llama_context_params_from_gpt_params(*params);

    llama_context *ctx = llama_new_context_with_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params->model.c_str());
        llama_free_model(model);
        return nullptr;
    }

    {
        LOG("warming up the model with an empty run\n");
        std::vector<llama_token> tmp = {llama_token_bos(model), llama_token_eos(model),};
        llama_decode(ctx, llama_batch_get_one(tmp.data(), std::min((int) tmp.size(), params->n_batch), 0, 0));
        llama_kv_cache_clear(ctx);
        llama_synchronize(ctx);
        llama_reset_timings(ctx);
    }
    auto state = new(go_llama_state);
    state->ctx = ctx;
    state->model = model;
    state->params = params;

    LOG("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);
    if (state->params->seed == LLAMA_DEFAULT_SEED) {
        state->params->seed = time(nullptr);
    }
    LOG("%s: seed  = %u\n", __func__, state->params->seed);
    const int n_ctx_train = llama_n_ctx_train(state->model);

    //TODO: Setup session tokens

    GGML_ASSERT(llama_add_eos_token(state->model) != 1);
    //TODO: If session tokens are found check if prompt matches any of the session tokens
    llama_sampling_params &sparams = state->params->sparams;
    LOG("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", state->params->n_ctx, state->params->n_batch,
        state->params->n_predict, state->params->n_keep);

    if (state->params->grp_attn_n != 1) {
        GGML_ASSERT(state->params->grp_attn_n > 0 && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(state->params->grp_attn_w % state->params->grp_attn_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, state->params->grp_attn_n, state->params->grp_attn_w);
    }
    if (sparams.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(*p_state->params);
        p_state->ctx_guidance = llama_new_context_with_model(state->model, lparams);

        // Tokenize negative prompt
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

        emb->guidance_inp = ::llama_tokenize(p_state->ctx_guidance, sparams.cfg_negative_prompt, true, true);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(p_state->ctx_guidance, emb->guidance_inp).c_str());

        std::vector<llama_token> original_inp = ::llama_tokenize(state->ctx, p_state->params->prompt, true, true);
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, original_inp).c_str());

        p_state->original_prompt_len = original_inp.size();
        p_state->guidance_offset = emb->guidance_inp.size() - p_state->original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(p_state->original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(p_state->guidance_offset));
    }
    return state;
}
*/
// context warmup
void go_llama_synchronize(struct go_llama_state *state) {
    llama_synchronize(state->ctx);
}
void go_llama_reset_timings(struct go_llama_state *state) {
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
struct llama_sampling_context * go_llama_sampling_init(struct go_llama_state * state) {
    llama_numa_init(state->params->numa);
    return llama_sampling_init(state->params->sparams);
}
int go_llama_sampling_sample(struct go_llama_state * state) {
    return llama_sampling_sample(state->ctx_sampling, state->ctx, state->ctx_guidance);
}
void go_llama_sampling_accept(struct go_llama_state * state, int id, bool apply_grammar) {
    printf("ctx %d", state != nullptr);
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
// state
void *go_llama_params_to_gpt_params(void *p_ptr) {
    auto p = *(go_llama_params *) p_ptr;
    gpt_params params;
    auto params_ptr = (gpt_params *) malloc(sizeof(gpt_params));
    params.model = p.model;
    params.use_mmap = p.use_mmap;
    params.interactive = p.interactive;
    params.interactive_first = p.interactive_first;
    params.input_prefix_bos = p.input_prefix_bos;
    params.input_prefix = p.input_prefix;
    params.input_suffix = p.input_suffix;
    params.display_prompt = p.display_prompt;
    params.prompt = p.prompt;
    params.n_predict = p.n_predict;
    params.n_batch = p.n_batch;
    params.grp_attn_n = p.grp_attn_n;
    params.grp_attn_w = p.grp_attn_w;
    params.n_keep = p.n_keep;
    if (p.n_ctx > 0) {
        params.n_ctx = p.n_ctx;
    }
    if (p.rope_freq_base > 0) {
        params.rope_freq_base = p.rope_freq_base;
    }
    if (p.rope_freq_scale > 0) {
        params.rope_freq_scale = p.rope_freq_scale;
    }
    *params_ptr = params;
    return (void *) params_ptr;
}

void go_llama_backend_init() {
    llama_backend_init();
}
llama_model * go_llama_load_model_from_file(void *params_ptr) {
    auto params = (gpt_params *) params_ptr;
    auto mparams = llama_model_params_from_gpt_params(*params);
    return llama_load_model_from_file(params->model.c_str(), mparams);
}
llama_context * go_llama_new_context_with_model(llama_model * model, void *params_ptr) {
    auto params = (gpt_params *) params_ptr;
    auto cparams = llama_context_params_from_gpt_params(*params);
    return llama_new_context_with_model(model, cparams);
}
void go_llama_free(struct go_llama_state *state) {
    llama_print_timings(state->ctx);
    llama_free(state->ctx);
    llama_free_model(state->model);
    llama_backend_free();
    if (state->ctx_guidance) { llama_free(state->ctx_guidance); }
    if (state->ctx_sampling) llama_sampling_free(state->ctx_sampling);
    free(state->params);
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

const char * go_llama_token_to_piece(struct go_llama_state *state, const llama_token *tokens, unsigned int size) {
    std::string token_str;
    for (unsigned int i = 0; i < size; i++) {
        token_str.append(llama_token_to_piece(state->ctx, tokens[i]));
    }
    unsigned long token_str_size = token_str.size();
    char *res = (char *) malloc(token_str_size + 1);
    strncpy(res, token_str.c_str(), token_str_size);
    return res;
}
go_llama_token go_llama_token_еos(struct go_llama_state *state) {
    return llama_token_eos(state->model);
}
go_llama_token go_llama_token_bos(struct go_llama_state *state) {
    return llama_token_bos(state->model);
}
bool go_llama_should_add_bos_token(struct go_llama_state *state) {
    return llama_should_add_bos_token(state->model);
}