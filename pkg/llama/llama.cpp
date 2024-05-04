#include "../../includes/common.h"
#include "../options/options.h"
#include "llama.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct go_llama_state *go_llama_init(void *params_ptr) {
    auto p = (go_llama_params *) params_ptr;
    auto params = (gpt_params *) go_llama_params_to_gpt_params(*p);
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
    /*
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
    */
    return state;
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