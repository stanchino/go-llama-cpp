#include "llama.h"
#include "../../includes/common.h"

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

struct go_llama_state * go_llama_init(void * params_ptr) {
    auto p = (go_llama_params *) params_ptr;
    auto params = (gpt_params *) go_llama_params_to_gpt_params(*p);
    auto mparams = llama_model_params_from_gpt_params(*params);
    llama_model * model;

    llama_backend_init();
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    // TODO: Add support for HF and URL loading
    model = llama_load_model_from_file(params->model.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params->model.c_str());
        return nullptr;
    }
    auto cparams = llama_context_params_from_gpt_params(*params);

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params->model.c_str());
        llama_free_model(model);
        return nullptr;
    }

    {
        LOG("warming up the model with an empty run\n");
        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(ctx, llama_batch_get_one(tmp.data(), std::min((int) tmp.size(), params->n_batch), 0, 0));
        llama_kv_cache_clear(ctx);
        llama_synchronize(ctx);
        llama_reset_timings(ctx);
    }
    auto state = new(go_llama_state);
    state->ctx = ctx;
    state->model = model;
    state->llama_params = params;
    state->params = p;
    state->ctx_guidance = nullptr;
    return state;
}

/*
int go_llama_set_adapters(char ** adapters, int size) {
    for (int i = 0; i < size; ++i) {
        const std::string & lora_adapter = adapters[i];
        int err = llama_model_apply_lora_from_file(g_model,
                                                   lora_adapter.c_str(),
                                                   1.0f,
                                                   nullptr,
                                                   (int) g_ctx_params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return err;
        }
    }
    return 0;
}
*/

void go_llama_free(struct go_llama_state * state) {
    llama_print_timings(state->ctx);
    if (state->ctx_guidance) { llama_free(state->ctx_guidance); }
    llama_free(state->ctx);
    llama_free_model(state->model);
    if (state->ctx_sampling) llama_sampling_free(state->ctx_sampling);
    llama_backend_free();
    free(state->llama_params);
}