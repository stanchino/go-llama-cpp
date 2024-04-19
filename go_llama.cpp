#include "go_llama.h"

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <csignal>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

llama_model * g_model;
llama_context * g_ctx;
llama_context_params g_ctx_params;
gpt_params g_params;

void* go_llama_init(void * params_ptr) {
    go_llama_params p = *(go_llama_params*)params_ptr;
    gpt_params params;
    params.model = p.model;
    auto mparams = llama_model_params_from_gpt_params(params);
    mparams.use_mmap = p.use_mmap;
    llama_model * model;

    llama_backend_init();
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    // TODO: Add support for HF and URL loading
    model = llama_load_model_from_file(p.model, mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
        return nullptr;
    }
    auto cparams = llama_context_params_from_gpt_params(params);

    llama_context * ctx = llama_new_context_with_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model.c_str());
        llama_free_model(model);
        return nullptr;
    }

    {
        LOG("warming up the model with an empty run\n");
        std::vector<llama_token> tmp = { llama_token_bos(model), llama_token_eos(model), };
        llama_decode(ctx, llama_batch_get_one(tmp.data(), std::min((int) tmp.size(), params.n_batch), 0, 0));
        llama_kv_cache_clear(ctx);
        llama_synchronize(ctx);
        llama_reset_timings(ctx);
    }
    go_llama_state * state;
    state = new go_llama_state;
    state->model = model;
    state->ctx = ctx;
    state->ctx_params = cparams;
    g_params = params;
    return state;
}

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

void go_llama_free(void * state_ptr) {
    go_llama_state * state = (go_llama_state *)state_ptr;
    llama_print_timings(state->ctx);
    llama_free(state->ctx);
    llama_free_model(state->model);
    llama_backend_free();
}