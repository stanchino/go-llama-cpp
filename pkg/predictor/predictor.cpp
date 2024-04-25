#include "predictor.h"
#include "../llama/llama.h"
#include "../../includes/common.h"

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

go_llama_state * g_state;
static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!is_interacting && g_state->params->interactive) {
            is_interacting = true;
        } else {
            printf("%s\n", ANSI_COLOR_RESET);
            go_llama_free(g_state);
            _exit(130);
        }
    }
}
#endif

bool go_llama_token_is_eog(llama_model * model, std::vector<llama_token> eot_ids, llama_token token) {
    bool is_eog = llama_token_is_eog(model, token) ||
            std::any_of(eot_ids.begin(), eot_ids.end(), [&token](llama_token eot_id) {
                return eot_id == token;
            });
    if (is_eog) {
        LOG("found EOS token\n");
    }
    return is_eog;
}

bool go_llama_token_is_anti_prompt(struct go_llama_state * state_ptr, llama_token token, const std::vector<llama_token> & anti_prompt_ids) {
    auto state = g_state = (go_llama_state *) state_ptr;
    gpt_params params = *state->llama_params;
    llama_context * ctx = state->ctx;
    llama_sampling_context * ctx_sampling = state->ctx_sampling;

    // check for reverse prompt using special tokens
    for (const llama_token & anti_prompt : anti_prompt_ids) {
        if (anti_prompt == token) {
            LOG("found anti-prompt in last sampling: %s\n", llama_token_to_piece(ctx, token).c_str());
            return true;
        }
    }

    // check for reverse prompt in the last n_prev tokens
    if (!params.antiprompt.empty()) {
        const int n_prev = 64;
        const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

        // Check if each of the reverse prompts appears at the end of the output.
        // If we're not running interactively, the reverse prompt might be tokenized with some following characters
        // so we'll compensate for that by widening the search window a bit.
        for (std::string & anti_prompt : params.antiprompt) {
            size_t extra_padding = params.interactive ? 0 : 2;
            size_t search_start_pos = last_output.length() > static_cast<size_t>(anti_prompt.length() + extra_padding)
                                      ? last_output.length() - static_cast<size_t>(anti_prompt.length() + extra_padding)
                                      : 0;

            if (last_output.find(anti_prompt, search_start_pos) != std::string::npos) {
                LOG("found anti-prompt in last output: %s\n", last_output.c_str());
                return true;
            }
        }
    }
    return false;
}

int go_llama_predict(struct go_llama_state * state_ptr, const char * prompt) {
    auto state = g_state = (go_llama_state *) state_ptr;
    gpt_params params = *state->llama_params;
    llama_context * ctx = state->ctx;
    llama_model * model = state->model;

    llama_sampling_params & sparams = params.sparams;

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_TEE("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    if (params.rope_freq_base != 0.0) {
        LOG_TEE("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 0.0) {
        LOG_TEE("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    LOG_TEE("%s: build = %d (%s)\n",      __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(nullptr);
    }

    LOG_TEE("%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    llama_numa_init(params.numa);

    llama_context * ctx_guidance = state->ctx_guidance;

    /* TODO: load lora adapter
    // load the model and apply lora adapter, if any
    LOG("%s: load the model and apply lora adapter, if any\n", __func__);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);
    */
    if (sparams.cfg_scale > 1.f) {
        struct llama_context_params lparams = llama_context_params_from_gpt_params(params);
        ctx_guidance = llama_new_context_with_model(model, lparams);
    }

    const int n_ctx_train = llama_n_ctx_train(model);
    const unsigned int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE("%s: warning: model was trained on only %d context tokens (%d specified)\n",
                __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_TEE("\n");
        LOG_TEE("%s\n", get_system_info(params).c_str());
    }

    //TODO: Setup session tokens

    const bool add_bos = llama_should_add_bos_token(model);
    GGML_ASSERT(llama_add_eos_token(model) != 1);
    LOG("add_bos: %d\n", add_bos);


    // Tokenize negative prompt
    std::vector<llama_token> guidance_inp;
    unsigned int guidance_offset = 0;
    unsigned int original_prompt_len = 0;
    if (ctx_guidance) {
        LOG("cfg_negative_prompt: \"%s\"\n", log_tostr(sparams.cfg_negative_prompt));

        guidance_inp = ::llama_tokenize(ctx_guidance, sparams.cfg_negative_prompt, true, true);
        LOG("guidance_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx_guidance, guidance_inp).c_str());

        std::vector<llama_token> original_inp = ::llama_tokenize(ctx, params.prompt, true, true);
        LOG("original_inp tokenized: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, original_inp).c_str());

        original_prompt_len = original_inp.size();
        guidance_offset = guidance_inp.size() - original_prompt_len;
        LOG("original_prompt_len: %s", log_tostr(original_prompt_len));
        LOG("guidance_offset:     %s", log_tostr(guidance_offset));
    }

    //TODO: If session tokens are found check if prompt matches any of the session tokens

    // enable interactive mode if interactive start is specified
    if (params.interactive_first) {
        params.interactive = true;
    }

    // ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action{};
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, nullptr);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (params.interactive) {
        LOG_TEE("%s: interactive mode on.\n", __func__);

        if (!params.antiprompt.empty()) {
            for (const auto & antiprompt : params.antiprompt) {
                LOG_TEE("Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (params.input_prefix_bos) {
            LOG_TEE("Input prefix with BOS\n");
        }

        if (!params.input_prefix.empty()) {
            LOG_TEE("Input prefix: '%s'\n", params.input_prefix.c_str());
        }

        if (!params.input_suffix.empty()) {
            LOG_TEE("Input suffix: '%s'\n", params.input_suffix.c_str());
        }
    }
    LOG_TEE("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG_TEE("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG_TEE("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);

    // group-attention state
    // number of grouped KV tokens so far (used only if params.grp_attn_n > 1)
    int ga_i = 0;

    const int ga_n = params.grp_attn_n;
    const int ga_w = params.grp_attn_w;

    if (ga_n != 1) {
        GGML_ASSERT(ga_n > 0                    && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(ga_w % ga_n == 0            && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG_TEE("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, ga_n, ga_w);
    }
    LOG_TEE("\n\n");

    if (params.interactive) {
        const char *control_message;
        if (params.multiline_input) {
            control_message = " - To return control to LLaMa, end your input with '\\'.\n"
                              " - To return control without starting a new line, end your input with '/'.\n";
        } else {
            control_message = " - Press Return to return control to LLaMa.\n"
                              " - To return control without starting a new line, end your input with '/'.\n"
                              " - If you want to submit another line, end your input with '\\'.\n";
        }
        LOG_TEE("== Running in interactive mode. ==\n");
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
        LOG_TEE(       " - Press Ctrl+C to interject at any time.\n");
#endif
        LOG_TEE(       "%s\n", control_message);

        is_interacting = params.interactive_first;
    }

    if (is_interacting) {
        params.prompt = predictorInputCallback(state);
        is_interacting = false;
    } else if (!std::string(prompt).empty()) {
        params.prompt = prompt;
    } else if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }
    std::vector<llama_token> embd_inp;

    if (!params.prompt.empty()) {
        LOG("tokenize the prompt\n");
        embd_inp = ::llama_tokenize(ctx, params.prompt, true, true);
    }

    LOG("prompt: \"%s\"\n", log_tostr(params.prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_inp).c_str());
    }
    // Check prompt size
    if (embd_inp.size() > n_ctx - 4) {
        LOG_TEE("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return 1;
    }
    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int) embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

    bool is_antiprompt        = false;
    bool input_echo           = true;
    bool display              = params.display_prompt;

    int n_past             = 0;
    int n_remain           = params.n_predict;
    unsigned int n_total_consumed         = 0;
    int n_past_guidance    = 0;

    // the first thing we will do is to output the prompt, so set color accordingly
    printf(ANSI_COLOR_GREEN);

    std::vector<llama_token> embd_full;
    std::vector<llama_token> embd_out;
    std::vector<llama_token> embd_guidance;

    // tokenized anti prompts
    std::vector<llama_token> anti_prompt_ids;
    if (!params.antiprompt.empty()) {
        anti_prompt_ids.reserve(params.antiprompt.size());
        for (std::string & antiprompt : params.antiprompt) {
            anti_prompt_ids.emplace_back(::llama_tokenize(ctx, antiprompt, false, true)[0]);
        }
    }

    // tokenized custom eot tokens
    std::vector<llama_token> eot_ids;
    if (state->params->eot_prompts) {
        eot_ids.reserve(state->params->eot_prompts->len);
        for (size_t i = 0; i < state->params->eot_prompts->len; i++) {
            eot_ids.emplace_back(llama_tokenize(ctx, state->params->eot_prompts->data[i], false, true)[0]);
        }
    }

    struct llama_sampling_context * ctx_sampling = state->ctx_sampling = llama_sampling_init(sparams);

    while ((n_remain != 0 && !is_antiprompt) || params.interactive) {
        // predict
        if (!embd_full.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            unsigned int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd_full if necessary.
            if (embd_full.size() > max_embd_size) {
                const unsigned int skipped_tokens = embd_full.size() - max_embd_size;
                embd_full.resize(max_embd_size);

                printf(ANSI_COLOR_RED);
                printf("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
                printf(ANSI_COLOR_RESET);
                fflush(stdout);
            }

            // Ensure the output doesn't exceed the context size by truncating embd_guidance if necessary
            if (ga_n == 1) {
                // infinite text generation via context shifting
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
                if (n_past + (int) embd_full.size() + std::max<unsigned int>(0, guidance_offset) > n_ctx) {
                    if (params.n_predict == -2) {
                        LOG_TEE("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, params.n_predict);
                        break;
                    }

                    const int n_left    = n_past - params.n_keep;
                    const int n_discard = n_left/2;

                    LOG("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
                        n_past, n_left, n_ctx, params.n_keep, n_discard);

                    llama_kv_cache_seq_rm (ctx, 0, params.n_keep            , params.n_keep + n_discard);
                    llama_kv_cache_seq_add(ctx, 0, params.n_keep + n_discard, n_past, -n_discard);

                    n_past -= n_discard;

                    if (ctx_guidance) {
                        n_past_guidance -= n_discard;
                    }

                    LOG("after swap: n_past = %d, n_past_guidance = %d\n", n_past, n_past_guidance);

                    LOG("embd_full: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_full).c_str());

                    LOG("clear session path\n");
                }
            } else {
                // context extension via Self-Extend
                while (n_past >= ga_i + ga_w) {
                    const int ib = (ga_n*ga_i)/ga_w;
                    const int bd = (ga_w/ga_n)*(ga_n - 1);
                    const int dd = (ga_w/ga_n) - ib*bd - ga_w;

                    LOG("\n");
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, n_past, ib*bd, ga_i + ib*bd, n_past + ib*bd);
                    LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib*bd, ga_i + ib*bd + ga_w, ga_n, (ga_i + ib*bd)/ga_n, (ga_i + ib*bd + ga_w)/ga_n);
                    LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib*bd + ga_w, n_past + ib*bd, dd, ga_i + ib*bd + ga_w + dd, n_past + ib*bd + dd);

                    llama_kv_cache_seq_add(ctx, 0, ga_i,                n_past,              ib*bd);
                    llama_kv_cache_seq_div(ctx, 0, ga_i + ib*bd,        ga_i + ib*bd + ga_w, ga_n);
                    llama_kv_cache_seq_add(ctx, 0, ga_i + ib*bd + ga_w, n_past + ib*bd,      dd);

                    n_past -= bd;

                    ga_i += ga_w/ga_n;

                    LOG("\nn_past_old = %d, n_past = %d, ga_i = %d\n\n", n_past + bd, n_past, ga_i);
                }
            }

            //TODO: try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)

            // evaluate tokens in batches
            // embd_full is typically prepared beforehand to fit within a batch, but not always
            if (ctx_guidance) {
                unsigned int input_size;
                llama_token * input_buf;

                if (n_past_guidance < (int) guidance_inp.size()) {
                    // Guidance context should have the same data with these modifications:
                    //
                    // * Replace the initial prompt
                    // * Shift everything by guidance_offset
                    embd_guidance = guidance_inp;
                    if (embd_full.begin() + original_prompt_len < embd_full.end()) {
                        embd_guidance.insert(
                                embd_guidance.end(),
                                embd_full.begin() + original_prompt_len,
                                embd_full.end()
                        );
                    }

                    input_buf  = embd_guidance.data();
                    input_size = embd_guidance.size();

                    LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_guidance).c_str());
                } else {
                    input_buf  = embd_full.data();
                    input_size = embd_full.size();
                }

                for (int i = 0; i < input_size; i += params.n_batch) {
                    int n_eval = std::min((int) input_size - i, params.n_batch);
                    if (llama_decode(ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, n_past_guidance, 0))) {
                        LOG_TEE("%s : failed to eval\n", __func__);
                        return 1;
                    }

                    n_past_guidance += n_eval;
                }
            }

            for (int i = 0; i < (int) embd_full.size(); i += params.n_batch) {
                int n_eval = (int) embd_full.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, embd_full).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd_full[i], n_eval, n_past, 0))) {
                    LOG_TEE("%s : failed to eval\n", __func__);
                    return 1;
                }

                n_past += n_eval;

                LOG("n_past = %d\n", n_past);
                // Display total tokens alongside total time
                if (params.n_print > 0 && n_past % params.n_print == 0) {
                    LOG_TEE("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
                }
            }

            //TODO: save embedding tokens to the session tokens
        }

        embd_full.clear();
        embd_out.clear();
        embd_guidance.clear();

        if ((int) embd_inp.size() <= n_total_consumed && !is_interacting) {
            // if the input is consumed there is no user interaction
            //TODO: optionally save the session on first sample (for faster prompt loading next time)
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, ctx_guidance);
            // deal with end of generation tokens in interactive mode
            if (go_llama_token_is_eog(model, eot_ids, id) || go_llama_token_is_anti_prompt(state, id, anti_prompt_ids)) {
                if (params.interactive) {
                    is_interacting = true;
                }
                is_antiprompt = true;
            }

            llama_sampling_accept(ctx_sampling, ctx, id, true);

            LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, ctx_sampling->prev).c_str());

            embd_full.push_back(id);
            if (!is_antiprompt) embd_out.push_back(id);
            input_echo = true;

            // decrement remaining sampling budget
            --n_remain;

            LOG("n_remain: %d\n", n_remain);
        } else {
            // processing user input
            // some user input remains from prompt or interaction, forward it to processing
            LOG("embd_inp.size(): %d, n_total_consumed: %d\n", (int) embd_inp.size(), n_total_consumed);
            while ((int) embd_inp.size() > n_total_consumed) {
                embd_full.push_back(embd_inp[n_total_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_total_consumed], false);

                ++n_total_consumed;
                if ((int) embd_full.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // display text
        LOG("input %d, display %d\n", input_echo, display);
        if (input_echo && display) {
            for (auto id : embd_out) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                LOG("%s", token_str.c_str());
                predictorOutputCallback(state, token_str.c_str());
            }
        }
        // reset color to default if there is no pending user input
        if (input_echo && (int) embd_inp.size() == n_total_consumed) {
            display = true;
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_total_consumed) {
            if (n_past > 0 && is_interacting) {
                LOG("waiting for user input\n");

                if (params.input_prefix_bos || add_bos) {
                    LOG("adding input prefix BOS token\n");
                    embd_inp.push_back(llama_token_bos(model));
                }

                if (!params.input_prefix.empty()) {
                    LOG("appending input prefix: '%s'\n", params.input_prefix.c_str());
                    printf("%s", params.input_prefix.c_str());
                }

                std::string buffer = predictorInputCallback(state);
                display = true;

                // Add tokens to embd_full only if the input buffer is non-empty
                // Entering a empty line lets the user pass control back
                if (buffer.length() > 1) {
                    // append input suffix if any
                    if (!params.input_suffix.empty()) {
                        LOG("appending input suffix: '%s'\n", params.input_suffix.c_str());
                        printf("%s", params.input_suffix.c_str());
                    }

                    LOG("buffer: '%s'\n", buffer.c_str());

                    const size_t original_size = embd_inp.size();

                    if (params.escape) {
                        process_escapes(buffer);
                    }

                    const auto line_pfx = ::llama_tokenize(ctx, params.input_prefix, false, true);
                    const auto line_inp = ::llama_tokenize(ctx, buffer,              false, false);
                    const auto line_sfx = ::llama_tokenize(ctx, params.input_suffix, false, true);

                    LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(ctx, line_inp).c_str());

                    embd_inp.insert(embd_inp.end(), line_pfx.begin(), line_pfx.end());
                    embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());
                    embd_inp.insert(embd_inp.end(), line_sfx.begin(), line_sfx.end());

                    if (params.display_prompt) {
                        for (size_t i = original_size; i < embd_inp.size(); ++i) {
                            predictorOutputCallback(state, llama_token_to_piece(ctx, embd_inp[i]).c_str());
                        }
                    }

                    n_remain -= (int) line_inp.size() + add_bos;
                    LOG("n_remain: %d\n", n_remain);
                } else {
                    LOG("empty line, passing control back\n");
                }

                input_echo = false; // do not echo this again
            }

            if (n_past > 0) {
                if (is_interacting) {
                    llama_sampling_reset(ctx_sampling);
                }
                is_interacting = is_antiprompt = false;
            }
        }

        // end of generation
        if (!embd_full.empty() && go_llama_token_is_eog(model, eot_ids, embd_full.back()) && !params.interactive) {
            LOG_TEE(" [end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (params.interactive && n_remain <= 0 && params.n_predict >= 0) {
            n_remain = params.n_predict;
            is_interacting = true;
        }
    }
    predictorEndOutputCallback(state);

    //TODO: save output tokens to the session tokens
    return 0;
}