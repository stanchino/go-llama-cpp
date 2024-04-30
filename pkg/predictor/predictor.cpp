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

typedef struct embeddings {
    std::vector<llama_token> eot_ids;
    std::vector<llama_token> anti_prompt_ids;
    std::vector<llama_token> embd_full;
    std::vector<llama_token> embd_out;
    std::vector<llama_token> embd_in;
    std::vector<llama_token> guidance_inp;
    std::vector<llama_token> embd_guidance;
} embeddings;

go_llama_state *g_state;
go_llama_predict_state *g_predict_state;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (!g_predict_state->is_interacting && g_state->params->interactive) {
            g_predict_state->is_interacting = true;
        } else {
            printf("%s\n", ANSI_COLOR_RESET);
            go_llama_free(g_state);
            _exit(130);
        }
    }
}
#endif

bool token_is_eog(llama_model *model, std::vector<llama_token> eot_ids, llama_token token) {
    bool is_eog = llama_token_is_eog(model, token) ||
                  std::any_of(eot_ids.begin(), eot_ids.end(), [&token](llama_token eot_id) {
                      return eot_id == token;
                  });
    if (is_eog) {
        LOG("found EOS token\n");
    }
    return is_eog;
}

std::vector<llama_token> generate_eot_ids(struct go_llama_state *state) {
    std::vector<llama_token> eot_ids;
    if (state->init_params->eot_prompts) {
        eot_ids.reserve(state->init_params->eot_prompts->len);
        for (size_t i = 0; i < state->init_params->eot_prompts->len; i++) {
            eot_ids.emplace_back(llama_tokenize(state->ctx, state->init_params->eot_prompts->data[i], false, true)[0]);
        }
    }

    return eot_ids;
}

std::vector<llama_token> generate_anti_prompt_ids(struct go_llama_state *state) {
    std::vector<llama_token> anti_prompt_ids;
    if (state->init_params->anti_prompts) {
        anti_prompt_ids.reserve(state->init_params->anti_prompts->len);
        for (size_t i = 0; i < state->init_params->anti_prompts->len; i++) {
            anti_prompt_ids.emplace_back(llama_tokenize(state->ctx, state->init_params->anti_prompts->data[i], false, true)[0]);
        }
    }

    return anti_prompt_ids;
}

bool token_is_anti_prompt(struct go_llama_state *state, llama_token token,
                          const std::vector<llama_token> &anti_prompt_ids) {
    // check for reverse prompt using special tokens
    bool is_anti_prompt = std::any_of(anti_prompt_ids.begin(), anti_prompt_ids.end(),
                                      [&token](llama_token anti_prompt) {
                                          return anti_prompt == token;
                                      });
    if (is_anti_prompt) {
        LOG("found anti-prompt in last sampling: %s\n", llama_token_to_piece(state->ctx, token).c_str());
    }
    return is_anti_prompt;
    // check for reverse prompt in the last n_prev tokens
    /*
    llama_sampling_context *ctx_sampling = state->ctx_sampling;
    if (!params.antiprompt.empty()) {
        const int n_prev = 64;
        const std::string last_output = llama_sampling_prev_str(ctx_sampling, ctx, n_prev);

        // Check if each of the reverse prompts appears at the end of the output.
        // If we're not running interactively, the reverse prompt might be tokenized with some following characters
        // so we'll compensate for that by widening the search window a bit.
        for (std::string &anti_prompt: params.antiprompt) {
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
    */
}

/*
void prepare_params(struct go_llama_predict_state * p_state) {
    if (p_state->params->n_ctx != 0 && p_state->params->n_ctx < 8) {
        LOG("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        p_state->params->n_ctx = 8;
    }

    if (p_state->params->rope_freq_base != 0.0) {
        LOG("%s: warning: changing RoPE frequency base to %g.\n", __func__, p_state->params->rope_freq_base);
    }

    if (p_state->params->rope_freq_scale != 0.0) {
        LOG("%s: warning: scaling RoPE frequency by %g.\n", __func__, p_state->params->rope_freq_scale);
    }

    LOG("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);

    if (p_state->params->seed == LLAMA_DEFAULT_SEED) {
        p_state->params->seed = time(nullptr);
    }

    LOG("%s: seed  = %u\n", __func__, p_state->params->seed);

    // enable interactive mode if interactive start is specified
    if (p_state->params->interactive_first) {
        p_state->params->interactive = true;
    }

    if (p_state->params->interactive) {
        LOG("%s: interactive mode on.\n", __func__);

        if (!p_state->params->antiprompt.empty()) {
            for (const auto &antiprompt: p_state->params->antiprompt) {
                LOG("Reverse prompt: '%s'\n", antiprompt.c_str());
            }
        }

        if (p_state->params->input_prefix_bos) {
            LOG("Input prefix with BOS\n");
        }

        if (!p_state->params->input_prefix.empty()) {
            LOG("Input prefix: '%s'\n", p_state->params->input_prefix.c_str());
        }

        if (!p_state->params->input_suffix.empty()) {
            LOG("Input suffix: '%s'\n", p_state->params->input_suffix.c_str());
        }
    }
}
*/

void go_llama_set_prompt(struct go_llama_state * state, struct go_llama_predict_state * p_state, const char * prompt) {
    p_state->params->prompt = prompt;
    auto emb = (embeddings *) p_state->embeddings;
    // Tokenize the prompt
    if (!p_state->params->prompt.empty()) {
        LOG("tokenize the prompt\n");
        emb->embd_in = ::llama_tokenize(state->ctx, p_state->params->prompt, true, true);
    }

    LOG("prompt: \"%s\"\n", log_tostr(p_state->params->prompt));
    LOG("tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, emb->embd_in).c_str());

    // Should not run without any tokens
    if (emb->embd_in.empty()) {
        emb->embd_in.push_back(llama_token_bos(state->model));
        LOG("embd_inp was considered empty and bos was added: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, emb->embd_in).c_str());
    }
    const bool add_bos = llama_should_add_bos_token(state->model);
    LOG("add_bos: %d\n", add_bos);
    // number of tokens to keep when resetting context
    if (p_state->params->n_keep<0 || p_state->params->n_keep>(int) emb->embd_in.size()) {
        p_state->params->n_keep = (int) emb->embd_in.size();
    } else {
        p_state->params->n_keep += add_bos; // always keep the BOS token
    }

    // print system information
    LOG("\n%s\n", get_system_info(*p_state->params).c_str());
}

struct go_llama_predict_state * go_llama_init_predict_state(struct go_llama_state * state) {
    auto p_state = new(go_llama_predict_state);
    auto emb = new(embeddings);
    p_state->params = state->params;
    p_state->n_total_consumed = 0;
    p_state->n_past = 0;
    p_state->n_remain = 0;
    p_state->n_past_guidance = 0;
    p_state->guidance_offset = 0;
    p_state->original_prompt_len = 0;
    p_state->is_anti_prompt = false;
    p_state->is_interacting = false;
    p_state->input_echo = true;
    p_state->display = p_state->params->display_prompt;
    LOG("%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG("%s: built with %s for %s\n", __func__, LLAMA_COMPILER, LLAMA_BUILD_TARGET);
    if (p_state->params->seed == LLAMA_DEFAULT_SEED) {
        p_state->params->seed = time(nullptr);
    }
    LOG("%s: seed  = %u\n", __func__, p_state->params->seed);
    const int n_ctx_train = llama_n_ctx_train(state->model);
    p_state->n_ctx = llama_n_ctx(state->ctx);
    LOG("n_ctx: %d\n", p_state->n_ctx);

    p_state->n_remain = p_state->params->n_predict;
    p_state->display = p_state->params->display_prompt;
    emb->eot_ids = generate_eot_ids(state);
    emb->anti_prompt_ids = generate_anti_prompt_ids(state);

    if (p_state->n_ctx > n_ctx_train) {
        LOG("%s: warning: model was trained on only %d context tokens (%d specified)\n",
            __func__, n_ctx_train, p_state->n_ctx);
    }

    //TODO: Setup session tokens

    GGML_ASSERT(llama_add_eos_token(state->model) != 1);
    //TODO: If session tokens are found check if prompt matches any of the session tokens
    llama_sampling_params &sparams = p_state->params->sparams;
    LOG("sampling: \n%s\n", llama_sampling_print(sparams).c_str());
    LOG("sampling order: \n%s\n", llama_sampling_order_print(sparams).c_str());
    LOG("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", p_state->n_ctx, p_state->params->n_batch,
        p_state->params->n_predict, p_state->params->n_keep);

    if (p_state->params->grp_attn_n != 1) {
        GGML_ASSERT(p_state->params->grp_attn_n > 0 && "grp_attn_n must be positive");                     // NOLINT
        GGML_ASSERT(p_state->params->grp_attn_w % p_state->params->grp_attn_n == 0 && "grp_attn_w must be a multiple of grp_attn_n");     // NOLINT
        //GGML_ASSERT(n_ctx_train % ga_w == 0     && "n_ctx_train must be a multiple of grp_attn_w");    // NOLINT
        //GGML_ASSERT(n_ctx >= n_ctx_train * ga_n && "n_ctx must be at least n_ctx_train * grp_attn_n"); // NOLINT
        LOG("self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d\n", n_ctx_train, p_state->params->grp_attn_n, p_state->params->grp_attn_w);
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
    p_state->embeddings = (void *) emb;
    return p_state;
}

int predict_embeddings(struct go_llama_state *state, struct go_llama_predict_state * p_state) {
    auto emb = (struct embeddings *) p_state->embeddings;
    // Note: (p_state.n_ctx - 4) here is to match the logic for commandline prompt handling via
    // --prompt or --file which uses the same value.
    unsigned int max_embd_size = p_state->n_ctx - 4;

    // Ensure the input doesn't exceed the context size by truncating p_state->embd_full if necessary.
    if (emb->embd_full.size() > max_embd_size) {
        const unsigned int skipped_tokens = emb->embd_full.size() - max_embd_size;
        emb->embd_full.resize(max_embd_size);

        printf("%s<<input too long: skipped %d token%s>>%s", ANSI_COLOR_RED, skipped_tokens, skipped_tokens != 1 ? "s" : "", ANSI_COLOR_RESET);
        fflush(stdout);
    }

    // Ensure the output doesn't exceed the context size by truncating embd_guidance if necessary
    if (p_state->params->grp_attn_n == 1) {
        // infinite text generation via context shifting
        // if we run out of context:
        // - take the n_keep first tokens from the original prompt (via p_state->n_past)
        // - take half of the last (p_state.n_ctx - n_keep) tokens and recompute the logits in batches
        if (p_state->n_past + (int) emb->embd_full.size() + std::max<unsigned int>(0, p_state->guidance_offset) > p_state->n_ctx) {
            if (p_state->params->n_predict == -2) {
                LOG("\n\n%s: context full and n_predict == -%d => stopping\n", __func__, p_state->params->n_predict);
                return -1;
            }

            const int n_left = p_state->n_past - p_state->params->n_keep;
            const int n_discard = n_left / 2;

            LOG("context full, swapping: p_state->n_past = %d, n_left = %d, p_state.n_ctx = %d, n_keep = %d, n_discard = %d\n",
                p_state->n_past, n_left, p_state->n_ctx, p_state->params->n_keep, n_discard);

            llama_kv_cache_seq_rm(state->ctx, 0, p_state->params->n_keep, p_state->params->n_keep + n_discard);
            llama_kv_cache_seq_add(state->ctx, 0, p_state->params->n_keep + n_discard, p_state->n_past, -n_discard);

            p_state->n_past -= n_discard;

            if (p_state->ctx_guidance) {
                p_state->n_past_guidance -= n_discard;
            }

            LOG("after swap: p_state->n_past = %d, p_state->p_state->n_past_guidance = %d\n", p_state->n_past, p_state->n_past_guidance);

            LOG("p_state->embd_full: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, emb->embd_full).c_str());

            LOG("clear session path\n");
        }
    } else {
        // context extension via Self-Extend
        // group-attention state
        // number of grouped KV tokens so far (used only if p_state->params->grp_attn_n > 1)
        const int ga_w = p_state->params->grp_attn_w;
        const int ga_n = p_state->params->grp_attn_n;
        int ga_i = 0;
        while (p_state->n_past >= ga_i + ga_w) {
            const int ib = (ga_n * ga_i) / ga_w;
            const int bd = (ga_w / ga_n) * (ga_n - 1);
            const int dd = (ga_w / ga_n) - ib * bd - ga_w;

            LOG("\n");
            LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i, p_state->n_past, ib * bd, ga_i + ib * bd,
                p_state->n_past + ib * bd);
            LOG("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n,
                (ga_i + ib * bd) / ga_n, (ga_i + ib * bd + ga_w) / ga_n);
            LOG("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", ga_i + ib * bd + ga_w, p_state->n_past + ib * bd, dd,
                ga_i + ib * bd + ga_w + dd, p_state->n_past + ib * bd + dd);

            llama_kv_cache_seq_add(state->ctx, 0, ga_i, p_state->n_past, ib * bd);
            llama_kv_cache_seq_div(state->ctx, 0, ga_i + ib * bd, ga_i + ib * bd + ga_w, ga_n);
            llama_kv_cache_seq_add(state->ctx, 0, ga_i + ib * bd + ga_w, p_state->n_past + ib * bd, dd);

            p_state->n_past -= bd;

            ga_i += ga_w / ga_n;

            LOG("\np_state->n_past_old = %d, p_state->n_past = %d, ga_i = %d\n\n", p_state->n_past + bd, p_state->n_past, ga_i);
        }
    }

    //TODO: try to reuse a matching prefix from the loaded session instead of re-eval (via p_state->n_past)

    // evaluate tokens in batches
    // p_state->embd_full is typically prepared beforehand to fit within a batch, but not always
    if (p_state->ctx_guidance) {
        unsigned int input_size;
        llama_token *input_buf;

        if (p_state->n_past_guidance < (int) emb->guidance_inp.size()) {
            // Guidance context should have the same data with these modifications:
            //
            // * Replace the initial prompt
            // * Shift everything by guidance_offset
            emb->embd_guidance = emb->guidance_inp;
            if (emb->embd_full.begin() + p_state->original_prompt_len < emb->embd_full.end()) {
                emb->embd_guidance.insert(
                        emb->embd_guidance.end(),
                        emb->embd_full.begin() + p_state->original_prompt_len,
                        emb->embd_full.end()
                );
            }

            input_buf = emb->embd_guidance.data();
            input_size = emb->embd_guidance.size();

            LOG("guidance context: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, emb->embd_guidance).c_str());
        } else {
            input_buf = emb->embd_full.data();
            input_size = emb->embd_full.size();
        }

        for (int i = 0; i < input_size; i += p_state->params->n_batch) {
            int n_eval = std::min((int) input_size - i, p_state->params->n_batch);
            if (llama_decode(p_state->ctx_guidance, llama_batch_get_one(input_buf + i, n_eval, p_state->n_past_guidance, 0))) {
                LOG("%s : failed to eval\n", __func__);
                return 1;
            }

            p_state->n_past_guidance += n_eval;
        }
    }

    for (int i = 0; i < (int) emb->embd_full.size(); i += p_state->params->n_batch) {
        int n_eval = (int) emb->embd_full.size() - i;
        if (n_eval > p_state->params->n_batch) {
            n_eval = p_state->params->n_batch;
        }

        LOG("eval: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, emb->embd_full).c_str());

        if (llama_decode(state->ctx, llama_batch_get_one(&emb->embd_full[i], n_eval, p_state->n_past, 0))) {
            LOG("%s : failed to eval\n", __func__);
            return 1;
        }

        p_state->n_past += n_eval;

        LOG("p_state->n_past = %d\n", p_state->n_past);
        // Display total tokens alongside total time
        if (p_state->params->n_print > 0 && p_state->n_past % p_state->params->n_print == 0) {
            LOG("%sTokens consumed so far = %d / %d %s\n", ANSI_COLOR_BLUE, p_state->n_past, p_state->n_ctx, ANSI_COLOR_RESET);
        }
    }

    return 0;

    //TODO: save embedding tokens to the session tokens
}

void process_input(struct go_llama_state * state, struct go_llama_predict_state * p_state) {
    auto emb = (struct embeddings *) p_state->embeddings;
    // if the input is consumed there is no user interaction
    //TODO: optionally save the session on first sample (for faster prompt loading next time)
    const llama_token id = llama_sampling_sample(p_state->ctx_sampling, state->ctx, p_state->ctx_guidance);
    // deal with end of generation tokens in interactive mode
    if (token_is_eog(state->model, emb->eot_ids, id) ||
            token_is_anti_prompt(state, id, emb->anti_prompt_ids)) {
        if (p_state->params->interactive) {
            p_state->is_interacting = true;
        }
        p_state->is_anti_prompt = true;
    }

    llama_sampling_accept(p_state->ctx_sampling, state->ctx, id, true);

    LOG("last: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, p_state->ctx_sampling->prev).c_str());

    emb->embd_full.push_back(id);
    if (!p_state->is_anti_prompt) emb->embd_out.push_back(id);
    p_state->input_echo = true;

    // decrement remaining sampling budget
    --p_state->n_remain;

    LOG("n_remain: %d\n", p_state->n_remain);
}

void process_output(struct go_llama_state * state, struct go_llama_predict_state * p_state) {
    auto emb = (struct embeddings *) p_state->embeddings;
    // processing user input
    // some user input remains from prompt or interaction, forward it to processing
    LOG("embd_inp.size(): %d, n_total_consumed: %d\n", (int) emb->embd_in.size(), p_state->n_total_consumed);
    while ((int) emb->embd_in.size() > p_state->n_total_consumed) {
        emb->embd_full.push_back(emb->embd_in[p_state->n_total_consumed]);

        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        llama_sampling_accept(p_state->ctx_sampling, state->ctx, emb->embd_in[p_state->n_total_consumed], false);

        ++p_state->n_total_consumed;
        if ((int) emb->embd_full.size() >= state->params->n_batch) {
            break;
        }
    }
}

void process_interaction(struct go_llama_state * state, struct go_llama_predict_state * p_state) {
    auto emb = (struct embeddings *) p_state->embeddings;
    if (p_state->n_past <= 0 || !p_state->is_interacting) return;
    LOG("waiting for user input\n");

    if (state->params->input_prefix_bos) {
        LOG("adding input prefix BOS token\n");
        emb->embd_in.push_back(llama_token_bos(state->model));
    }

    if (!state->params->input_prefix.empty()) {
        LOG("prepending input prefix: '%s'\n", state->params->input_prefix.c_str());
    }

    std::string buffer = predictorInputCallback(p_state);
    p_state->display = true;

    // Add tokens to embd_full only if the input buffer is non-empty
    // Entering a empty line lets the user pass control back
    if (buffer.length() > 1) {
        // append input suffix if any
        if (!state->params->input_suffix.empty()) {
            LOG("appending input suffix: '%s'\n", state->params->input_suffix.c_str());
        }

        LOG("buffer: '%s'\n", buffer.c_str());

        const size_t original_size = emb->embd_in.size();

        if (state->params->escape) {
            process_escapes(buffer);
        }

        const auto line_pfx = ::llama_tokenize(state->ctx, state->params->input_prefix, false, true);
        const auto line_inp = ::llama_tokenize(state->ctx, buffer, false, false);
        const auto line_sfx = ::llama_tokenize(state->ctx, state->params->input_suffix, false, true);

        LOG("input tokens: %s\n", LOG_TOKENS_TOSTR_PRETTY(state->ctx, line_inp).c_str());

        emb->embd_in.insert(emb->embd_in.end(), line_pfx.begin(), line_pfx.end());
        emb->embd_in.insert(emb->embd_in.end(), line_inp.begin(), line_inp.end());
        emb->embd_in.insert(emb->embd_in.end(), line_sfx.begin(), line_sfx.end());

        if (state->params->display_prompt) {
            for (size_t i = original_size; i < emb->embd_in.size(); ++i) {
                predictorOutputCallback(p_state, llama_token_to_piece(state->ctx, emb->embd_in[i]).c_str());
            }
        }

        p_state->n_remain -= (int) line_inp.size();
        LOG("n_remain: %d\n", p_state->n_remain);
    } else {
        LOG("empty line, passing control back\n");
    }

    p_state->input_echo = false; // do not echo this again
}

int go_llama_predict(struct go_llama_state * state_ptr, struct go_llama_predict_state * p_state_ptr) {
    auto l_state = g_state = (go_llama_state *) state_ptr;
    auto p_state = g_predict_state = (go_llama_predict_state *) p_state_ptr;
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
    // Check prompt size
    auto emb = (struct embeddings *) p_state->embeddings;
    if (emb->embd_in.size() > p_state->n_ctx - 4) {
        LOG("%s: error: prompt is too long (%d tokens, max %d)\n", __func__, (int) emb->embd_in.size(), p_state->n_ctx - 4);
        return 1;
    }

    llama_numa_init(p_state->params->numa);
    p_state->ctx_sampling = llama_sampling_init(p_state->params->sparams);
    while ((p_state->n_remain != 0 && !p_state->is_anti_prompt) || p_state->params->interactive) {
        // predict
        if (!emb->embd_full.empty()) {
            int predicted = predict_embeddings(l_state, p_state);
            if (predicted > 0) {
                return 1;
            } else if (predicted < 0) {
                break;
            }
        }

        emb->embd_full.clear();
        emb->embd_out.clear();
        emb->embd_guidance.clear();

        if ((int) emb->embd_in.size() <= p_state->n_total_consumed && !p_state->is_interacting) {
            process_input(l_state, p_state);
        } else {
            process_output(l_state, p_state);
        }

        // display text
        if (p_state->input_echo && p_state->display) {
            for (auto id: p_state->params->display_prompt ? emb->embd_full : emb->embd_out) {
                const std::string token_str = llama_token_to_piece(l_state->ctx, id);
                predictorOutputCallback(p_state, token_str.c_str());
            }
        }
        // reset color to default if there is no pending user input
        if (p_state->input_echo && (int) emb->embd_in.size() == p_state->n_total_consumed) {
            p_state->display = true;
        }

        // if not currently processing queued inputs;
        if ((int) emb->embd_in.size() <= p_state->n_total_consumed) {
            process_interaction(l_state, p_state);

            if (p_state->n_past > 0) {
                if (p_state->is_interacting) {
                    llama_sampling_reset(p_state->ctx_sampling);
                }
                p_state->is_interacting = p_state->is_anti_prompt = false;
            }
        }

        // end of generation
        if (!emb->embd_full.empty() && token_is_eog(l_state->model, emb->eot_ids, emb->embd_full.back()) && !p_state->params->interactive) {
            LOG("[end of text]\n");
            break;
        }

        // In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
        // We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
        if (p_state->params->interactive && p_state->n_remain <= 0 && p_state->params->n_predict >= 0) {
            p_state->n_remain = p_state->params->n_predict;
            p_state->is_interacting = true;
        }
    }
    predictorEndOutputCallback(p_state);
    llama_kv_cache_clear(l_state->ctx);
    //TODO: save output tokens to the session tokens
    return 0;
}

void go_llama_predict_free(struct go_llama_predict_state * p_state) {
    if (p_state->ctx_guidance) { llama_free(p_state->ctx_guidance); }
    if (p_state->ctx_sampling) llama_sampling_free(p_state->ctx_sampling);
    free(p_state);
}