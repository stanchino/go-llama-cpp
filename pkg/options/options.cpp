#include "../../includes/common.h"
#include "options.h"

void *go_llama_params_to_gpt_params(go_llama_params p) {
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
    if (p.n_ctx > 0) {
          params_ptr->n_ctx = p.n_ctx;
      }
      if (p.rope_freq_base > 0) {
          params_ptr->rope_freq_base = p.rope_freq_base;
      }
      if (p.rope_freq_scale > 0) {
          params_ptr->rope_freq_scale = p.rope_freq_scale;
      }
    if (p.anti_prompts != nullptr) {
        auto anti_prompts = (charArray *) p.anti_prompts;
        for (unsigned int i = 0; i < anti_prompts->len; i++) {
            params.antiprompt.emplace_back(anti_prompts->data[i]);
        }
    }
    *params_ptr = params;
    return (void *) params_ptr;
}