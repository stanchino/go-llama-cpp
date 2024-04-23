#include "../../includes/common.h"
#include "options.h"

void * go_llama_params_to_gpt_params(go_llama_params p) {
    gpt_params params;
    auto params_ptr = (gpt_params*) malloc(sizeof(gpt_params));
    params.model             = p.model;
    params.use_mmap          = p.use_mmap;
    params.interactive       = p.interactive;
    params.interactive_first = p.interactive_first;
    params.input_prefix      = p.input_prefix;
    params.input_suffix      = p.input_suffix;
    params.display_prompt    = p.display_prompt;
    if (p.antiprompt != nullptr) {
        auto antiprompts = (charArray*) p.antiprompt;
        for (unsigned int i = 0; i < antiprompts->len; i++) {
            params.antiprompt.emplace_back(antiprompts->data[i]);
        }
    }
    *params_ptr = params;
    return (void*) params_ptr;
}