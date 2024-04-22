#include "tokenizer.h"
#include "../../includes/common.h"

struct tokens_list go_llama_tokenize(void * state_ptr, const char * prompt) {
    auto state = (go_llama_state*) state_ptr;
    const bool add_bos = llama_should_add_bos_token(llama_get_model(state->ctx));
    std::vector<llama_token> tokens = llama_tokenize(state->ctx, prompt, add_bos);
    unsigned int size = tokens.size();
    struct tokens_list list = {
            size,
            (go_llama_token *) malloc(sizeof(llama_token) * size)
    };
    for (unsigned int i = 0; i < size; i++) {
        list.tokens[i] = tokens[i];
    }
    return list;
}

const char * go_llama_token_to_piece(void * state_ptr, const llama_token * tokens, unsigned int size) {
    auto state = (go_llama_state*) state_ptr;
    std::string token_str;
    for (unsigned int i = 0; i < size; i++) {
        token_str.append(llama_token_to_piece(state->ctx, tokens[i]));
    }
    unsigned long token_str_size = token_str.size();
    char * res = (char *) malloc(token_str_size + 1);
    strncpy(res, token_str.c_str(), token_str_size);
    return res;
}
