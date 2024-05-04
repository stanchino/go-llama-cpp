#include "../../includes/common.h"
#include "../llama/llama.h"
#include "tokenizer.h"

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

go_llama_token go_llama_token_bos(struct go_llama_state *state) {
    return llama_token_bos(state->model);
}
bool go_llama_should_add_bos_token(struct go_llama_state *state) {
    return llama_should_add_bos_token(state->model);
}
