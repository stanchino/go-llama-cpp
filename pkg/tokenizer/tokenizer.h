#ifndef BINDING_TOKENIZER_H
#define BINDING_TOKENIZER_H
#ifdef __cplusplus

#include <vector>
#include <string>
#include "../llama/llama.h"

extern "C" {
#endif
typedef int go_llama_token;
typedef struct tokens_list {
    unsigned long size;
    go_llama_token *tokens;
} tokens_list;
go_llama_token go_llama_token_bos(struct go_llama_state *state);
bool go_llama_should_add_bos_token(struct go_llama_state *state);
struct tokens_list go_llama_tokenize(struct go_llama_state *state, const char *prompt, bool add_special, bool parse_special);
const char *go_llama_token_to_piece(struct go_llama_state *state, const go_llama_token *tokens, unsigned int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_TOKENIZER_H