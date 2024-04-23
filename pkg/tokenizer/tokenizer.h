#ifndef BINDING_TOKENIZER_H
#define BINDING_TOKENIZER_H
#ifdef __cplusplus
#include <vector>
#include <string>
#include "../llama/llama.h"
extern "C" {
#endif
typedef int32_t go_llama_token;
typedef struct tokens_list {
    unsigned long size;
    go_llama_token * tokens;
} tokens_list;
struct tokens_list go_llama_tokenize(struct go_llama_state * state, const char * prompt);
const char * go_llama_token_to_piece(struct go_llama_state * state, const go_llama_token * tokens, unsigned int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_TOKENIZER_H