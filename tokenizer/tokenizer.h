#ifndef BINDING_MAIN_H
#define BINDING_MAIN_H
#ifdef __cplusplus
#include <vector>
#include <string>
#include "llama.h"
#include "common.h"
#include "go_llama.h"
extern "C" {
#endif
typedef struct tokens_list {
    unsigned long size;
    llama_token * tokens;
} tokens_list;
struct tokens_list go_llama_tokenize(void * state_ptr, const char * prompt);
const char * go_llama_token_to_piece(void * state_ptr, const llama_token * tokens, unsigned int size);
#ifdef __cplusplus
}
#endif
#endif //BINDING_MAIN_H