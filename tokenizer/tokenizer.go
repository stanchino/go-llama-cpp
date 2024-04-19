package tokenizer

/*
#cgo CFLAGS: -I../llama.cpp/common -I../llama.cpp -I.. -I.
#cgo CXXFLAGS: -I../llama.cpp/common -I../llama.cpp -I.. -I.
#cgo darwin CXXFLAGS: -std=c++11
#include <stdlib.h>
#include "llama.h"
#include "go_llama.h"
#include "tokenizer.h"
*/
import "C"
import (
    "unsafe"

    "github.com/stanchino/go-llama-cpp"
)

type Tokenizer struct {
    *llama.GoLlama
}

func NewTokenizer(l *llama.GoLlama) *Tokenizer {
    return &Tokenizer{
        GoLlama: l,
    }
}

func (t *Tokenizer) Tokenize(text string) []int {
    var tokenList C.tokens_list = C.go_llama_tokenize(t.State, C.CString(text))
    tokens := unsafe.Slice(tokenList.tokens, tokenList.size)
    result := make([]int, tokenList.size)
    for i, t := range tokens {
        result[i] = int(t)
    }
    C.free(unsafe.Pointer(tokenList.tokens))
    return result
}

func (t *Tokenizer) ToString(tokens []int) string {
    result := make([]C.llama_token, len(tokens))
    for i, t := range tokens {
        result[i] = C.llama_token(t)
    }
    str := C.go_llama_token_to_piece(t.State, &result[0], C.uint(len(tokens)))
    return C.GoString(str)
}
