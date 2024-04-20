package tokenizer

/*
#cgo CFLAGS: -std=c11 -I${SRCDIR}/../llama/ -I${SRCDIR}/../../includes/
#cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/../llama/ -I${SRCDIR}/../../includes/
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "go_llama.h"
#include "tokenizer.h"
*/
import "C"
import (
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"unsafe"
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
	result := make([]C.go_llama_token, len(tokens))
	for i, t := range tokens {
		result[i] = C.go_llama_token(t)
	}
	str := C.go_llama_token_to_piece(t.State, &result[0], C.uint(len(tokens)))
	return C.GoString(str)
}
