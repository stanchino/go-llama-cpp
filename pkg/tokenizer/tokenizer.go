package tokenizer

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include "../options/options.h"
#include "../llama/llama.h"
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

func (t *Tokenizer) TokenBos() int {
	return int(C.go_llama_token_bos((*C.struct_go_llama_state)(t.State)))
}

func (t *Tokenizer) AddBos() bool {
	return bool(C.go_llama_should_add_bos_token((*C.struct_go_llama_state)(t.State)))
}

func (t *Tokenizer) Tokenize(text string, special ...bool) []int {
	addSpecial := false
	parseSpecial := false
	if len(special) == 1 {
		addSpecial = special[0]
	}
	if len(special) == 2 {
		addSpecial = special[0]
		parseSpecial = special[1]
	}
	tokenList := C.go_llama_tokenize(
		(*C.struct_go_llama_state)(t.State),
		C.CString(text),
		C.bool(addSpecial),
		C.bool(parseSpecial))
	return t.ToSlice(unsafe.Pointer(&tokenList))

}

func (t *Tokenizer) ToSlice(tokenListPtr unsafe.Pointer) []int {
	tokenList := (*C.tokens_list)(tokenListPtr)
	tokens := unsafe.Slice(tokenList.tokens, tokenList.size)
	result := make([]int, tokenList.size)
	for i, t := range tokens {
		result[i] = int(t)
	}
	return result
}

func (t *Tokenizer) ToTokensList(tokens []int) unsafe.Pointer {
	result := make([]C.go_llama_token, len(tokens))
	for i, t := range tokens {
		result[i] = C.go_llama_token(t)
	}
	tokensList := C.tokens_list{
		size:   C.ulong(len(tokens)),
		tokens: &result[0],
	}

	return unsafe.Pointer(&tokensList)
}

func (t *Tokenizer) ToString(tokens []int) string {
	if len(tokens) == 0 {
		return ""
	}
	result := make([]C.go_llama_token, len(tokens))
	for i, t := range tokens {
		result[i] = C.go_llama_token(t)
	}
	str := C.go_llama_token_to_piece((*C.struct_go_llama_state)(t.State), &result[0], C.uint(len(tokens)))
	return C.GoString(str)
}
