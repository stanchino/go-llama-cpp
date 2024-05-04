package decoder

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include "../llama/llama.h"
#include "decoder.h"
*/
import "C"
import (
	"errors"
	"unsafe"
)

type Decoder struct {
	State unsafe.Pointer
}

func NewDecoder(state unsafe.Pointer) *Decoder {
	return &Decoder{
		State: state,
	}
}

func (d *Decoder) DecodeBatch(embPtr unsafe.Pointer, pos int, numTokens int, startPos int) error {
	if ok := int(C.go_llama_decode_batch(
		(*C.struct_go_llama_state)(d.State),
		*(*C.tokens_list)(embPtr),
		C.int(pos),
		C.int(numTokens),
		C.int(startPos))); ok == 1 {
		return errors.New("failed to decode batch")
	}
	return nil
}
