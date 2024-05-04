package sampling

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdbool.h>
#include "../llama/llama.h"
*/
import "C"
import (
	"log"
	"unsafe"
)

type Sampling struct {
	State unsafe.Pointer
}

func NewSampling(state unsafe.Pointer) *Sampling {
	return &Sampling{
		State: state,
	}
}

func (s *Sampling) Init() {
	state := (*C.go_llama_state)(s.State)
	state.ctx_sampling = C.go_llama_sampling_init((*C.go_llama_state)(s.State))
}
func (s *Sampling) Sample() int {
	return int(C.go_llama_sampling_sample((*C.go_llama_state)(s.State)))
}
func (s *Sampling) Accept(id int, applyGrammar bool) {
	C.go_llama_sampling_accept(
		(*C.go_llama_state)(s.State),
		C.int(id),
		C.bool(applyGrammar))
}
func (s *Sampling) Prev() unsafe.Pointer {
	return unsafe.Pointer(C.go_llama_sampling_prev(
		(*C.go_llama_state)(s.State)))
}
func (s *Sampling) Reset() {
	log.Println("resetting sampling")
	C.go_llama_sampling_reset((*C.go_llama_state)(s.State))
}
