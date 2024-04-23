package predictor

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include "../options/options.h"
#include "../llama/llama.h"
#include "predictor.h"
*/
import "C"

import (
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"sync"
	"unsafe"
)

type Predictor struct {
	*llama.GoLlama
	outputCallback    func(string)
	inputCallback     func() string
	endOutputCallback func()
}

var (
	m          sync.RWMutex
	predictors = map[uintptr]*Predictor{}
)

func NewPredictor(l *llama.GoLlama) *Predictor {
	p := &Predictor{
		GoLlama: l,
	}
	predictors[uintptr(p.State)] = p
	return p
}

func (p *Predictor) SetOutputCallback(cb func(token string)) {
	p.outputCallback = cb
}

func (p *Predictor) SetEndOutputCallback(cb func()) {
	p.endOutputCallback = cb
}

func (p *Predictor) SetInputCallback(cb func() string) {
	p.inputCallback = cb
}

func (p *Predictor) Predict(prompt string) {
	C.go_llama_predict((*C.struct_go_llama_state)(p.State), C.CString(prompt))
}

func (p *Predictor) OutputCallback(token string) {
	if p.outputCallback != nil {
		p.outputCallback(token)
	}
}
func (p *Predictor) InputCallback() string {
	if p.inputCallback != nil {
		return p.inputCallback()
	}
	return ""
}
func (p *Predictor) EndOutputCallback() {
	if p.endOutputCallback != nil {
		p.endOutputCallback()
	}
}

//export predictorInputCallback
func predictorInputCallback(statePtr *C.struct_go_llama_state) *C.char {
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		return C.CString(predictor.InputCallback())
	}
	return nil
}

//export predictorOutputCallback
func predictorOutputCallback(statePtr *C.struct_go_llama_state, token *C.cchar_t) {
	m.RLock()
	defer m.RUnlock()
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		predictor.OutputCallback(C.GoString(token))
	}
}

//export predictorEndOutputCallback
func predictorEndOutputCallback(statePtr *C.struct_go_llama_state) {
	m.RLock()
	defer m.RUnlock()
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		predictor.EndOutputCallback()
	}
}
