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
	"errors"
	"fmt"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"sync"
	"unsafe"
)

type Predictor struct {
	*llama.GoLlama
	PredictState      unsafe.Pointer
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
	p.InitState()
	return p
}

func (p *Predictor) InitState() {
	state := unsafe.Pointer(C.go_llama_init_predict_state((*C.struct_go_llama_state)(p.State)))
	p.PredictState = state
	if !p.Options.InteractiveFirst && p.Options.Prompt != "" {
		p.SetPrompt(p.Options.Prompt)
	}
	predictors[uintptr(state)] = p
}

func (p *Predictor) SetPrompt(prompt string) {
	C.go_llama_set_prompt(
		(*C.struct_go_llama_state)(p.State),
		(*C.struct_go_llama_predict_state)(p.PredictState),
		C.CString(p.GoLlama.Options.ApplyTemplate(prompt)))
}

func (p *Predictor) SetOutputCallback(cb func(token string)) {
	p.outputCallback = cb
}

func (p *Predictor) SetEndOutputCallback(cb func()) {
	p.endOutputCallback = cb
}

func (p *Predictor) SetInputCallback(cb func() string) {
	if p.GoLlama.Options.InteractiveFirst {
		fmt.Println(
			"== Running in interactive mode. ==\n" +
				" - Press Ctrl+C to interject at any time.\\n" +
				" - Press Return to return control to LLaMa.\n" +
				" - To return control without starting a new line, end your input with '/'.\n" +
				" - If you want to submit another line, end your input with '\\'.")
		if p.GoLlama.Options.InteractiveFirst {
			p.SetPrompt(cb())
		}
	}
	p.inputCallback = cb
}

func (p *Predictor) WithPrompt(prompt string) *Predictor {
	p.SetPrompt(prompt)
	return p
}

func (p *Predictor) Predict() error {
	if p.GoLlama.Options.Interactive {
		if p.inputCallback == nil {
			return errors.New("no input callback provided in interactive mode")
		}
	}
	C.go_llama_predict(
		(*C.struct_go_llama_state)(p.GoLlama.State),
		(*C.struct_go_llama_predict_state)(p.PredictState))
	return nil
}

func (p *Predictor) OutputCallback(token string) {
	if p.outputCallback != nil {
		p.outputCallback(token)
	}
}
func (p *Predictor) InputCallback() string {
	if p.inputCallback != nil {
		return p.GoLlama.Options.ApplyTemplate(p.inputCallback())
	}
	return ""
}
func (p *Predictor) EndOutputCallback() {
	if p.endOutputCallback != nil {
		p.endOutputCallback()
	}
}

//export predictorInputCallback
func predictorInputCallback(statePtr *C.struct_go_llama_predict_state) *C.char {
	m.Lock()
	defer m.Unlock()
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		return C.CString(predictor.InputCallback())
	}
	return nil
}

func (p *Predictor) Free() {
	C.go_llama_predict_free((*C.struct_go_llama_predict_state)(p.PredictState))
	delete(predictors, uintptr(p.PredictState))
}

//export predictorOutputCallback
func predictorOutputCallback(statePtr *C.struct_go_llama_predict_state, token *C.cchar_t) {
	m.RLock()
	defer m.RUnlock()
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		predictor.OutputCallback(C.GoString(token))
	}
}

//export predictorEndOutputCallback
func predictorEndOutputCallback(statePtr *C.struct_go_llama_predict_state) {
	m.RLock()
	defer m.RUnlock()
	if predictor, ok := predictors[uintptr(unsafe.Pointer(statePtr))]; ok {
		predictor.EndOutputCallback()
	}
}
