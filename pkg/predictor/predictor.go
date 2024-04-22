package predictor

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include "../llama/llama.h"
#include "predictor.h"
*/
import "C"

import (
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"log"
	"sync"
)

type Predictor struct {
	*llama.GoLlama
}

var (
	m        sync.RWMutex
	callback func(string) error
)

func NewPredictor(l *llama.GoLlama) *Predictor {
	return &Predictor{
		GoLlama: l,
	}
}

func (p *Predictor) SetTokenCallback(cb func(token string) error) {
	callback = cb
}

func (p *Predictor) Predict(prompt string, result chan string) {
	p.SetTokenCallback(func(token string) error {
		result <- token
		return nil
	})
	C.go_llama_predict(p.State, C.CString(prompt))
}

//export tokenCallback
func tokenCallback(token *C.cchar_t) {
	m.RLock()
	defer m.RUnlock()
	if err := callback(C.GoString(token)); err != nil {
		log.Println(err)
	}
}