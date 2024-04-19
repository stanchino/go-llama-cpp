package llama

/*
#cgo CFLAGS: -I./llama.cpp/common -I./llama.cpp -I.
#cgo CXXFLAGS: -I./llama.cpp/common -I./llama.cpp -I.
#cgo LDFLAGS: -L./llama.cpp -lllama
#cgo darwin LDFLAGS: -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#cgo darwin CXXFLAGS: -std=c++11
#include <stdlib.h>
#include "llama.h"
#include "go_llama.h"
*/
import "C"
import (
    "unsafe"
)

type GoLlama struct {
    State   unsafe.Pointer
    Params  unsafe.Pointer
    Options Options
}

func NewGoLlama(options Options) *GoLlama {
    params := options.ToInitParams()
    state := C.go_llama_init(options.ToInitParams())
    return &GoLlama{
        State:   unsafe.Pointer(state),
        Params:  params,
        Options: options,
    }
}

func (l *GoLlama) Free() {
    C.go_llama_free(l.State)
    C.free(l.State)
}
