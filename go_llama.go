package llama

/*
#cgo CFLAGS: -std=c11 -I${SRCDIR}/includes/
#cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/includes/
#include <stdlib.h>
#include <stdbool.h>
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
