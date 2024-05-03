package llama

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -L${SRCDIR}/../../lib
#cgo darwin,arm64 LDFLAGS: -lllama_arm64 -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#cgo linux,amd64 LDFLAGS: -lllama_x86_64 -lpthread
#include <stdlib.h>
#include <stdbool.h>
#include "../util/util.h"
#include "../options/options.h"
#include "llama.h"
*/
import "C"
import (
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"unsafe"
)

type GoLlama struct {
	State   unsafe.Pointer
	Options *options.Options
}

func NewGoLlama(opt *options.Options) (*GoLlama, error) {
	params, err := opt.ToInitParams()
	if err != nil {
		return nil, err
	}
	return &GoLlama{
		State:   unsafe.Pointer(C.go_llama_init(params)),
		Options: opt,
	}, nil
}

func (l *GoLlama) Free() {
	C.go_llama_free((*C.struct_go_llama_state)(l.State))
	C.free(l.State)
}
