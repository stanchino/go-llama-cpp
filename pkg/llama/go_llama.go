package llama

/*
#cgo CFLAGS: -std=c11 -I${SRCDIR}/../../includes/
#cgo CXXFLAGS: -std=c++11 -I${SRCDIR}/../../includes/
#cgo LDFLAGS: -L${SRCDIR}/../../lib
#cgo darwin,arm64 LDFLAGS: -lllama_arm64 -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#cgo linux,amd64 LDFLAGS: -lllama_amd64 -lpthread
#include <stdlib.h>
#include <stdbool.h>
#include "go_llama.h"
*/
import "C"
import (
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/options"
)

type GoLlama struct {
	State   unsafe.Pointer
	Params  unsafe.Pointer
	Options options.Options
}

func NewGoLlama(options options.Options) *GoLlama {
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