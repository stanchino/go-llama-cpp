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
#include "llama.h"
*/
import "C"
import (
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/options"
)

type GoLlama struct {
	State   unsafe.Pointer
	Options options.Options
}

func NewGoLlama(options options.Options) *GoLlama {
	params := options.ToInitParams()
	state := C.go_llama_init(params)
	return &GoLlama{
		State:   unsafe.Pointer(state),
		Options: options,
	}
}

func (l *GoLlama) Free() {
	l.Options.Free()
	C.go_llama_free(l.State)
	C.free(l.State)
}
