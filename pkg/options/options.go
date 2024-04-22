package options

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdbool.h>
#include <stdlib.h>
#include "options.h"
#include "../util/util.h"
*/
import "C"
import (
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/util"
)

type Options struct {
	ModelName        string
	UseMMap          bool
	AntiPrompts      []string
	AntiPromptsArray *util.CharArray
	InitParams       unsafe.Pointer
}

func (o *Options) ToInitParams() unsafe.Pointer {
	options := C.go_llama_params{
		model:    C.CString(o.ModelName),
		use_mmap: C.bool(o.UseMMap),
	}
	if len(o.AntiPrompts) > 0 {
		o.AntiPromptsArray = util.NewCharArray(o.AntiPrompts)
		options.antiprompt = o.AntiPromptsArray.Inner
	}
	o.InitParams = unsafe.Pointer(&options)
	return o.InitParams
}

func (o *Options) Free() {
	o.AntiPromptsArray.Free()
}
