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
	Interactive      bool
	InteractiveFirst bool
	DisplayPrompt    bool
	InputPrefix      string
	InputSuffix      string
	AntiPrompts      []string
}

func (o *Options) ToInitParams() unsafe.Pointer {
	options := C.go_llama_params{
		model:             C.CString(o.ModelName),
		use_mmap:          C.bool(o.UseMMap),
		interactive:       C.bool(o.Interactive),
		interactive_first: C.bool(o.InteractiveFirst),
		display_prompt:    C.bool(o.DisplayPrompt),
		input_prefix:      C.CString(o.InputPrefix),
		input_suffix:      C.CString(o.InputSuffix),
	}
	if len(o.AntiPrompts) > 0 {
		arr := util.NewCharArray(o.AntiPrompts)
		defer arr.Free()
		options.antiprompt = (*C.charArray)(arr.Pointer)
	}
	return unsafe.Pointer(C.go_llama_params_to_gpt_params(options))
}
