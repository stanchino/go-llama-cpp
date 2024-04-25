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
	"fmt"
	"github.com/stanchino/go-llama-cpp/pkg/util"
	"unsafe"
)

type Options struct {
	ModelName        string
	UseMMap          bool
	ContextSize      int
	Interactive      bool
	InteractiveFirst bool
	DisplayPrompt    bool
	InputPrefix      string
	InputSuffix      string
	Template         string
	AntiPrompts      []string
	EndOfTextPrompts []string
}

func (o *Options) ToInitParams() unsafe.Pointer {
	return unsafe.Pointer(&C.go_llama_params{
		model:             C.CString(o.ModelName),
		use_mmap:          C.bool(o.UseMMap),
		interactive:       C.bool(o.Interactive),
		interactive_first: C.bool(o.InteractiveFirst),
		display_prompt:    C.bool(o.DisplayPrompt),
		n_ctx:             C.int(o.ContextSize),
		input_prefix:      C.CString(o.InputPrefix),
		input_suffix:      C.CString(o.InputSuffix),
		anti_prompts:      (*C.charArray)(util.NewCharArray(o.AntiPrompts).Pointer),
		eot_prompts:       (*C.charArray)(util.NewCharArray(o.EndOfTextPrompts).Pointer),
	})
}

func (o *Options) ApplyTemplate(str string) string {
	if o.Template == "" || str == "" {
		return str
	}
	return fmt.Sprintf(o.Template, str)
}
