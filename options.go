package llama

/*
#cgo CFLAGS: -I./llama.cpp/common -I./llama.cpp -I.
#cgo CXXFLAGS: -I./llama.cpp/common -I./llama.cpp -I.
#cgo darwin CXXFLAGS: -std=c++11
#include <stdbool.h>
#include "options.h"
*/
import "C"
import "unsafe"

type Options struct {
    ModelName string
    UseMMap   bool
}

func (o *Options) ToInitParams() unsafe.Pointer {
    return unsafe.Pointer(&C.go_llama_params{
        model:    C.CString(o.ModelName),
        use_mmap: C.bool(o.UseMMap),
    })
}
