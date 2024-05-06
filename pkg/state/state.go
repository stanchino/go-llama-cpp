package state

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include "../llama/llama.h"
#include <sys/mman.h>
*/
import "C"
import (
	"log"
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/options"
)

func NewState(opt *options.Options) (unsafe.Pointer, error) {
	params, err := opt.Params()
	if err != nil {
		return nil, err
	}
	C.go_llama_backend_init()
	model := C.go_llama_load_model_from_file((*C.struct_go_llama_params)(params))
	if model == nil {
		log.Fatalf("go_llama_model_init: error: failed to load model '%s'\n", opt.Model)
	}
	context := C.go_llama_new_context_with_model((*C.struct_llama_model)(model), (*C.struct_go_llama_params)(params))
	if context == nil {
		log.Fatalf("go_llama_model_init: error: ailed to create context with model '%s'\n", opt.Model)
	}

	return unsafe.Pointer(C.go_llama_init_state(model, context, (*C.struct_go_llama_params)(params))), nil
}
