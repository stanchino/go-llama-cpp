package util

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include "util.h"
*/
import "C"
import "unsafe"

type CharArray struct {
	Inner unsafe.Pointer
}

func NewCharArray(arr []string) *CharArray {
	length := C.ulong(len(arr))
	charArray := C.makeCharArray(length)
	for i, s := range arr {
		C.setArrayString(charArray, C.CString(s), C.ulong(i))
	}
	return &CharArray{
		Inner: unsafe.Pointer(charArray),
	}
}

func (c *CharArray) Free() {
	C.freeCharArray((*C.charArray)(c.Inner))
}
