package cache

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include "../llama/llama.h"
#include "cache.h"
*/
import "C"
import "unsafe"

type Cache struct {
	State unsafe.Pointer
}

func NewCache(state unsafe.Pointer) *Cache {
	return &Cache{
		State: state,
	}
}

func (c *Cache) Remove(seqId int, p0 int, p1 int) {
	C.go_llama_kv_cache_seq_rm((*C.go_llama_state)(c.State), C.int(seqId), C.int(p0), C.int(p1))
}

func (c *Cache) Add(seqId int, p0 int, p1 int, delta int) {
	C.go_llama_kv_cache_seq_add((*C.go_llama_state)(c.State), C.int(seqId), C.int(p0), C.int(p1), C.int(delta))
}

func (c *Cache) Div(seqId int, p0 int, p1 int, d int) {
	C.go_llama_kv_cache_seq_div((*C.go_llama_state)(c.State), C.int(seqId), C.int(p0), C.int(p1), C.int(d))
}
