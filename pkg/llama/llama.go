package llama

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#cgo LDFLAGS: -L${SRCDIR}/../../lib
#cgo darwin,arm64 LDFLAGS: -lllama_arm64 -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
#cgo linux,amd64 LDFLAGS: -lllama_x86_64 -lpthread
#include <stdlib.h>
#include <stdbool.h>
#include "llama.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"log"
	"strings"
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/cache"
	"github.com/stanchino/go-llama-cpp/pkg/decoder"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/sampling"
	"github.com/stanchino/go-llama-cpp/pkg/tokenizer"
)

type GoLlama struct {
	State     unsafe.Pointer
	Options   *options.Options
	Cache     *cache.Cache
	Decoder   *decoder.Decoder
	Sampling  *sampling.Sampling
	Tokenizer *tokenizer.Tokenizer
}

func NewGoLlama(opt *options.Options) (*GoLlama, error) {
	params, err := opt.Params()
	if err != nil {
		return nil, err
	}
	state := unsafe.Pointer(C.go_llama_init(params))

	return &GoLlama{
		Options:   opt,
		State:     state,
		Cache:     cache.NewCache(state),
		Decoder:   decoder.NewDecoder(state),
		Sampling:  sampling.NewSampling(state),
		Tokenizer: tokenizer.NewTokenizer(state),
	}, nil
}

func (l *GoLlama) StringifyTokens(tokens []int) string {
	var result string
	for k, v := range tokens {
		if v == 0 {
			continue
		}
		result += fmt.Sprintf("'%s':%d", strings.Replace(l.Tokenizer.ToString([]int{v}), "\n", "\\n", -1), v)
		if k < len(tokens)-1 {
			result += ", "
		}
	}
	return fmt.Sprintf("[%s]", result)
}

func (l *GoLlama) DecodeInBatches(emb []int, past int, batchSize int) (int, error) {
	// log.Println("predict in batches, tokens: ", l.StringifyTokens(emb))
	embLen := len(emb)
	maxEmbSize := l.Options.ContextSize - 4
	if embLen > maxEmbSize {
		log.Printf("input too long, skipped to %d tokens\n", embLen-maxEmbSize)
		emb = emb[embLen-maxEmbSize:]
		embLen = maxEmbSize
	}
	// Ensure the output doesn't exceed the context size by truncating embd_guidance if necessary
	if l.Options.GroupAttnFactor == 1 {
		// infinite text generation via context shifting
		// if we run out of context:
		// - take the n_keep first tokens from the original prompt (via p_state->n_past)
		// - take half of the last (p_state.n_ctx - n_keep) tokens and recompute the logits in batches
		if past+embLen > l.Options.ContextSize {
			if l.Options.NumPredict == -2 {
				log.Printf("context full and n_predict == -%d => stopping\n", l.Options.NumPredict)
				return past, errors.New("context full")
			}
			numLeft := past - l.Options.NumKeep
			numDiscard := numLeft / 2

			log.Printf("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
				past, numLeft, l.Options.ContextSize, l.Options.NumKeep, numDiscard)

			l.Cache.Remove(0, l.Options.NumKeep, l.Options.NumKeep+numDiscard)
			l.Cache.Add(0, l.Options.NumKeep+numDiscard, past, -numDiscard)

			past -= numDiscard
			log.Printf("after swap: n_past = %d, embd: %s\n", past, l.StringifyTokens(emb))
		}
	} else {
		// context extension via Self-Extend
		// group-attention state
		// number of grouped KV tokens so far (used only if l.Options.GroupAttnFactor > 1)
		grpAttnId := 0
		grpAttnNum := l.Options.GroupAttnFactor
		grpAttnWeight := l.Options.GroupAttnWeight
		for past >= grpAttnId+grpAttnWeight {
			ib := (grpAttnNum * grpAttnId) / grpAttnWeight
			bd := (grpAttnWeight / grpAttnNum) * (grpAttnNum - 1)
			dd := (grpAttnWeight / grpAttnNum) - ib*bd - grpAttnWeight

			log.Printf("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", grpAttnId, past, ib*bd, grpAttnId+ib*bd,
				past+ib*bd)
			log.Printf("div:   [%6d, %6d] / %6d -> [%6d, %6d]\n", grpAttnId+ib*bd, grpAttnId+ib*bd+grpAttnWeight, grpAttnNum,
				(grpAttnId+ib*bd)/grpAttnNum, (grpAttnId+ib*bd+grpAttnWeight)/grpAttnNum)
			log.Printf("shift: [%6d, %6d] + %6d -> [%6d, %6d]\n", grpAttnId+ib*bd+grpAttnWeight, past+ib*bd, dd,
				grpAttnId+ib*bd+grpAttnWeight+dd, past+ib*bd+dd)

			l.Cache.Add(0, grpAttnId, past, ib*bd)
			l.Cache.Div(0, grpAttnId+ib*bd, grpAttnId+ib*bd+grpAttnWeight, grpAttnNum)
			l.Cache.Add(0, grpAttnId+ib*bd+grpAttnWeight, past+ib*bd, dd)

			past -= bd

			grpAttnId += grpAttnWeight / grpAttnNum

			log.Printf("\np_state->n_past_old = %d, p_state->n_past = %d, ga_i = %d\n\n", past+bd, past, grpAttnId)
		}
	}
	for i := 0; i < embLen; i += batchSize {
		eval := embLen - i
		if eval > batchSize {
			eval = batchSize
		}
		err := l.Decoder.DecodeBatch(l.Tokenizer.ToTokensList(emb), i, eval, past)
		if err != nil {
			return past, err
		}
		past += eval
	}
	return past, nil
}

func (l *GoLlama) Free() {
	C.go_llama_free((*C.struct_go_llama_state)(l.State))
	C.free(l.State)
}
