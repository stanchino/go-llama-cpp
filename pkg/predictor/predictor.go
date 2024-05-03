package predictor

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdlib.h>
#include <stdbool.h>
#include "../options/options.h"
#include "../llama/llama.h"
#include "../tokenizer/tokenizer.h"
#include "predictor.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/tokenizer"
)

type Predictor struct {
	*llama.GoLlama
	Tokenizer         *tokenizer.Tokenizer
	PredictState      unsafe.Pointer
	IsInteracting     bool
	IsExiting         bool
	eogTokens         []int
	outputCallback    func(string)
	inputCallback     func() string
	endOutputCallback func()
	cancel            chan os.Signal
	reading           chan string
}

var (
	m          sync.RWMutex
	predictors = map[uintptr]*Predictor{}
)

func NewPredictor(l *llama.GoLlama) *Predictor {
	p := &Predictor{
		GoLlama:       l,
		IsInteracting: false,
		IsExiting:     false,
		Tokenizer:     tokenizer.NewTokenizer(l),
	}
	p.SetEndOfGenerationTokens()
	p.InitState()
	return p
}

func (p *Predictor) SetEndOfGenerationTokens() {
	if len(p.Options.EndOfTextPrompts) > 0 {
		for _, t := range p.Options.EndOfTextPrompts {
			p.eogTokens = append(p.eogTokens, p.Tokenizer.Tokenize(t, false, true)...)
		}
	}
	if len(p.Options.AntiPrompts) > 0 {
		for _, t := range p.Options.AntiPrompts {
			p.eogTokens = append(p.eogTokens, p.Tokenizer.Tokenize(t, false, true)...)
		}
	}
}

func (p *Predictor) InitState() {
	p.PredictState = unsafe.Pointer(C.go_llama_init_predict_state((*C.struct_go_llama_state)(p.State)))
	p.SamplingInit()
	predictors[uintptr(p.PredictState)] = p
}

func (p *Predictor) SetOutputCallback(cb func(token string)) {
	p.outputCallback = cb
}

func (p *Predictor) SetEndOutputCallback(cb func()) {
	p.endOutputCallback = cb
}

func (p *Predictor) SetInputCallback(cb func() string) {
	if p.GoLlama.Options.InteractiveFirst {
		fmt.Println(
			"== Running in interactive mode. ==\n" +
				" - Press Ctrl+C to interject at any time.\\n" +
				" - Press Return to return control to the model.\n" +
				" - To return control without starting a new line, end your input with '/'.\n" +
				" - If you want to submit another line, end your input with '\\'.")
	}
	p.inputCallback = cb
}

func (p *Predictor) WithPrompt(prompt string) *Predictor {
	p.Options.Prompt = prompt
	return p
}

func (p *Predictor) Predict() error {
	if p.GoLlama.Options.Interactive && p.inputCallback == nil {
		return errors.New("no input callback provided in interactive mode")
	}
	f, err := os.OpenFile("go-llama.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)
	p.cancel = make(chan os.Signal, 1)
	signal.Notify(p.cancel, os.Interrupt, syscall.SIGINT)
	go p.Cancel()
	if p.GoLlama.Options.InteractiveFirst {
		p.Options.Prompt = p.InputCallback()
	}
	past := 0
	totalConsumed := 0
	batchSize := p.Options.BatchSize
	remain := p.Options.NumPredict
	isAntiPrompt := false
	inputEcho := true
	display := p.Options.DisplayPrompt
	var emb []int
	var embIn = p.Tokenizer.Tokenize(p.GoLlama.Options.ApplyTemplate(p.Options.Prompt), true, true)
	if len(embIn) == 0 {
		embIn = append(embIn, p.Tokenizer.TokenBos())
	}
	addBos := p.Tokenizer.AddBos()
	if p.Options.NumKeep < 0 || p.Options.NumKeep > len(embIn) {
		p.Options.NumKeep = len(embIn)
	} else if addBos {
		p.Options.NumKeep += 1
	}
	log.Printf("Start prediction loop, contextSize:%d, eogTokens: %s\n", p.Options.ContextSize, p.StringifyTokens(p.eogTokens))
	for !p.IsExiting && ((remain != 0 && !isAntiPrompt) || !p.Options.Interactive) {
		if len(emb) > 0 {
			past, err = p.DecodeInBatches(emb, past, batchSize)
			if err != nil {
				log.Println(err)
				return err
			}
		}
		emb = []int{}
		if len(embIn) <= totalConsumed && !p.IsInteracting {
			id := p.SamplingSample()
			//log.Printf("output, tokens: %s\n", p.StringifyTokens([]int{id}))
			if p.IsEog(id) {
				p.IsInteracting = p.Options.Interactive
				display = p.Options.DisplayPrompt
				isAntiPrompt = true
			}
			p.SamplingAccept(id, true)
			emb = append(emb, id)
			inputEcho = true
			remain -= 1
			// log.Printf("n_remain: %d, last: %s\n", remain, p.StringifyTokens(p.SamplingPrev()))
		} else {
			for len(embIn) > totalConsumed {
				emb = append(emb, embIn[totalConsumed])
				p.SamplingAccept(embIn[totalConsumed], false)
				totalConsumed += 1
				if len(emb) > batchSize {
					break
				}
			}
			log.Printf("input, len(embIn): %d, totalConsumed: %d\n, tokens: %s\n", len(embIn), totalConsumed, p.StringifyTokens(embIn))
		}
		// display text if necessary
		if inputEcho && display && len(emb) > 0 {
			p.OutputCallback(p.Tokenizer.ToString(emb))
		}
		// start displaying output if input was consumed
		if inputEcho && len(embIn) == totalConsumed {
			display = true
		}
		// if not currently processing queued inputs
		if len(embIn) <= totalConsumed {
			if past > 0 && p.IsInteracting {
				log.Println("Waiting for input...")
				p.SamplingReset()
				if p.Options.InputPrefixBos {
					embIn = append(embIn, p.Tokenizer.TokenBos())
				}
				prompt := p.InputCallback()
				display = true
				if prompt != "" {
					lineIn := p.Tokenizer.Tokenize(p.Options.ApplyTemplate(prompt), false, true)
					embIn = append(embIn, lineIn...)
					if p.Options.DisplayPrompt {
						p.OutputCallback(prompt)
					}
					remain -= len(lineIn)
					log.Println("n_remain", remain)
				} else {
					log.Println("empty line, passing control back")
				}
				p.IsInteracting = false
			}

			// clear anti-prompt flag
			if past > 0 {
				isAntiPrompt = false
			}
			// don't display the input as it was already processed
			inputEcho = false
		}
		// end of generation
		if len(emb) > 0 && p.IsEog(emb[len(emb)-1]) && !p.Options.Interactive {
			log.Println("[end of text]")
			break
		}
		// in interactive mode, respect the maximum number of tokens and drop back to user input when reached.
		// we skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
		if p.Options.Interactive && remain <= 0 && p.Options.NumPredict >= 0 {
			remain = p.Options.NumPredict
			p.IsInteracting = true
		}
	}
	p.SamplingReset()
	p.EndOutputCallback()
	return nil
}

func (p *Predictor) DecodeInBatches(emb []int, past int, batchSize int) (int, error) {
	// log.Println("predict in batches, tokens: ", p.StringifyTokens(emb))
	embLen := len(emb)
	maxEmbSize := p.Options.ContextSize - 4
	if embLen > maxEmbSize {
		log.Printf("input too long, skipped to %d tokens\n", embLen-maxEmbSize)
		emb = emb[embLen-maxEmbSize:]
		embLen = maxEmbSize
	}
	// Ensure the output doesn't exceed the context size by truncating embd_guidance if necessary
	if p.Options.GroupAttnFactor == 1 {
		// infinite text generation via context shifting
		// if we run out of context:
		// - take the n_keep first tokens from the original prompt (via p_state->n_past)
		// - take half of the last (p_state.n_ctx - n_keep) tokens and recompute the logits in batches
		if past+embLen > p.Options.ContextSize {
			if p.Options.NumPredict == -2 {
				log.Printf("context full and n_predict == -%d => stopping\n", p.Options.NumPredict)
				return past, errors.New("context full")
			}
			numLeft := past - p.Options.NumKeep
			numDiscard := numLeft / 2

			log.Printf("context full, swapping: n_past = %d, n_left = %d, n_ctx = %d, n_keep = %d, n_discard = %d\n",
				past, numLeft, p.Options.ContextSize, p.Options.NumKeep, numDiscard)

			C.go_llama_kv_cache_seq_rm((*C.go_llama_state)(p.State), C.int(0), C.int(p.Options.NumKeep), C.int(p.Options.NumKeep+numDiscard))
			C.go_llama_kv_cache_seq_add((*C.go_llama_state)(p.State), C.int(0), C.int(p.Options.NumKeep+numDiscard), C.int(past), C.int(-numDiscard))

			past -= numDiscard
			log.Printf("after swap: n_past = %d, embd: %s\n", past, p.StringifyTokens(emb))
		}
	} else {
		// context extension via Self-Extend
		// group-attention state
		// number of grouped KV tokens so far (used only if p.Options.GroupAttnFactor > 1)
		grpAttnId := 0
		grpAttnNum := p.Options.GroupAttnFactor
		grpAttnWeight := p.Options.GroupAttnWeight
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

			C.go_llama_kv_cache_seq_add((*C.go_llama_state)(p.State), C.int(0), C.int(grpAttnId), C.int(past), C.int(ib*bd))
			C.go_llama_kv_cache_seq_div((*C.go_llama_state)(p.State), C.int(0), C.int(grpAttnId+ib*bd), C.int(grpAttnId+ib*bd+grpAttnWeight), C.int(grpAttnNum))
			C.go_llama_kv_cache_seq_add((*C.go_llama_state)(p.State), C.int(0), C.int(grpAttnId+ib*bd+grpAttnWeight), C.int(past+ib*bd), C.int(dd))

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
		err := p.DecodeBatch(emb, i, eval, past)
		if err != nil {
			return past, err
		}
		past += eval
	}
	return past, nil
}

func (p *Predictor) DecodeBatch(emb []int, pos int, numTokens int, startPos int) error {
	embTokensList := (*C.tokens_list)(p.Tokenizer.ToTokensList(emb))
	if ok := int(C.go_llama_decode_batch(
		(*C.struct_go_llama_state)(p.State),
		*embTokensList,
		C.int(pos),
		C.int(numTokens),
		C.int(startPos))); ok == 1 {
		return errors.New("failed to decode batch")
	}
	return nil
}

func (p *Predictor) SamplingInit() {
	C.go_llama_sampling_init((*C.struct_go_llama_predict_state)(p.PredictState))
}
func (p *Predictor) SamplingSample() int {
	return int(C.go_llama_sampling_sample(
		(*C.go_llama_state)(p.State),
		(*C.go_llama_predict_state)(p.PredictState)))
}
func (p *Predictor) SamplingAccept(id int, applyGrammar bool) {
	C.go_llama_sampling_accept(
		(*C.go_llama_state)(p.State),
		(*C.go_llama_predict_state)(p.PredictState),
		C.int(id),
		C.bool(applyGrammar))
}
func (p *Predictor) SamplingPrev() []int {
	return p.Tokenizer.ToSlice(unsafe.Pointer(C.go_llama_sampling_prev(
		(*C.go_llama_predict_state)(p.PredictState))))
}
func (p *Predictor) SamplingReset() {
	log.Println("resetting sampling")
	C.go_llama_sampling_reset((*C.go_llama_predict_state)(p.PredictState))
}

func (p *Predictor) StringifyTokens(tokens []int) string {
	var result string
	for k, v := range tokens {
		if v == 0 {
			continue
		}
		result += fmt.Sprintf("'%s':%d", strings.Replace(p.Tokenizer.ToString([]int{v}), "\n", "\\n", -1), v)
		if k < len(tokens)-1 {
			result += ", "
		}
	}
	return fmt.Sprintf("[%s]", result)
}

func (p *Predictor) OutputCallback(token string) {
	if p.outputCallback != nil {
		p.outputCallback(token)
	}
}

func (p *Predictor) IsEog(id int) bool {
	if C.go_llama_token_is_eog((*C.go_llama_state)(p.State), C.int(id)) {
		return true
	}
	if len(p.eogTokens) > 0 {
		for _, v := range p.eogTokens {
			if v == id {
				return true
			}
		}
	}
	return false
}

func (p *Predictor) InputCallback() string {
	if p.inputCallback != nil {
		// TODO Add Ctrl-C support
		p.reading = make(chan string, 1)
		go func() {
			p.reading <- p.inputCallback()
		}()
		result, ok := <-p.reading
		if ok {
			close(p.reading)
		}
		return result
	}
	return ""
}

func (p *Predictor) EndOutputCallback() {
	if p.endOutputCallback != nil {
		p.endOutputCallback()
	}
}

func (p *Predictor) Cancel() {
	for sig := range p.cancel {
		log.Printf("Received signal %s\n", sig)
		select {
		case <-p.reading:
		default:
			close(p.reading)
			p.IsExiting = true
		}
		if p.Options.Interactive && !p.IsInteracting {
			log.Println("Give control back to user...")
			p.IsInteracting = true
		} else if !p.Options.Interactive {
			log.Println("Exit if non-interactive...")
			p.IsExiting = true
		}
	}
}
func (p *Predictor) Free() {
	C.go_llama_predict_free((*C.struct_go_llama_predict_state)(p.PredictState))
	delete(predictors, uintptr(p.PredictState))
}
