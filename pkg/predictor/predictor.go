package predictor

import (
	"errors"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/stanchino/go-llama-cpp/pkg/llama"
)

type Predictor struct {
	*llama.GoLlama
	IsInteracting     bool
	Exit              bool
	eogTokens         []int
	outputCallback    func(string)
	inputCallback     func() string
	endOutputCallback func()
	reading           chan string
}

var (
	predictors = map[uintptr]*Predictor{}
)

func NewPredictor(l *llama.GoLlama) *Predictor {
	p := &Predictor{
		GoLlama:       l,
		IsInteracting: false,
		Exit:          false,
	}
	p.SetEndOfGenerationTokens()
	p.Sampling.Init()
	predictors[uintptr(p.State)] = p
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

func (p *Predictor) SetOutputCallback(cb func(token string)) {
	p.outputCallback = cb
}

func (p *Predictor) SetEndOutputCallback(cb func()) {
	p.endOutputCallback = cb
}

func (p *Predictor) SetInputCallback(cb func() string) {
	if p.Options.InteractiveFirst {
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
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGINT)
	go p.SigHandler(c)
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
	for !p.Exit && ((remain != 0 && !isAntiPrompt) || !p.Options.Interactive) {
		if len(emb) > 0 {
			past, err = p.DecodeInBatches(emb, past, batchSize)
			if err != nil {
				log.Println(err)
				return err
			}
		}
		emb = []int{}
		if len(embIn) <= totalConsumed && !p.IsInteracting {
			id := p.Sampling.Sample()
			//log.Printf("output, tokens: %s\n", p.StringifyTokens([]int{id}))
			if p.IsEog(id) {
				p.IsInteracting = p.Options.Interactive
				display = p.Options.DisplayPrompt
				isAntiPrompt = true
			}
			p.Sampling.Accept(id, true)
			emb = append(emb, id)
			inputEcho = true
			remain -= 1
			// log.Printf("n_remain: %d, last: %s\n", remain, p.StringifyTokens(l.ToSlice(p.SamplingPrev())))
		} else {
			for len(embIn) > totalConsumed {
				emb = append(emb, embIn[totalConsumed])
				p.Sampling.Accept(embIn[totalConsumed], false)
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
				p.Sampling.Reset()
				p.IsInteracting = false
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
	p.Sampling.Reset()
	p.EndOutputCallback()
	return nil
}

func (p *Predictor) OutputCallback(token string) {
	if p.outputCallback != nil {
		p.outputCallback(token)
	}
}

func (p *Predictor) IsEog(id int) bool {
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

func (p *Predictor) SigHandler(c chan os.Signal) {
	for sig := range c {
		log.Printf("Received signal %s\n", sig)
		if p.reading != nil {
			select {
			case <-p.reading:
			default:
				close(p.reading)
				p.Exit = true
			}
		}
		if p.Options.Interactive && !p.IsInteracting {
			log.Println("Give control back to user...")
			p.IsInteracting = true
		} else if p.Options.Interactive {
			log.Println("Stop interaction...")
			p.IsInteracting = false
		} else if !p.Options.Interactive {
			log.Println("Exit if non-interactive...")
			p.Exit = true
		}
	}
}

func (p *Predictor) Free() {
	delete(predictors, uintptr(p.State))
}
