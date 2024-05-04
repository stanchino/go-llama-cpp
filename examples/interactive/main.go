package main

import (
	"bufio"
	"flag"
	"fmt"
	"github.com/stanchino/go-llama-cpp/examples"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
	"log"
	"os"
)

func main() {
	config := flag.String("c", "", "Provide a path to the config file")
	flag.Parse()
	opt, cErr := options.LoadConfig(*config)
	if cErr != nil {
		log.Fatal(cErr)
	}
	opt.Interactive = true
	opt.InteractiveFirst = true

	// Initialize Llama
	l, lErr := llama.NewGoLlama(opt)
	if lErr != nil {
		log.Fatal(lErr)
	}
	defer l.Free()

	p := predictor.NewPredictor(l)
	defer p.Free()

	result := make(chan string)
	p.SetOutputCallback(func(token string) {
		result <- token
	})
	p.SetEndOutputCallback(func() {
		close(result)
	})
	scanner := bufio.NewScanner(os.Stdin)
	p.SetInputCallback(func() string {
		fmt.Printf("\n%s>", examples.AnsiColorGreen)
		scanner.Scan()
		fmt.Printf("%s", examples.AnsiColorReset)
		return scanner.Text()
	})
	go func() {
		if ok := p.Predict(); ok != nil {
			log.Fatal(ok)
		}
	}()
	for v := range result {
		fmt.Printf("%s%s", examples.AnsiColorYellow, v)
	}
	fmt.Print(examples.AnsiColorReset)
}

/*
func (p *InteractivePredictor) Predict() error {
	past := 0
	totalConsumed := 0
	batchSize := p.Options.BatchSize
	remain := p.Options.NumPredict
	isAntiPrompt := false
	inputEcho := true
	display := p.Options.DisplayPrompt
	var emb []int
	p.Options.Prompt = p.InputCallback()
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
	log.Println("Start prediction loop")
	var err error
	for !p.Exit && (remain != 0 && !isAntiPrompt) {
		if len(emb) > 0 {
			past, err = p.DecodeInBatches(emb, past, batchSize)
			if err != nil {
				log.Println(err)
				return err
			}
		}
		emb = []int{}
		if len(embIn) <= totalConsumed && !p.IsInteracting {
			log.Println("Processing output...")
			id := p.SamplingSample()
			if p.IsEog(id) {
				p.IsInteracting = true
				isAntiPrompt = true
				display = p.Options.DisplayPrompt
			}
			p.SamplingAccept(id, true)
			emb = append(emb, id)
			inputEcho = true
			remain -= 1
			log.Println("n_remain", remain)
		} else {
			log.Println("Processing input...")
			log.Printf("len(embIn): %v, totalConsumed: %v\n", len(embIn), totalConsumed)
			for len(embIn) > totalConsumed {
				emb = append(emb, embIn[totalConsumed])
				p.SamplingAccept(embIn[totalConsumed], false)
				totalConsumed += 1
				if len(emb) > batchSize {
					break
				}
			}
		}
		// display text
		if inputEcho && display {
			if len(emb) > 0 {
				p.OutputCallback(p.Tokenizer.ToString(emb))
			}
		}
		if inputEcho && len(embIn) == totalConsumed {
			display = true
		}
		// if not currently processing queued inputs;
		if len(embIn) <= totalConsumed {
			if past > 0 && p.IsInteracting {
				if p.Options.InputPrefixBos {
					embIn = append(embIn, p.Tokenizer.TokenBos())
				}
				prompt := p.InputCallback()
				display = true
				if prompt != "" {
					lineIn := p.Tokenizer.Tokenize(p.Options.ApplyTemplate(prompt), false, false)
					embIn = append(embIn, lineIn...)
					if p.Options.DisplayPrompt {
						p.OutputCallback(prompt)
					}
					remain -= len(lineIn)
					log.Println("n_remain", remain)
				} else {
					log.Println("empty line, passing control back")
				}
				p.SamplingReset()
				p.IsInteracting = false
			}
			if past > 0 {
				isAntiPrompt = false
			}
			inputEcho = false
		}
		// In interactive mode, respect the maximum number of tokens and drop back to user input when reached.
		// We skip this logic when n_predict == -1 (infinite) or -2 (stop at context size).
		if remain <= 0 && p.Options.NumPredict >= 0 {
			remain = p.Options.NumPredict
			p.IsInteracting = true
		}
	}
	p.EndOutputCallback()
	return nil
}
*/
