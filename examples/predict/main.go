package main

import (
    "flag"
    "fmt"
    "github.com/stanchino/go-llama-cpp/examples"
    "github.com/stanchino/go-llama-cpp/pkg/llama"
    "github.com/stanchino/go-llama-cpp/pkg/options"
    "github.com/stanchino/go-llama-cpp/pkg/predictor"
    "log"
)

func main() {
    config := flag.String("c", "", "Provide a path to the config file")
    prompt := flag.String("p", "", "Provide a prompt")
    flag.Parse()
    if *prompt == "" {
        log.Fatalln("no prompt provided")
    }
    opt, cErr := options.LoadConfig(*config)
    if cErr != nil {
        log.Fatalln(cErr)
    }
    opt.Prompt = *prompt
    opt.Interactive = false
    opt.InteractiveFirst = false

    // Initialize Llama
    l, lErr := llama.NewGoLlama(opt)
    if lErr != nil {
        log.Fatal(lErr)
    }
    defer l.Free()

    p := predictor.NewPredictor(l)
    defer p.Free()

    p.SetOutputCallback(func(token string) {
        fmt.Printf("%s", token)
    })
    p.SetEndOutputCallback(func() {
        fmt.Printf("%s", examples.AnsiColorReset)
    })
    fmt.Printf("%s", examples.AnsiColorYellow)
    if ok := p.Predict(); ok != nil {
        log.Fatal(ok)
    }
}

/*
func (p *Predictor) Predict() error {
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
	log.Println("Start prediction loop")
	fmt.Printf("%s", examples.AnsiColorYellow)
	var err error
	for !p.IsExiting && (remain != 0 && !isAntiPrompt) {
		if len(emb) > 0 {
			past, err = p.DecodeInBatches(emb, past, batchSize)
			if err != nil {
				log.Println(err)
				return err
			}
		}
		emb = []int{}
		if len(embIn) <= totalConsumed {
			log.Println("Processing output...")
			id := p.SamplingSample()
			if p.IsEog(id) {
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
		// if all the input tokens are consumed show the output
		if inputEcho && len(embIn) == totalConsumed {
			display = true
		}
		// if not currently processing queued inputs
		if len(embIn) <= totalConsumed {
			if past > 0 {
				isAntiPrompt = false
			}
			inputEcho = false
		}
		// end of generation
		if len(embIn) > 0 && p.IsEog(emb[len(emb)-1]) {
			log.Println("[end of text]")
			break
		}
	}
	p.EndOutputCallback()
	fmt.Printf("%s", examples.AnsiColorReset)
	return nil
}
*/
