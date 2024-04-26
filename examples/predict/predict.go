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
		log.Fatal("no prompt provided")
	}
	opt, err := options.LoadConfig(*config)
	if err != nil {
		log.Fatal(err)
	}
	opt.Interactive = false
	opt.InteractiveFirst = false
	opt.Prompt = *prompt
	l := llama.NewGoLlama(opt)
	defer l.Free()

	p := predictor.NewPredictor(l)
	p.SetOutputCallback(func(token string) {
		fmt.Printf("%s%s%s", examples.AnsiColorYellow, token, examples.AnsiColorReset)
	})
	if ok := p.Predict(); ok != nil {
		log.Fatal(ok)
	}
}
