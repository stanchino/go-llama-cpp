package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/tokenizer"
)

func main() {
	modelName := flag.String("m", "", "Provide a path to the model file")
	flag.Parse()
	if *modelName == "" {
		log.Fatal("You must provide a path to the model file")
	}
	l := llama.NewGoLlama(options.Options{
		ModelName: *modelName,
	})
	defer l.Free()

	prompt := "<|user|>\nWhat's your name?<|end|>\n<|assistant|>"

	t := tokenizer.NewTokenizer(l)
	fmt.Println(t.Tokenize("<|end|>", false, true))
	fmt.Printf("Prompt is: \n\x1b[33m%s\x1b[0m", t.ToString(t.Tokenize(prompt)))
}
