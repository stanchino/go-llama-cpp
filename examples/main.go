package main

import (
	"flag"
	"fmt"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
	"log"
)

func main() {
	modelName := flag.String("m", "", "Provide a path to the model file")
	flag.Parse()
	if *modelName == "" {
		log.Fatal("You must provide a path to the model file")
	}
	l := llama.NewGoLlama(options.Options{
		ModelName:   *modelName,
		AntiPrompts: []string{"<|eot_id|>", "<|user|>"},
	})
	defer l.Free()
	p := predictor.NewPredictor(l)
	prompt := "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a 1st grade school teacher.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow far are the stars?<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
	result := make(chan string)
	go p.Predict(prompt, result)
	log.Printf("Prompt is %s\n", prompt)
	for v := range result {
		if v == "[end of text]" {
			return
		}
		fmt.Print(v)
	}
}
