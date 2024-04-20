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
		ModelName: *modelName,
	})
	defer l.Free()
	p := predictor.NewPredictor(l)
	prompt := "Question: What is your name?"
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
