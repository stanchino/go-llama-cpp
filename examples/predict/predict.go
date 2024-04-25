package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/stanchino/go-llama-cpp/examples"

	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
)

func main() {
	modelName := flag.String("m", "", "Provide a path to the model file")
	prompt := flag.String("p", "", "Provide a prompt")
	flag.Parse()
	if *modelName == "" {
		log.Fatal("You must provide a path to the model file")
	}
	if *prompt == "" {
		log.Fatal("You must provide a prompt")
	}
	l := llama.NewGoLlama(options.Options{
		ModelName:        *modelName,
		ContextSize:      4096,
		UseMMap:          false,
		Interactive:      false,
		InteractiveFirst: false,
		DisplayPrompt:    false,
		AntiPrompts:      []string{"<|user|>"},
		//EndOfTextPrompts: []string{"<|end|>"},
		//Template:         "<|user|>%s<|end|><|assistant|>",
		Template: "<|begin_of_text|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
		// InputSuffix:      "\n<|end|>\n<|assistant|>\n",
		// InputPrefix:      "<|user|>\n",
	})
	defer l.Free()

	p := predictor.NewPredictor(l)
	p.SetOutputCallback(func(token string) {
		fmt.Printf("%s%s%s", examples.AnsiColorYellow, token, examples.AnsiColorReset)
	})
	p.Predict(*prompt)
}
