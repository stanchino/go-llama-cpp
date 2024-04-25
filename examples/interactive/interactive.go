package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/stanchino/go-llama-cpp/examples"

	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
)

func main() {
	modelName := flag.String("m", "", "Provide a path to the model file")
	flag.Parse()
	if *modelName == "" {
		log.Fatal("You must provide a path to the model file")
	}
	l := llama.NewGoLlama(options.Options{
		ModelName:        *modelName,
		ContextSize:      1024,
		UseMMap:          false,
		Interactive:      true,
		InteractiveFirst: true,
		DisplayPrompt:    false,
		AntiPrompts:      []string{"<|user|>"},
		EndOfTextPrompts: []string{"<|end|>"},
		Template:         "<|user|>\n%s\n<|end|>\n<|assistant|>\n",
		//Template:    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
	})
	defer l.Free()

	p := predictor.NewPredictor(l)
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
		return scanner.Text()
	})
	go p.Predict("Hi there")
	for v := range result {
		fmt.Printf("%s%s", examples.AnsiColorYellow, v)
	}
	fmt.Print(examples.AnsiColorReset)
}
