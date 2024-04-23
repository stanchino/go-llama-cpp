package main

import "C"
import (
	"bufio"
	"flag"
	"fmt"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
	"github.com/stanchino/go-llama-cpp/pkg/tokenizer"
	"log"
	"os"
)

func main() {
	modelName := flag.String("m", "", "Provide a path to the model file")
	interactive := flag.Bool("i", false, "Run in interactive mode")
	flag.Parse()
	if *modelName == "" {
		log.Fatal("You must provide a path to the model file")
	}
	l := llama.NewGoLlama(options.Options{
		ModelName:        *modelName,
		UseMMap:          false,
		Interactive:      *interactive,
		InteractiveFirst: *interactive,
		DisplayPrompt:    false,
		InputPrefix:      "<|user|>\n",
		InputSuffix:      "\n<|end|>\n<|assistant|>\n",
		AntiPrompts:      []string{"<|end|>"},
	})
	defer l.Free()

	//prompt := "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a 1st grade school teacher.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHow far are the stars?<|eot_id|>\n\n<|start_header_id|>assistant<|end_header_id|>\n\n"
	prompt := "<|user|>\nWhat's your name?<|end|>\n<|assistant|>"

	t := tokenizer.NewTokenizer(l)
	fmt.Printf("Tokenize->ToString prompt is:\n\"%s\"", t.ToString(t.Tokenize(prompt)))

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
		fmt.Print("\n\x1b[32m>")
		scanner.Scan()
		return scanner.Text()
	})
	go p.Predict("How old is the moon?")
	for v := range result {
		fmt.Printf("\x1b[33m%s", v)
	}
	fmt.Print("\x1b[0m")
}
