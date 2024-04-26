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
	config := flag.String("c", "", "Provide a path to the config file")
	flag.Parse()
	opt, err := options.LoadConfig(*config)
	if err != nil {
		log.Fatal(err)
	}
	opt.InteractiveFirst = true
	l := llama.NewGoLlama(opt)
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
