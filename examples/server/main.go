package main

import (
	"flag"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
	"io"
	"log"
	"net/http"
)

func main() {
	config := flag.String("c", "", "Provide a path to the config file")
	flag.Parse()
	opt, cErr := options.LoadConfig(*config)
	if cErr != nil {
		log.Fatal(cErr)
	}
	opt.Interactive = false
	opt.InteractiveFirst = false
	l, lErr := llama.NewGoLlama(opt)
	if lErr != nil {
		log.Fatal(lErr)
	}
	defer l.Free()

	p := predictor.NewPredictor(l)

	http.HandleFunc("/chat", func(w http.ResponseWriter, req *http.Request) {
		defer req.Body.Close()
		input, _ := io.ReadAll(req.Body)
		out := make(chan string)
		p.InitState()
		p.SetOutputCallback(func(token string) {
			out <- token
		})
		p.SetEndOutputCallback(func() {
			close(out)
		})
		go func(prompt string) {
			if ok := p.WithPrompt(prompt).Predict(); ok != nil {
				log.Fatal(ok)
			}
		}(string(input))
		for v := range out {
			io.WriteString(w, v)
			w.(http.Flusher).Flush()
		}
		p.Free()
	})
	if err := http.ListenAndServe(":8090", nil); err != nil {
		log.Fatal(err)
	}
}
