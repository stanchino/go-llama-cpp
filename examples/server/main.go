package main

import (
	"flag"
	"github.com/stanchino/go-llama-cpp/pkg/llama"
	"github.com/stanchino/go-llama-cpp/pkg/options"
	"github.com/stanchino/go-llama-cpp/pkg/predictor"
	"io"
	"log"
	"net/http"
	"os"
)

func main() {
	f, err := os.OpenFile("go-llama.log", os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)
	config := flag.String("c", "", "Provide a path to the config file")
	flag.Parse()
	opt, cErr := options.LoadConfig(*config)
	if cErr != nil {
		log.Fatal(cErr)
	}
	opt.Interactive = true
	opt.InteractiveFirst = true
	l, lErr := llama.NewGoLlama(opt)
	if lErr != nil {
		log.Fatal(lErr)
	}
	defer l.Free()

	p := predictor.NewPredictor(l)

	in := make(chan string)
	p.SetInputCallback(func() string {
		return <-in
	})
	go func() {
		if ok := p.Predict(); ok != nil {
			log.Fatal(ok)
		}
	}()
	http.HandleFunc("/chat", func(w http.ResponseWriter, req *http.Request) {
		defer func() {
			if ok := req.Body.Close(); ok != nil {
				log.Fatal(ok)
			}
		}()
		input, _ := io.ReadAll(req.Body)
		out := make(chan string)
		p.SetOutputCallback(func(token string) {
			out <- token
		})
		p.SetEndOutputCallback(func() {
			close(out)
		})
		in <- string(input)
		for v := range out {
			if _, ok := io.WriteString(w, v); ok != nil {
				log.Fatal(ok)
			}
			w.(http.Flusher).Flush()
		}
	})
	if err := http.ListenAndServe(":8090", nil); err != nil {
		log.Fatal(err)
	}
}
