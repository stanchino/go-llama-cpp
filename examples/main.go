package main

import (
    "fmt"
    "github.com/stanchino/go-llama-cpp"
    "github.com/stanchino/go-llama-cpp/predictor"
    "log"
)

func main() {
    //prompt := "Question: What is your name?"
    //params.ModelName = "/Users/stanchino/Development/models/TheBloke/TheBloke_zephyr-7B-beta-GGUF/zephyr-7b-beta.Q5_K_M.gguf"
    //prompt := "<|system|>\nYou are a Head of Engineering in a Unicorn EdTEch company</s>\n<|user|>\nSergio was a backend developer writing code in Go and JavaScript that worked in one of your teams for almost 3 years and was laid off. He helped in developing some of the key features of the company main product. Write a 200 word LinkedIn recommendation for Samantha.</s>\n"
    //params.ModelName = "/Users/stanchino/Development/models/mlabonne/NeuralBeagle14-7B-GGUF/neuralbeagle14-7b.Q4_K_M.gguf"
    //prompt := "Ina is People & Operations partner in the company for almost 3 years. She is very passionate about all the initiatives she was involved in and plays a critical role in the human resources and employee happiness programs. Write me a short LinkedIn recommendation for Ina."
    l := llama.NewGoLlama(llama.Options{
        ModelName: "/Users/stanchino/Development/models/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_S.gguf",
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

    /*
       t := tokenizer.NewTokenizer(l)
       tokens := t.Tokenize("Hello World!")
       str := t.ToString(tokens)
       fmt.Println(str)
    */
    /*
       result := make(chan string)
       go l.Predict(prompt, result)
       log.Printf("Prompt is %s\n", prompt)
       for v := range result {
           if v == "[end of text]" {
               return
           }
           fmt.Print(v)
       }*/

    /*
    	adapters := []*C.char{C.CString("/Users/stanchino/Downloads/adapter_model.bin")}
    	if err := C.go_llama_set_adapters((**C.char)(unsafe.Pointer(&adapters[0])), C.int(len(adapters))); err != 0 {
    		log.Fatalln("Error setting LoRa adapters")
    	}
    */
    // C.go_llama_free()
}
