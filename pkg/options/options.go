package options

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdbool.h>
#include <stdlib.h>
#include "options.h"
#include "../util/util.h"
*/
import "C"
import (
	"errors"
	"fmt"
	"github.com/stanchino/go-llama-cpp/pkg/util"
	"gopkg.in/yaml.v3"
	"log"
	"os"
	"path/filepath"
	"unsafe"
)

type Options struct {
	Model            string   `json:"model,omitempty" yaml:"model"`
	Prompt           string   `json:"prompt,omitempty" yaml:"prompt"`
	UseMMap          bool     `json:"useMMap,omitempty" yaml:"useMMap"`
	ContextSize      int      `json:"contextSize,omitempty" yaml:"contextSize"`
	RopeFreqBase     float64  `json:"ropeFreqBase,omitempty" yaml:"ropeFreqBase"`
	RopeFreqScale    float64  `json:"ropeFreqScale,omitempty" yaml:"ropeFreqScale"`
	Interactive      bool     `json:"interactive,omitempty" yaml:"interactive"`
	InteractiveFirst bool     `json:"interactiveFirst,omitempty" yaml:"interactiveFirst"`
	DisplayPrompt    bool     `json:"displayPrompt,omitempty" yaml:"displayPrompt"`
	InputPrefixBos   bool     `json:"inputPrefixBos,omitempty" yaml:"inputPrefixBos"`
	InputPrefix      string   `json:"inputPrefix,omitempty" yaml:"inputPrefix"`
	InputSuffix      string   `json:"inputSuffix,omitempty" yaml:"inputSuffix"`
	Template         string   `json:"template,omitempty" yaml:"template"`
	AntiPrompts      []string `json:"antiPrompts,omitempty" yaml:"antiPrompts"`
	EndOfTextPrompts []string `json:"endOfTextPrompts,omitempty" yaml:"endOfTextPrompts"`
	prompt           string
}

func LoadConfig(configFile string) (*Options, error) {
	if configFile == "" {
		return nil, errors.New("you must provide a path to the config file")
	}
	filename, _ := filepath.Abs(configFile)
	yamlFile, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("file %s not found: %s", configFile, err)
	}
	var opt Options
	err = yaml.Unmarshal(yamlFile, &opt)
	if err != nil {
		return nil, fmt.Errorf("unable to load configuration from file %s: %s", yamlFile, err)
	}
	if opt.Model, err = filepath.Abs(filepath.Clean(opt.Model)); err != nil {
		return nil, err
	}
	opt.Prepare()
	return &opt, nil
}

func (o *Options) Prepare() {
	if o.ContextSize < 0 {
		log.Println("warning: minimum context size is 8, using minimum size.")
		o.ContextSize = 8
	}
	if o.RopeFreqBase != 0 {
		log.Printf("warning: changing RoPE frequency base to %g.\n", o.RopeFreqBase)
	}
	if o.RopeFreqScale != 0 {
		log.Printf("warning: scaling RoPE frequency by  %g.\n", o.RopeFreqScale)
	}
	if o.InteractiveFirst {
		o.Interactive = true
	}
	if o.Interactive {
		if len(o.AntiPrompts) > 0 {
			for _, p := range o.AntiPrompts {
				log.Println("Reverse prompt: " + p)
			}
		}
		if o.InputPrefixBos {
			log.Println("Input prefix with BOS")
		}
		if o.InputPrefix != "" {
			log.Println("Input prefix: " + o.InputPrefix)
		}
		if o.InputSuffix != "" {
			log.Println("Input suffix: " + o.InputSuffix)
		}
	}
}

func (o *Options) ToInitParams() unsafe.Pointer {
	o.Prepare()
	return unsafe.Pointer(&C.go_llama_params{
		model:             C.CString(o.Model),
		use_mmap:          C.bool(o.UseMMap),
		interactive:       C.bool(o.Interactive),
		interactive_first: C.bool(o.InteractiveFirst),
		display_prompt:    C.bool(o.DisplayPrompt),
		n_ctx:             C.int(o.ContextSize),
		rope_freq_base:    C.double(o.RopeFreqBase),
		rope_freq_scale:   C.double(o.RopeFreqScale),
		input_prefix_bos:  C.bool(o.InputPrefixBos),
		input_prefix:      C.CString(o.InputPrefix),
		input_suffix:      C.CString(o.InputSuffix),
		prompt:            C.CString(o.prompt),
		anti_prompts:      (*C.charArray)(util.NewCharArray(o.AntiPrompts).Pointer),
		eot_prompts:       (*C.charArray)(util.NewCharArray(o.EndOfTextPrompts).Pointer),
	})
}

func (o *Options) ApplyTemplate(str string) string {
	if o.Template == "" || str == "" {
		return str
	}
	return fmt.Sprintf(o.Template, str)
}
