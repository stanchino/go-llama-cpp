package options

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++11
#include <stdbool.h>
#include "../llama/llama.h"
*/
import "C"
import "C"
import (
	"errors"
	"fmt"
	"github.com/creasty/defaults"
	"gopkg.in/yaml.v3"
	"log"
	"os"
	"path/filepath"
	"strings"
	"unsafe"
)

type Options struct {
	Model            string   `json:"model,omitempty" yaml:"model"`
	Prompt           string   `json:"prompt,omitempty" yaml:"prompt"`
	UseMMap          bool     `json:"useMMap,omitempty" yaml:"useMMap"`
	ContextSize      int      `default:"512" json:"contextSize,omitempty" yaml:"contextSize"`
	NumPredict       int      `default:"-1" json:"numPredict,omitempty" yaml:"numPredict"`
	BatchSize        int      `default:"2048" json:"batchSize,omitempty" yaml:"batchSize"`
	GroupAttnFactor  int      `default:"1" json:"groupAttnFactor,omitempty" yaml:"groupAttnFactor"`
	GroupAttnWeight  int      `default:"512" json:"groupAttnWeight,omitempty" yaml:"groupAttnWeight"`
	NumKeep          int      `default:"0" json:"numKeep,omitempty" yaml:"numKeep"`
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
	Warmup           bool     `default:"true" json:"warmup,omitempty" yaml:"warmup"`
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
	err = defaults.Set(&opt)
	if err != nil {
		return nil, fmt.Errorf("unable to set defaults: %s", err)
	}
	err = yaml.Unmarshal(yamlFile, &opt)
	if err != nil {
		return nil, fmt.Errorf("unable to load configuration from file %s: %s", yamlFile, err)
	}
	if ok := opt.Prepare(); ok != nil {
		return nil, ok
	}
	return &opt, nil
}

func (o *Options) Prepare() error {
	if strings.HasPrefix(o.Model, "~") {
		if dirname, err := os.UserHomeDir(); err != nil {
			return err
		} else {
			o.Model = filepath.Join(dirname, o.Model[1:])
		}

	}
	if model, err := filepath.Abs(filepath.Clean(o.Model)); err != nil {
		return err
	} else {
		o.Model = model
	}
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
	return nil
}

func (o *Options) ApplyTemplate(str string) string {
	if o.Template == "" || str == "" {
		return str
	}
	return fmt.Sprintf(o.Template, str)
}

func (o *Options) Params() (unsafe.Pointer, error) {
	if ok := o.Prepare(); ok != nil {
		return nil, ok
	}
	return unsafe.Pointer(&C.go_llama_params{
		model:             C.CString(o.Model),
		use_mmap:          C.bool(o.UseMMap),
		interactive:       C.bool(o.Interactive),
		interactive_first: C.bool(o.InteractiveFirst),
		display_prompt:    C.bool(o.DisplayPrompt),
		n_ctx:             C.int(o.ContextSize),
		rope_freq_base:    C.float(o.RopeFreqBase),
		rope_freq_scale:   C.float(o.RopeFreqScale),
		input_prefix_bos:  C.bool(o.InputPrefixBos),
		input_prefix:      C.CString(o.InputPrefix),
		input_suffix:      C.CString(o.InputSuffix),
		prompt:            C.CString(o.Prompt),
		n_predict:         C.int(o.NumPredict),
		n_batch:           C.int(o.BatchSize),
		n_keep:            C.int(o.NumKeep),
		grp_attn_n:        C.int(o.GroupAttnFactor),
		grp_attn_w:        C.int(o.GroupAttnWeight),
	}), nil
}
