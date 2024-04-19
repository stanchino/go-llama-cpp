.PHONY: binding.o libbinding.a build clean sources

default: build

# keep standard at C11 and C++11
CFLAGS   = -I./llama.cpp -I./llama.cpp/common -I. -O3 -DNDEBUG -std=c11 -fPIC
CXXFLAGS = -I./llama.cpp -I./llama.cpp/common -I. -O3 -DNDEBUG -std=c++11 -fPIC
LDFLAGS  =

# warnings
CFLAGS   += -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -Wno-unused-function
CXXFLAGS += -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function

ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

LLAMA_METAL_EMBED=0
ifeq ($(UNAME_S),Darwin)
	ifndef LLAMA_NO_METAL
		EXTRA_LLAMA_TARGETS += ggml-metal-embed.o
		LAMA_METAL_EMBED=1
	endif
endif

libllama.a:
	LLAMA_METAL_EMBED_LIBRARY=${LAMA_METAL_EMBED} $(MAKE) -C llama.cpp libllama.a $(EXTRA_LLAMA_TARGETS)

build: libllama.a sources
	go build options.go go_llama.go
	go build tokenizer/*.go
	go build predictor/*.go

sources:
	mkdir -p ./llama-src/common
	cp ./llama.cpp/*.h ./llama-src/
	cp ./llama.cpp/common/*.h ./llama-src/common/

clean:
	rm -rf lib/*.o
	rm -rf lib/*.a
	rm -rf llama-src
	$(MAKE) -C llama.cpp clean
