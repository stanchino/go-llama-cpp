.PHONY: binding.o libbinding.a build clean includes

default: build

# necessary llama.cpp header files
HEADERS = common.h llama.h ggml.h ggml-backend.h ggml-alloc.h log.h sampling.h grammar-parser.h

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

ifndef UNAME_M
UNAME_M := $(shell uname -m)
endif

ifeq ($(UNAME_M),x86_64)
  UNAME_M := "amd64"
endif

ifeq ($(UNAME_M),aarch64)
  UNAME_M := "arm64"
endif

LLAMA_METAL_EMBED=0
ifeq ($(UNAME_S),Darwin)
	UNAME_S := "osx"
	ifndef LLAMA_NO_METAL
		EXTRA_LLAMA_TARGETS += ggml-metal-embed.o
		LAMA_METAL_EMBED=1
	endif
endif

PLATFORM := `echo $(UNAME_S) | tr '[:upper:]' '[:lower:]'`
ARCH := `echo $(UNAME_M) | tr '[:upper:]' '[:lower:]'`

libllama.a:
	LLAMA_METAL_EMBED_LIBRARY=${LAMA_METAL_EMBED} $(MAKE) -C llama.cpp libllama.a $(EXTRA_LLAMA_TARGETS)
	cp llama.cpp/libllama.a lib/libllama_${PLATFORM}_${ARCH}.a

build: libllama.a includes
	go build ./...

includes: $(foreach h,$(HEADERS),includes/$(h))

includes/%: %
	cp $^ includes

rebuild: clean build
clean:
	rm -rf includes/*.h
	$(MAKE) -C llama.cpp clean

vpath %.h llama.cpp llama.cpp/common