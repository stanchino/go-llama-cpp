.PHONY: binding.o libbinding.a build clean sources

default: build

# llama.cpp header files
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

LLAMA_METAL_EMBED=0
ifeq ($(UNAME_S),Darwin)
	ifndef LLAMA_NO_METAL
		EXTRA_LLAMA_TARGETS += ggml-metal-embed.o
		LAMA_METAL_EMBED=1
	endif
endif

libllama.a:
	LLAMA_METAL_EMBED_LIBRARY=${LAMA_METAL_EMBED} $(MAKE) -C llama.cpp libllama.a $(EXTRA_LLAMA_TARGETS)
	cp llama.cpp/libllama.a lib/libllama_${UNAME_M}.a

build: libllama.a $(foreach h,$(HEADERS),includes/$(h))
	go build ./...

includes/%: %
	cp $^ includes

rebuild: clean build
clean:
	rm -rf lib/*.o
	rm -rf lib/*.a
	rm -rf includes/*.h
	$(MAKE) -C llama.cpp clean

vpath %.h llama.cpp llama.cpp/common