FROM --platform=linux/aarch64 golang:alpine
RUN apk update && apk upgrade && apk add --no-cache git make gcc g++ && apk --no-cache add ca-certificates
WORKDIR /app
COPY examples examples
COPY config config
COPY includes includes
COPY lib lib
COPY pkg pkg
COPY go.mod go.mod
COPY go.sum go.sum
COPY go.work go.work
COPY go.work.sum go.work.sum
# COPY llama.cpp llama.cpp
# COPY Makefile Makefile
# RUN make clean build
ENTRYPOINT ["go",  "run", "./examples/interactive/main.go", "-c", "./config/phi-3-mini-4K-instruct.yaml"]
# ENTRYPOINT ["/bin/sh"]