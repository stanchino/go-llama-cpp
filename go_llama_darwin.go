//go:build darwin
// +build darwin

package llama

/*
#cgo darwin LDFLAGS: -L${SRCDIR}/lib -lllama_arm64 -framework Accelerate -framework Foundation -framework Metal -framework MetalKit -framework MetalPerformanceShaders
*/
import "C"
