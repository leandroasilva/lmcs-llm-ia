//go:build !darwin && !linux
// +build !darwin,!linux

package compute

import "fmt"

// NewOpenCLBackend returns an error on platforms without OpenCL support
func NewOpenCLBackend() (Backend, error) {
	return nil, fmt.Errorf("OpenCL not supported on this platform")
}
