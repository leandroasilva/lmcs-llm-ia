package compute

import (
	"fmt"
	"runtime"
)

// DeviceType represents the type of compute device
type DeviceType int

const (
	DeviceCPU DeviceType = iota
	DeviceGPU_OpenCL
)

func (d DeviceType) String() string {
	switch d {
	case DeviceCPU:
		return "CPU"
	case DeviceGPU_OpenCL:
		return "GPU (OpenCL)"
	default:
		return "Unknown"
	}
}

// DeviceInfo holds information about a compute device
type DeviceInfo struct {
	Type         DeviceType
	Name         string
	Vendor       string
	Version      string
	MemoryMB     int
	MaxWorkGroup int
}

// Buffer represents a device-side memory buffer
type Buffer struct {
	Size      int
	FloatData []float32
}

// NewBuffer creates a new buffer from float64 slice (converts to float32 for GPU)
func NewBuffer(data []float64) *Buffer {
	float32Data := make([]float32, len(data))
	for i, v := range data {
		float32Data[i] = float32(v)
	}
	return &Buffer{
		Size:      len(data),
		FloatData: float32Data,
	}
}

// ToFloat64 converts buffer back to float64 slice
func (b *Buffer) ToFloat64() []float64 {
	data := make([]float64, b.Size)
	for i, v := range b.FloatData {
		data[i] = float64(v)
	}
	return data
}

// Backend defines the interface for compute operations (CPU or GPU)
type Backend interface {
	// Info returns device information
	Info() DeviceInfo

	// Initialize prepares the backend for computation
	Initialize() error

	// Release frees all resources
	Release() error

	// Matrix Operations
	MatMul(a, b *Buffer, m, n, k int) (*Buffer, error)       // C = A @ B, A:[m,k], B:[k,n], C:[m,n]
	MatMulAdd(a, b, c *Buffer, m, n, k int) (*Buffer, error) // C = A @ B + C
	MatVecMul(a, b *Buffer, m, k int) (*Buffer, error)       // c = A @ b, A:[m,k], b:[k]

	// Element-wise Operations
	Add(a, b *Buffer) (*Buffer, error)                      // element-wise addition
	Scale(a *Buffer, scalar float32) (*Buffer, error)       // element-wise multiplication by scalar
	AddScaled(a, b *Buffer, scale float32) (*Buffer, error) // a + b * scale

	// Activation Functions
	Softmax(x *Buffer, axis int) (*Buffer, error) // softmax along axis
	ReLU(x *Buffer) (*Buffer, error)              // ReLU activation

	// Normalization
	LayerNorm(x *Buffer, gamma, beta *Buffer, rows, cols int, eps float32) (*Buffer, error)

	// Reduction
	Sum(x *Buffer) (float32, error)  // sum all elements
	Mean(x *Buffer) (float32, error) // mean of all elements

	// Embedding
	EmbeddingLookup(weights *Buffer, indices []int, vocabSize, embedDim int) (*Buffer, error)

	// Utility
	CopyToDevice(data []float64) (*Buffer, error)
	CopyFromDevice(buf *Buffer) ([]float64, error)
	Sync() error
}

// NewBackend creates the best available backend
// UseGPU: if true, tries GPU first with CPU fallback
func NewBackend(useGPU bool) (Backend, error) {
	if useGPU {
		// Try OpenCL first
		if runtime.GOOS == "darwin" || runtime.GOOS == "linux" {
			gpu, err := NewOpenCLBackend()
			if err == nil {
				if initErr := gpu.Initialize(); initErr == nil {
					return gpu, nil
				}
				gpu.Release()
			}
		}
	}

	// Fallback to CPU
	cpu := NewCPUBackend()
	if err := cpu.Initialize(); err != nil {
		return nil, fmt.Errorf("failed to initialize CPU backend: %w", err)
	}
	return cpu, nil
}

// AutoSelectBackend automatically selects the best backend
func AutoSelectBackend() (Backend, error) {
	// Check if we have a usable GPU
	if runtime.GOOS == "darwin" {
		// On macOS, check for OpenCL support
		gpu, err := NewOpenCLBackend()
		if err == nil {
			info := gpu.Info()
			if info.Type == DeviceGPU_OpenCL {
				if initErr := gpu.Initialize(); initErr == nil {
					return gpu, nil
				}
			}
			gpu.Release()
		}
	}

	// Default to CPU
	return NewBackend(false)
}

// BackendConfig holds configuration for backend selection
type BackendConfig struct {
	UseGPU       bool   // Force GPU usage
	DeviceName   string // Specific device name to use (optional)
	PlatformName string // Specific OpenCL platform (optional)
}

// NewBackendWithConfig creates a backend with specific configuration
func NewBackendWithConfig(cfg *BackendConfig) (Backend, error) {
	if cfg == nil {
		cfg = &BackendConfig{UseGPU: false}
	}

	if cfg.UseGPU {
		gpu, err := NewOpenCLBackend()
		if err != nil {
			return nil, fmt.Errorf("GPU requested but OpenCL not available: %w", err)
		}
		if err := gpu.Initialize(); err != nil {
			gpu.Release()
			return nil, fmt.Errorf("GPU initialization failed: %w", err)
		}
		return gpu, nil
	}

	return NewBackend(false)
}
