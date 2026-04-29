package compute

import (
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// CPUBackend implements the Backend interface using CPU computation via gonum
type CPUBackend struct {
	info DeviceInfo
}

// NewCPUBackend creates a new CPU backend
func NewCPUBackend() *CPUBackend {
	return &CPUBackend{
		info: DeviceInfo{
			Type:         DeviceCPU,
			Name:         "CPU (gonum)",
			Vendor:       "Generic",
			Version:      "1.0",
			MemoryMB:     0, // Unlimited (uses system RAM)
			MaxWorkGroup: 1,
		},
	}
}

func (b *CPUBackend) Info() DeviceInfo {
	return b.info
}

func (b *CPUBackend) Initialize() error {
	return nil
}

func (b *CPUBackend) Release() error {
	return nil
}

func (b *CPUBackend) CopyToDevice(data []float64) (*Buffer, error) {
	return NewBuffer(data), nil
}

func (b *CPUBackend) CopyFromDevice(buf *Buffer) ([]float64, error) {
	if buf == nil {
		return nil, fmt.Errorf("nil buffer")
	}
	return buf.ToFloat64(), nil
}

func (b *CPUBackend) Sync() error {
	return nil
}

// MatMul computes C = A @ B using gonum
// A: [m, k], B: [k, n], C: [m, n]
func (b *CPUBackend) MatMul(a, bBuf *Buffer, m, n, k int) (*Buffer, error) {
	if a == nil || bBuf == nil {
		return nil, fmt.Errorf("nil input buffer")
	}
	if len(a.FloatData) != m*k || len(bBuf.FloatData) != k*n {
		return nil, fmt.Errorf("dimension mismatch: A[%d] expected m*k=%d, B[%d] expected k*n=%d",
			len(a.FloatData), m*k, len(bBuf.FloatData), k*n)
	}

	// Convert to gonum matrices
	A := mat.NewDense(m, k, a.ToFloat64())
	B := mat.NewDense(k, n, bBuf.ToFloat64())
	C := mat.NewDense(m, n, nil)

	C.Mul(A, B)

	// Convert back to Buffer
	cData := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			cData[i*n+j] = C.At(i, j)
		}
	}

	return NewBuffer(cData), nil
}

// MatMulAdd computes C = A @ B + C
func (b *CPUBackend) MatMulAdd(a, bBuf, c *Buffer, m, n, k int) (*Buffer, error) {
	result, err := b.MatMul(a, bBuf, m, n, k)
	if err != nil {
		return nil, err
	}

	if c != nil && len(c.FloatData) == m*n {
		for i := 0; i < m*n; i++ {
			result.FloatData[i] += c.FloatData[i]
		}
	}

	return result, nil
}

// MatVecMul computes c = A @ b where A: [m, k], b: [k]
func (b *CPUBackend) MatVecMul(a, bBuf *Buffer, m, k int) (*Buffer, error) {
	// Treat b as [k, 1] matrix
	return b.MatMul(a, bBuf, m, 1, k)
}

// Add performs element-wise addition
func (b *CPUBackend) Add(a, bBuf *Buffer) (*Buffer, error) {
	if a == nil || bBuf == nil {
		return nil, fmt.Errorf("nil input buffer")
	}
	if a.Size != bBuf.Size {
		return nil, fmt.Errorf("size mismatch: %d vs %d", a.Size, bBuf.Size)
	}

	result := make([]float32, a.Size)
	for i := 0; i < a.Size; i++ {
		result[i] = a.FloatData[i] + bBuf.FloatData[i]
	}

	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// Scale multiplies all elements by a scalar
func (b *CPUBackend) Scale(a *Buffer, scalar float32) (*Buffer, error) {
	if a == nil {
		return nil, fmt.Errorf("nil input buffer")
	}

	result := make([]float32, a.Size)
	for i := 0; i < a.Size; i++ {
		result[i] = a.FloatData[i] * scalar
	}

	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// AddScaled computes a + b * scale
func (b *CPUBackend) AddScaled(a, bBuf *Buffer, scale float32) (*Buffer, error) {
	if a == nil || bBuf == nil {
		return nil, fmt.Errorf("nil input buffer")
	}
	if a.Size != bBuf.Size {
		return nil, fmt.Errorf("size mismatch: %d vs %d", a.Size, bBuf.Size)
	}

	result := make([]float32, a.Size)
	for i := 0; i < a.Size; i++ {
		result[i] = a.FloatData[i] + bBuf.FloatData[i]*scale
	}

	return &Buffer{Size: a.Size, FloatData: result}, nil
}

// Softmax computes softmax along the specified axis
func (b *CPUBackend) Softmax(x *Buffer, axis int) (*Buffer, error) {
	if x == nil {
		return nil, fmt.Errorf("nil input buffer")
	}

	// For simplicity, assume axis=1 (row-wise softmax)
	// x is treated as a 1D vector here; caller should reshape if needed
	result := make([]float32, x.Size)

	// Find max for numerical stability
	maxVal := x.FloatData[0]
	for i := 1; i < x.Size; i++ {
		if x.FloatData[i] > maxVal {
			maxVal = x.FloatData[i]
		}
	}

	// Compute exp and sum
	var sum float32
	for i := 0; i < x.Size; i++ {
		result[i] = float32(math.Exp(float64(x.FloatData[i] - maxVal)))
		sum += result[i]
	}

	// Normalize
	if sum > 0 {
		for i := 0; i < x.Size; i++ {
			result[i] /= sum
		}
	}

	return &Buffer{Size: x.Size, FloatData: result}, nil
}

// ReLU applies ReLU activation
func (b *CPUBackend) ReLU(x *Buffer) (*Buffer, error) {
	if x == nil {
		return nil, fmt.Errorf("nil input buffer")
	}

	result := make([]float32, x.Size)
	for i := 0; i < x.Size; i++ {
		if x.FloatData[i] > 0 {
			result[i] = x.FloatData[i]
		} else {
			result[i] = 0
		}
	}

	return &Buffer{Size: x.Size, FloatData: result}, nil
}

// LayerNorm applies layer normalization
func (b *CPUBackend) LayerNorm(x, gamma, beta *Buffer, rows, cols int, eps float32) (*Buffer, error) {
	if x == nil || gamma == nil || beta == nil {
		return nil, fmt.Errorf("nil input buffer")
	}

	result := make([]float32, rows*cols)

	for r := 0; r < rows; r++ {
		// Compute mean
		var mean float32
		for c := 0; c < cols; c++ {
			mean += x.FloatData[r*cols+c]
		}
		mean /= float32(cols)

		// Compute variance
		var variance float32
		for c := 0; c < cols; c++ {
			diff := x.FloatData[r*cols+c] - mean
			variance += diff * diff
		}
		variance /= float32(cols)

		// Normalize
		std := float32(math.Sqrt(float64(variance + eps)))
		for c := 0; c < cols; c++ {
			normalized := (x.FloatData[r*cols+c] - mean) / std
			result[r*cols+c] = normalized*gamma.FloatData[c] + beta.FloatData[c]
		}
	}

	return &Buffer{Size: rows * cols, FloatData: result}, nil
}

// Sum computes the sum of all elements
func (b *CPUBackend) Sum(x *Buffer) (float32, error) {
	if x == nil {
		return 0, fmt.Errorf("nil input buffer")
	}

	var sum float32
	for i := 0; i < x.Size; i++ {
		sum += x.FloatData[i]
	}
	return sum, nil
}

// Mean computes the mean of all elements
func (b *CPUBackend) Mean(x *Buffer) (float32, error) {
	sum, err := b.Sum(x)
	if err != nil {
		return 0, err
	}
	return sum / float32(x.Size), nil
}

// EmbeddingLookup looks up embedding vectors for given indices
func (b *CPUBackend) EmbeddingLookup(weights *Buffer, indices []int, vocabSize, embedDim int) (*Buffer, error) {
	if weights == nil {
		return nil, fmt.Errorf("nil weights buffer")
	}
	if len(weights.FloatData) != vocabSize*embedDim {
		return nil, fmt.Errorf("weights size mismatch: expected %d, got %d", vocabSize*embedDim, len(weights.FloatData))
	}

	seqLen := len(indices)
	result := make([]float32, seqLen*embedDim)

	for i, idx := range indices {
		if idx < 0 || idx >= vocabSize {
			return nil, fmt.Errorf("index %d out of range [0, %d)", idx, vocabSize)
		}
		for j := 0; j < embedDim; j++ {
			result[i*embedDim+j] = weights.FloatData[idx*embedDim+j]
		}
	}

	return &Buffer{Size: seqLen * embedDim, FloatData: result}, nil
}

// ParallelMatMul performs matrix multiplication using goroutines for large matrices
func (b *CPUBackend) ParallelMatMul(a, bBuf *Buffer, m, n, k int) (*Buffer, error) {
	if m*n*k < 1000000 {
		// Small matrix - use sequential
		return b.MatMul(a, bBuf, m, n, k)
	}

	// Large matrix - use parallel computation
	if a == nil || bBuf == nil {
		return nil, fmt.Errorf("nil input buffer")
	}

	aFloat64 := a.ToFloat64()
	bFloat64 := bBuf.ToFloat64()
	cData := make([]float64, m*n)

	numWorkers := 4
	rowsPerWorker := m / numWorkers

	var wg sync.WaitGroup
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if w == numWorkers-1 {
			endRow = m
		}

		go func(sRow, eRow int) {
			defer wg.Done()
			for i := sRow; i < eRow; i++ {
				for j := 0; j < n; j++ {
					var sum float64
					for l := 0; l < k; l++ {
						sum += aFloat64[i*k+l] * bFloat64[l*n+j]
					}
					cData[i*n+j] = sum
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return NewBuffer(cData), nil
}
