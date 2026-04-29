package compute

import (
	"math"
	"testing"
)

func TestNewCPUBackend(t *testing.T) {
	backend := NewCPUBackend()
	if backend == nil {
		t.Fatal("expected non-nil backend")
	}

	info := backend.Info()
	if info.Type != DeviceCPU {
		t.Errorf("expected DeviceCPU, got %v", info.Type)
	}
	if info.Name != "CPU (gonum)" {
		t.Errorf("unexpected name: %s", info.Name)
	}

	if err := backend.Initialize(); err != nil {
		t.Fatalf("failed to initialize: %v", err)
	}
	defer backend.Release()
}

func TestCPUBackend_MatMul(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	// Test 2x3 @ 3x2 = 2x2
	// A = [[1, 2, 3], [4, 5, 6]]
	// B = [[7, 8], [9, 10], [11, 12]]
	// C = [[58, 64], [139, 154]]
	a := NewBuffer([]float64{1, 2, 3, 4, 5, 6})
	b := NewBuffer([]float64{7, 8, 9, 10, 11, 12})

	result, err := backend.MatMul(a, b, 2, 2, 3)
	if err != nil {
		t.Fatalf("MatMul failed: %v", err)
	}

	expected := []float64{58, 64, 139, 154}
	if result.Size != 4 {
		t.Errorf("expected size 4, got %d", result.Size)
	}

	result64 := result.ToFloat64()
	for i, exp := range expected {
		if math.Abs(result64[i]-exp) > 1e-5 {
			t.Errorf("result[%d] = %f, expected %f", i, result64[i], exp)
		}
	}
}

func TestCPUBackend_Add(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	a := NewBuffer([]float64{1, 2, 3, 4})
	b := NewBuffer([]float64{5, 6, 7, 8})

	result, err := backend.Add(a, b)
	if err != nil {
		t.Fatalf("Add failed: %v", err)
	}

	expected := []float64{6, 8, 10, 12}
	result64 := result.ToFloat64()
	for i, exp := range expected {
		if math.Abs(result64[i]-exp) > 1e-5 {
			t.Errorf("result[%d] = %f, expected %f", i, result64[i], exp)
		}
	}
}

func TestCPUBackend_Scale(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	a := NewBuffer([]float64{1, 2, 3, 4})

	result, err := backend.Scale(a, 2.0)
	if err != nil {
		t.Fatalf("Scale failed: %v", err)
	}

	expected := []float64{2, 4, 6, 8}
	result64 := result.ToFloat64()
	for i, exp := range expected {
		if math.Abs(result64[i]-exp) > 1e-5 {
			t.Errorf("result[%d] = %f, expected %f", i, result64[i], exp)
		}
	}
}

func TestCPUBackend_Softmax(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	x := NewBuffer([]float64{1.0, 2.0, 3.0})

	result, err := backend.Softmax(x, 0)
	if err != nil {
		t.Fatalf("Softmax failed: %v", err)
	}

	// Check that probabilities sum to 1
	var sum float64
	result64 := result.ToFloat64()
	for _, v := range result64 {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Errorf("softmax sum = %f, expected 1.0", sum)
	}

	// Check ordering: highest input -> highest probability
	if result64[2] <= result64[1] || result64[1] <= result64[0] {
		t.Errorf("softmax ordering incorrect: %v", result64)
	}
}

func TestCPUBackend_ReLU(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	x := NewBuffer([]float64{-1, 0, 1, -2, 3})

	result, err := backend.ReLU(x)
	if err != nil {
		t.Fatalf("ReLU failed: %v", err)
	}

	expected := []float64{0, 0, 1, 0, 3}
	result64 := result.ToFloat64()
	for i, exp := range expected {
		if math.Abs(result64[i]-exp) > 1e-5 {
			t.Errorf("result[%d] = %f, expected %f", i, result64[i], exp)
		}
	}
}

func TestCPUBackend_LayerNorm(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	// 2 rows, 3 cols
	x := NewBuffer([]float64{1, 2, 3, 4, 5, 6})
	gamma := NewBuffer([]float64{1, 1, 1})
	beta := NewBuffer([]float64{0, 0, 0})

	result, err := backend.LayerNorm(x, gamma, beta, 2, 3, 1e-5)
	if err != nil {
		t.Fatalf("LayerNorm failed: %v", err)
	}

	if result.Size != 6 {
		t.Errorf("expected size 6, got %d", result.Size)
	}

	// Each row should have mean ~0 and std ~1
	result64 := result.ToFloat64()
	for r := 0; r < 2; r++ {
		var mean float64
		for c := 0; c < 3; c++ {
			mean += result64[r*3+c]
		}
		mean /= 3
		if math.Abs(mean) > 1e-5 {
			t.Errorf("row %d mean = %f, expected ~0", r, mean)
		}
	}
}

func TestCPUBackend_EmbeddingLookup(t *testing.T) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	// Vocab: 3 tokens, embed_dim: 4
	// Token 0: [1, 2, 3, 4]
	// Token 1: [5, 6, 7, 8]
	// Token 2: [9, 10, 11, 12]
	weights := NewBuffer([]float64{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	})
	indices := []int{2, 0, 1}

	result, err := backend.EmbeddingLookup(weights, indices, 3, 4)
	if err != nil {
		t.Fatalf("EmbeddingLookup failed: %v", err)
	}

	expected := []float64{
		9, 10, 11, 12, // Token 2
		1, 2, 3, 4, // Token 0
		5, 6, 7, 8, // Token 1
	}

	result64 := result.ToFloat64()
	for i, exp := range expected {
		if math.Abs(result64[i]-exp) > 1e-5 {
			t.Errorf("result[%d] = %f, expected %f", i, result64[i], exp)
		}
	}
}

func TestBackendFactory(t *testing.T) {
	// Test CPU backend creation
	cpu, err := NewBackend(false)
	if err != nil {
		t.Fatalf("failed to create CPU backend: %v", err)
	}
	if cpu.Info().Type != DeviceCPU {
		t.Errorf("expected CPU backend, got %v", cpu.Info().Type)
	}
	cpu.Release()
}

func TestAutoSelectBackend(t *testing.T) {
	backend, err := AutoSelectBackend()
	if err != nil {
		t.Fatalf("AutoSelectBackend failed: %v", err)
	}
	if backend == nil {
		t.Fatal("expected non-nil backend")
	}
	defer backend.Release()

	info := backend.Info()
	t.Logf("Auto-selected backend: %s (%s)", info.Name, info.Type)
}

func BenchmarkCPUBackend_MatMul_256x256(b *testing.B) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	size := 256
	a := NewBuffer(make([]float64, size*size))
	c := NewBuffer(make([]float64, size*size))

	for i := range a.FloatData {
		a.FloatData[i] = float32(i) * 0.001
		c.FloatData[i] = float32(i) * 0.002
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := backend.MatMul(a, c, size, size, size)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkCPUBackend_MatMul_512x512(b *testing.B) {
	backend := NewCPUBackend()
	backend.Initialize()
	defer backend.Release()

	size := 512
	a := NewBuffer(make([]float64, size*size))
	c := NewBuffer(make([]float64, size*size))

	for i := range a.FloatData {
		a.FloatData[i] = float32(i) * 0.001
		c.FloatData[i] = float32(i) * 0.002
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := backend.MatMul(a, c, size, size, size)
		if err != nil {
			b.Fatal(err)
		}
	}
}
