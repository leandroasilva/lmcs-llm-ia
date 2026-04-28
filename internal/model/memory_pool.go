package model

import (
	"sync"

	"gonum.org/v1/gonum/mat"
)

// matrixPoolCache gerencia pools de matrizes para reutilização
type matrixPoolCache struct {
	pools map[string]*sync.Pool
	mu    sync.RWMutex
}

// Global pool cache
var globalMatrixPool = &matrixPoolCache{
	pools: make(map[string]*sync.Pool),
}

// getMatrixPoolKey cria uma chave única para o pool baseado nas dimensões
func getMatrixPoolKey(rows, cols int) string {
	return matKey(rows, cols)
}

// matKey gera chave para matriz
func matKey(rows, cols int) string {
	// Usando array fixo para melhor performance
	if rows < 1000 && cols < 1000 {
		return string(rune(rows<<16 | cols))
	}
	// Fallback para matrizes grandes
	return string(rune(rows%1000<<16 | cols%1000))
}

// getMatrixFromPool obtém uma matriz do pool ou cria uma nova
func getMatrixFromPool(rows, cols int) *mat.Dense {
	key := getMatrixPoolKey(rows, cols)

	globalMatrixPool.mu.RLock()
	pool, exists := globalMatrixPool.pools[key]
	globalMatrixPool.mu.RUnlock()

	if !exists {
		// Criar novo pool para estas dimensões
		pool = &sync.Pool{
			New: func() interface{} {
				return mat.NewDense(rows, cols, nil)
			},
		}

		globalMatrixPool.mu.Lock()
		globalMatrixPool.pools[key] = pool
		globalMatrixPool.mu.Unlock()
	}

	// Obter matriz do pool
	m := pool.Get().(*mat.Dense)

	// Redimensionar se necessário (reutilizando underlying slice)
	data := make([]float64, rows*cols)
	m.Mul(mat.NewDense(rows, cols, data), mat.NewDense(rows, cols, nil))

	return m
}

// putMatrixToPool devolve uma matriz ao pool
func putMatrixToPool(m *mat.Dense) {
	if m == nil {
		return
	}

	rows, cols := m.Dims()
	key := getMatrixPoolKey(rows, cols)

	globalMatrixPool.mu.RLock()
	pool, exists := globalMatrixPool.pools[key]
	globalMatrixPool.mu.RUnlock()

	if exists {
		// Resetar matriz antes de devolver ao pool
		m.Reset()
		pool.Put(m)
	}
	// Se pool não existe, deixar GC coletar
}

// slicePool gerencia pools de slices float64
type slicePool struct {
	pools map[int]*sync.Pool // key: capacity
	mu    sync.RWMutex
}

var globalSlicePool = &slicePool{
	pools: make(map[int]*sync.Pool),
}

// nextPowerOfTwo calcula a próxima potência de 2
func nextPowerOfTwo(n int) int {
	if n <= 0 {
		return 1
	}

	n--
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	n++

	return n
}

// vecPool é um pool específico para vetores (1D arrays)
var vecPool = sync.Pool{
	New: func() interface{} {
		s := make([]float64, 0, 1024)
		return &s
	},
}

// getVec obtém vetor do pool
func getVec(size int) []float64 {
	ptr := vecPool.Get().(*[]float64)
	s := *ptr

	if cap(s) < size {
		s = make([]float64, size)
	} else {
		s = s[:size]
	}

	return s
}

// putVec devolve vetor ao pool
func putVec(s []float64) {
	if cap(s) <= 1024 {
		// Limpar
		for i := range s {
			s[i] = 0
		}
		s = s[:0]
		ptr := &s
		vecPool.Put(ptr)
	}
	// Vetores grandes são deixados para o GC
}

// rowPool é um pool para matrizes de uma linha (1 x N)
var rowPool = sync.Pool{
	New: func() interface{} {
		return mat.NewDense(1, 64, nil)
	},
}

// getRowMatrix obtém matriz de linha do pool
func getRowMatrix(cols int) *mat.Dense {
	m := rowPool.Get().(*mat.Dense)
	data := make([]float64, cols)
	m = mat.NewDense(1, cols, data)
	return m
}

// putRowMatrix devolve matriz de linha ao pool
func putRowMatrix(m *mat.Dense) {
	if m != nil {
		rows, cols := m.Dims()
		if rows == 1 && cols <= 1024 {
			m.Reset()
			rowPool.Put(m)
		}
	}
}
