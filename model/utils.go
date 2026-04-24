package model

import (
	"math"
	"math/rand"
	"strings"
)

// Softmax calcula a função softmax sobre um vetor de logits
func Softmax(x []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	res := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}

	for i := range res {
		res[i] /= sum
	}

	return res
}

// Sample amostra um índice baseado nas probabilidades
func Sample(probs []float64) int {
	r := rand.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// Generator helper para construção eficiente de strings
type Generator struct {
	strings.Builder
}

// NewGenerator cria um novo Generator
func NewGenerator() *Generator {
	return &Generator{}
}
