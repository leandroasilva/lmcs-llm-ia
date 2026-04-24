package model

import (
	"math"
	"math/rand"
	"sort"
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

// SampleTopK amostra considerando apenas o top-K probabilidades mais altas
func SampleTopK(probs []float64, k int) int {
	if k >= len(probs) {
		return Sample(probs)
	}

	// Criar lista de (probabilidade, índice)
	type probIdx struct {
		prob float64
		idx  int
	}

	pairs := make([]probIdx, len(probs))
	for i, p := range probs {
		pairs[i] = probIdx{p, i}
	}

	// Ordenar por probabilidade (decrescente)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})

	// Pegar top-K
	topK := pairs[:k]

	// Renormalizar probabilidades
	sum := 0.0
	for _, p := range topK {
		sum += p.prob
	}

	if sum == 0 {
		return topK[rand.Intn(len(topK))].idx
	}

	// Amostrar do top-K renormalizado
	r := rand.Float64()
	cumsum := 0.0
	for _, p := range topK {
		cumsum += p.prob / sum
		if r <= cumsum {
			return p.idx
		}
	}

	return topK[len(topK)-1].idx
}

// Generator helper para construção eficiente de strings
type Generator struct {
	strings.Builder
}

// NewGenerator cria um novo Generator
func NewGenerator() *Generator {
	return &Generator{}
}
