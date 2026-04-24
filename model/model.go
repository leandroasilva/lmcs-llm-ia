package model

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
	"time"
)

// LmcsLLM representa o modelo de linguagem em nível de caractere
type LmcsLLM struct {
	Weights  [][]float64
	Chars    []rune
	CharToID map[rune]int
	IDToChar map[int]rune
	Size     int
	mu       sync.RWMutex
}

// New cria e inicializa um novo modelo
func New(text string) *LmcsLLM {
	log.Println("Inicializando novo modelo...")

	uniqueMap := make(map[rune]bool)
	for _, r := range text {
		uniqueMap[r] = true
	}

	var chars []rune
	for r := range uniqueMap {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	size := len(chars)
	cToID := make(map[rune]int)
	idToC := make(map[int]rune)
	weights := make([][]float64, size)

	// Semente para reproducibilidade
	rand.Seed(time.Now().UnixNano())

	for i, r := range chars {
		cToID[r] = i
		idToC[i] = r
		weights[i] = make([]float64, size)
		for j := range weights[i] {
			// Inicialização Xavier simplificada
			weights[i][j] = (rand.Float64() - 0.5) * 2.0 / float64(size)
		}
	}

	log.Printf("Modelo criado com %d caracteres únicos\n", size)
	return &LmcsLLM{
		Weights:  weights,
		Chars:    chars,
		CharToID: cToID,
		IDToChar: idToC,
		Size:     size,
		mu:       sync.RWMutex{},
	}
}

// Load carrega um modelo salvo em disco
func Load(path string) (*LmcsLLM, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("erro ao abrir modelo: %w", err)
	}
	defer file.Close()

	var m LmcsLLM
	if err := gob.NewDecoder(file).Decode(&m); err != nil {
		return nil, fmt.Errorf("erro ao decodificar modelo: %w", err)
	}

	m.mu = sync.RWMutex{} // Reinicializar mutex
	return &m, nil
}

// Save salva o modelo em disco
func (m *LmcsLLM) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("erro ao criar arquivo: %w", err)
	}
	defer file.Close()

	if err := gob.NewEncoder(file).Encode(m); err != nil {
		return fmt.Errorf("erro ao codificar modelo: %w", err)
	}

	return nil
}

// Train treina o modelo com o texto fornecido
func (m *LmcsLLM) Train(text string, epochs int, lr float64, batchSize int) {
	runes := []rune(text)
	n := len(runes)

	if n < 2 {
		log.Println("Texto muito curto para treinamento")
		return
	}

	log.Printf("Iniciando treinamento: %d épocas, lr=%.4f, batch=%d\n", epochs, lr, batchSize)

	for e := 1; e <= epochs; e++ {
		startTime := time.Now()
		totalLoss := 0.0
		samples := 0

		// Processar em mini-batches
		for batchStart := 0; batchStart < n-1; batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > n-1 {
				batchEnd = n - 1
			}

			// Acumular gradientes para o batch
			gradients := make([][]float64, m.Size)
			for i := range gradients {
				gradients[i] = make([]float64, m.Size)
			}

			batchLoss := 0.0
			batchSamples := 0

			// Processar cada par no batch sequencialmente
			for i := batchStart; i < batchEnd; i++ {
				curr, ok1 := m.CharToID[runes[i]]
				next, ok2 := m.CharToID[runes[i+1]]
				if !ok1 || !ok2 {
					continue
				}

				// Forward
				logits := make([]float64, m.Size)
				copy(logits, m.Weights[curr])

				probs := Softmax(logits)

				// Calcular loss
				loss := -math.Log(probs[next] + 1e-9)

				// Calcular e acumular gradientes
				for j := 0; j < m.Size; j++ {
					target := 0.0
					if j == next {
						target = 1.0
					}
					gradients[curr][j] += probs[j] - target
				}

				batchLoss += loss
				batchSamples++
			}

			// Atualizar pesos com gradientes acumulados
			if batchSamples > 0 {
				for i := range gradients {
					for j := range gradients[i] {
						if gradients[i][j] != 0 {
							m.Weights[i][j] -= lr * gradients[i][j] / float64(batchSamples)
						}
					}
				}
			}

			totalLoss += batchLoss
			samples += batchSamples
		}

		elapsed := time.Since(startTime)
		avgLoss := totalLoss / float64(samples)
		log.Printf("Época %d/%d - Loss: %.4f | Tempo: %v\n", e, epochs, avgLoss, elapsed)
	}
}

// Generate gera texto a partir de uma semente
func (m *LmcsLLM) Generate(seed rune, length int, temperature float64) string {
	if temperature <= 0 {
		temperature = 0.8
	}

	result := NewGenerator()
	result.WriteRune(seed)
	curr := seed

	for i := 0; i < length; i++ {
		id, ok := m.CharToID[curr]
		if !ok {
			break
		}

		m.mu.RLock()
		logits := make([]float64, m.Size)
		copy(logits, m.Weights[id])
		m.mu.RUnlock()

		// Aplicar temperatura
		if temperature != 1.0 {
			for j := range logits {
				logits[j] /= temperature
			}
		}

		probs := Softmax(logits)

		// Amostragem categórica
		nextIdx := Sample(probs)
		curr = m.IDToChar[nextIdx]
		result.WriteRune(curr)
	}

	return result.String()
}
