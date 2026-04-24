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
	Weights     [][]float64 // Matriz de pesos [vocab_size * context_size, vocab_size]
	Chars       []rune
	CharToID    map[rune]int
	IDToChar    map[int]rune
	VocabSize   int
	ContextSize int // Tamanho do contexto (quantos caracteres anteriores olhar)
	mu          sync.RWMutex
}

// New cria e inicializa um novo modelo
func New(text string, contextSize int) *LmcsLLM {
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

	vocabSize := len(chars)
	cToID := make(map[rune]int)
	idToC := make(map[int]rune)

	// Pesos: [vocab_size * context_size, vocab_size]
	inputSize := vocabSize * contextSize
	weights := make([][]float64, inputSize)

	// Semente para reproducibilidade
	rand.Seed(time.Now().UnixNano())

	for i, r := range chars {
		cToID[r] = i
		idToC[i] = r
	}

	// Inicializar pesos para cada posição do contexto
	for i := 0; i < inputSize; i++ {
		weights[i] = make([]float64, vocabSize)
		for j := range weights[i] {
			// Inicialização Xavier
			weights[i][j] = (rand.Float64() - 0.5) * 2.0 / float64(vocabSize)
		}
	}

	log.Printf("Modelo criado: vocab=%d, context=%d, params=%d\n",
		vocabSize, contextSize, inputSize*vocabSize)

	return &LmcsLLM{
		Weights:     weights,
		Chars:       chars,
		CharToID:    cToID,
		IDToChar:    idToC,
		VocabSize:   vocabSize,
		ContextSize: contextSize,
		mu:          sync.RWMutex{},
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
	log.Printf("Modelo carregado: vocab=%d, context=%d\n", m.VocabSize, m.ContextSize)
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

// Train treina o modelo com o texto fornecido usando contexto
func (m *LmcsLLM) Train(text string, epochs int, lr float64, batchSize int) {
	runes := []rune(text)
	n := len(runes)

	if n < m.ContextSize+1 {
		log.Println("Texto muito curto para treinamento")
		return
	}

	log.Printf("Iniciando treinamento: %d épocas, lr=%.4f, batch=%d, context=%d\n",
		epochs, lr, batchSize, m.ContextSize)

	for e := 1; e <= epochs; e++ {
		startTime := time.Now()
		totalLoss := 0.0
		samples := 0

		// Processar em mini-batches
		for batchStart := m.ContextSize; batchStart < n-1; batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > n-1 {
				batchEnd = n - 1
			}

			// Acumular gradientes para o batch
			gradients := make([][]float64, len(m.Weights))
			for i := range gradients {
				gradients[i] = make([]float64, m.VocabSize)
			}

			batchLoss := 0.0
			batchSamples := 0

			// Processar cada par no batch
			for i := batchStart; i < batchEnd; i++ {
				// Obter contexto (caracteres anteriores)
				context := runes[i-m.ContextSize : i]
				nextChar := runes[i+1]

				nextID, okNext := m.CharToID[nextChar]
				if !okNext {
					continue
				}

				// Converter contexto para índice
				contextIdx := m.contextToIndex(context)

				// Forward pass
				m.mu.RLock()
				logits := make([]float64, m.VocabSize)
				copy(logits, m.Weights[contextIdx])
				m.mu.RUnlock()

				probs := Softmax(logits)

				// Calcular loss (cross-entropy)
				loss := -math.Log(probs[nextID] + 1e-9)

				// Calcular e acumular gradientes
				for j := 0; j < m.VocabSize; j++ {
					target := 0.0
					if j == nextID {
						target = 1.0
					}
					gradients[contextIdx][j] += probs[j] - target
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

		// Log a cada 10 épocas ou última
		if e%10 == 0 || e == epochs {
			log.Printf("Época %d/%d - Loss: %.4f | Tempo: %v\n", e, epochs, avgLoss, elapsed)
		}
	}
}

// contextToIndex converte um contexto de runes para um índice linear
func (m *LmcsLLM) contextToIndex(context []rune) int {
	// Usar hash para mapear contexto para índice
	hash := 0
	for _, r := range context {
		charID, ok := m.CharToID[r]
		if !ok {
			charID = 0
		}
		hash = hash*31 + charID
	}

	return int(math.Abs(float64(hash))) % len(m.Weights)
}

// Generate gera texto a partir de um contexto inicial
func (m *LmcsLLM) Generate(seed string, length int, temperature float64, topK int) string {
	if temperature <= 0 {
		temperature = 0.7
	}
	if topK <= 0 {
		topK = 40
	}

	result := NewGenerator()
	result.WriteString(seed)

	// Usar os últimos contextSize caracteres como contexto
	contextRunes := []rune(seed)
	if len(contextRunes) > m.ContextSize {
		contextRunes = contextRunes[len(contextRunes)-m.ContextSize:]
	}

	for i := 0; i < length; i++ {
		// Padding se necessário
		for len(contextRunes) < m.ContextSize {
			contextRunes = append([]rune{contextRunes[0]}, contextRunes...)
		}

		// Obter índice do contexto
		contextIdx := m.contextToIndex(contextRunes)

		m.mu.RLock()
		logits := make([]float64, m.VocabSize)
		copy(logits, m.Weights[contextIdx])
		m.mu.RUnlock()

		// Aplicar temperatura
		if temperature != 1.0 {
			for j := range logits {
				logits[j] /= temperature
			}
		}

		probs := Softmax(logits)

		// Top-K sampling
		nextIdx := SampleTopK(probs, topK)
		nextChar := m.IDToChar[nextIdx]

		result.WriteRune(nextChar)

		// Atualizar contexto
		contextRunes = append(contextRunes[1:], nextChar)
	}

	return result.String()
}
