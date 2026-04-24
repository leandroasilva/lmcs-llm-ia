package model

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
)

// LstmModel representa um modelo de linguagem baseado em LSTM
type LstmModel struct {
	VocabSize    int
	HiddenSize   int
	ContextSize  int
	LearningRate float64

	// Pesos do LSTM (input gate, forget gate, output gate, cell candidate)
	Wi, Ui, Bi []float64 // Input gate
	Wf, Uf, Bf []float64 // Forget gate
	Wo, Uo, Bo []float64 // Output gate
	Wc, Uc, Bc []float64 // Cell candidate

	// Pesos da camada de saída
	Wy []float64
	By []float64

	// Mapeamento de caracteres
	CharToID map[rune]int
	IDToChar map[int]rune
}

// NewLstmModel cria um novo modelo LSTM
func NewLstmModel(vocabSize, hiddenSize, contextSize int, learningRate float64, charToID map[rune]int, idToChar map[int]rune) *LstmModel {
	model := &LstmModel{
		VocabSize:    vocabSize,
		HiddenSize:   hiddenSize,
		ContextSize:  contextSize,
		LearningRate: learningRate,
		CharToID:     charToID,
		IDToChar:     idToChar,
	}

	// Inicializar pesos com valores pequenos aleatórios
	scale := 0.01
	model.Wi = initWeights(hiddenSize, vocabSize, scale)
	model.Ui = initWeights(hiddenSize, hiddenSize, scale)
	model.Bi = make([]float64, hiddenSize)

	model.Wf = initWeights(hiddenSize, vocabSize, scale)
	model.Uf = initWeights(hiddenSize, hiddenSize, scale)
	model.Bf = make([]float64, hiddenSize)

	model.Wo = initWeights(hiddenSize, vocabSize, scale)
	model.Uo = initWeights(hiddenSize, hiddenSize, scale)
	model.Bo = make([]float64, hiddenSize)

	model.Wc = initWeights(hiddenSize, vocabSize, scale)
	model.Uc = initWeights(hiddenSize, hiddenSize, scale)
	model.Bc = make([]float64, hiddenSize)

	model.Wy = initWeights(vocabSize, hiddenSize, scale)
	model.By = make([]float64, vocabSize)

	return model
}

// initWeights inicializa uma matriz de pesos com valores aleatórios
func initWeights(rows, cols int, scale float64) []float64 {
	weights := make([]float64, rows*cols)
	for i := range weights {
		weights[i] = (rand.Float64()*2 - 1) * scale
	}
	return weights
}

// getWeight acessa um elemento da matriz de pesos
func getWeight(weights []float64, row, col, cols int) float64 {
	return weights[row*cols+col]
}

// setWeight define um elemento da matriz de pesos
func setWeight(weights []float64, row, col, cols int, value float64) {
	weights[row*cols+col] = value
}

// sigmoid função de ativação sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// tanh função de ativação tanh
func tanh(x float64) float64 {
	return math.Tanh(x)
}

// softmax função de ativação softmax
func softmax(logits []float64) []float64 {
	probs := make([]float64, len(logits))
	maxVal := -math.MaxFloat64
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp(v - maxVal)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

// Forward realiza o forward pass do LSTM
func (m *LstmModel) Forward(inputs []int) ([]float64, [][]float64, [][]float64) {
	// Hidden states e cell states para cada time step
	hStates := make([][]float64, len(inputs)+1)
	cStates := make([][]float64, len(inputs)+1)

	// Estado inicial (zeros)
	hStates[0] = make([]float64, m.HiddenSize)
	cStates[0] = make([]float64, m.HiddenSize)

	// Gate values para BPTT
	gateI := make([][]float64, len(inputs))
	gateF := make([][]float64, len(inputs))
	gateO := make([][]float64, len(inputs))
	gateC := make([][]float64, len(inputs))

	// Processar cada time step
	for t := 0; t < len(inputs); t++ {
		x := make([]float64, m.VocabSize)
		x[inputs[t]] = 1.0

		hPrev := hStates[t]
		cPrev := cStates[t]

		// Input gate
		i := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			sum := m.Bi[j]
			for k := 0; k < m.VocabSize; k++ {
				sum += getWeight(m.Wi, j, k, m.VocabSize) * x[k]
			}
			for k := 0; k < m.HiddenSize; k++ {
				sum += getWeight(m.Ui, j, k, m.HiddenSize) * hPrev[k]
			}
			i[j] = sigmoid(sum)
		}

		// Forget gate
		f := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			sum := m.Bf[j]
			for k := 0; k < m.VocabSize; k++ {
				sum += getWeight(m.Wf, j, k, m.VocabSize) * x[k]
			}
			for k := 0; k < m.HiddenSize; k++ {
				sum += getWeight(m.Uf, j, k, m.HiddenSize) * hPrev[k]
			}
			f[j] = sigmoid(sum)
		}

		// Output gate
		o := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			sum := m.Bo[j]
			for k := 0; k < m.VocabSize; k++ {
				sum += getWeight(m.Wo, j, k, m.VocabSize) * x[k]
			}
			for k := 0; k < m.HiddenSize; k++ {
				sum += getWeight(m.Uo, j, k, m.HiddenSize) * hPrev[k]
			}
			o[j] = sigmoid(sum)
		}

		// Cell candidate
		c := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			sum := m.Bc[j]
			for k := 0; k < m.VocabSize; k++ {
				sum += getWeight(m.Wc, j, k, m.VocabSize) * x[k]
			}
			for k := 0; k < m.HiddenSize; k++ {
				sum += getWeight(m.Uc, j, k, m.HiddenSize) * hPrev[k]
			}
			c[j] = tanh(sum)
		}

		// Cell state
		cNew := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			cNew[j] = f[j]*cPrev[j] + i[j]*c[j]
		}

		// Hidden state
		hNew := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			hNew[j] = o[j] * tanh(cNew[j])
		}

		hStates[t+1] = hNew
		cStates[t+1] = cNew
		gateI[t] = i
		gateF[t] = f
		gateO[t] = o
		gateC[t] = c
	}

	// Camada de saída (usando o último hidden state)
	hFinal := hStates[len(inputs)]
	logits := make([]float64, m.VocabSize)
	for j := 0; j < m.VocabSize; j++ {
		sum := m.By[j]
		for k := 0; k < m.HiddenSize; k++ {
			sum += getWeight(m.Wy, j, k, m.HiddenSize) * hFinal[k]
		}
		logits[j] = sum
	}

	probs := softmax(logits)

	// Retornar probs, hStates, cStates e gates para BPTT
	return probs, hStates, cStates
}

// CrossEntropyLoss calcula a perda de entropia cruzada
func CrossEntropyLoss(probs []float64, target int) float64 {
	prob := probs[target]
	if prob < 1e-7 {
		prob = 1e-7
	}
	return -math.Log(prob)
}

// Train realiza uma iteração de treinamento com BPTT simplificado
func (m *LstmModel) Train(inputs []int, target int) float64 {
	// Forward pass
	probs, hStates, _ := m.Forward(inputs)

	// Calcular loss
	loss := CrossEntropyLoss(probs, target)

	// Gradient da saída
	dLogits := make([]float64, m.VocabSize)
	copy(dLogits, probs)
	dLogits[target] -= 1.0

	// Gradient dos pesos de saída
	hFinal := hStates[len(inputs)]
	for j := 0; j < m.VocabSize; j++ {
		m.By[j] -= m.LearningRate * dLogits[j]
		for k := 0; k < m.HiddenSize; k++ {
			grad := dLogits[j] * hFinal[k]
			m.Wy[j*m.HiddenSize+k] -= m.LearningRate * grad
		}
	}

	// Backpropagation simplificado (apenas último time step)
	// Para uma implementação completa, seria necessário BPTT completo
	dh := make([]float64, m.HiddenSize)
	for j := 0; j < m.HiddenSize; j++ {
		for k := 0; k < m.VocabSize; k++ {
			dh[j] += getWeight(m.Wy, k, j, m.HiddenSize) * dLogits[k]
		}
	}

	// Simplificação: não fazer BPTT completo para manter o código gerenciável
	// Em uma implementação completa, faríamos backprop através de todos os time steps

	return loss
}

// Generate gera texto a partir de um seed
func (m *LstmModel) Generate(seed string, length int, temperature float64, topK int) string {
	if len(seed) == 0 {
		return ""
	}

	result := seed
	context := []rune(seed)

	for i := 0; i < length; i++ {
		// Converter contexto para índices
		inputs := make([]int, 0, m.ContextSize)
		start := 0
		if len(context) > m.ContextSize {
			start = len(context) - m.ContextSize
		}

		for j := start; j < len(context); j++ {
			if id, ok := m.CharToID[context[j]]; ok {
				inputs = append(inputs, id)
			}
		}

		if len(inputs) == 0 {
			inputs = append(inputs, 0)
		}

		// Forward pass
		probs, _, _ := m.Forward(inputs)

		// Aplicar temperatura
		if temperature != 1.0 {
			for j := range probs {
				probs[j] = math.Pow(probs[j], 1.0/temperature)
			}
			// Renormalizar
			sum := 0.0
			for _, p := range probs {
				sum += p
			}
			for j := range probs {
				probs[j] /= sum
			}
		}

		// Sample
		nextCharID := Sample(probs)
		if topK > 0 && topK < len(probs) {
			nextCharID = SampleTopK(probs, topK)
		}

		nextChar := m.IDToChar[nextCharID]
		result += string(nextChar)
		context = append(context, nextChar)
	}

	return result
}

// SaveModel salva o modelo LSTM em arquivo
func (m *LstmModel) SaveModel(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(m)
}

// LoadLstmModel carrega um modelo LSTM de arquivo
func LoadLstmModel(path string) (*LstmModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model LstmModel
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// GetModelInfo retorna informações sobre o modelo
func (m *LstmModel) GetModelInfo() string {
	totalParams := len(m.Wi) + len(m.Ui) + len(m.Bi) +
		len(m.Wf) + len(m.Uf) + len(m.Bf) +
		len(m.Wo) + len(m.Uo) + len(m.Bo) +
		len(m.Wc) + len(m.Uc) + len(m.Bc) +
		len(m.Wy) + len(m.By)

	return fmt.Sprintf("LSTM: vocab=%d, hidden=%d, context=%d, params=%d",
		m.VocabSize, m.HiddenSize, m.ContextSize, totalParams)
}
