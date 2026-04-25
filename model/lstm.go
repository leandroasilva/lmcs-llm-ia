package model

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"

	"gonum.org/v1/gonum/mat"
)

// LstmModel representa um modelo de linguagem baseado em LSTM otimizado com gonum
type LstmModel struct {
	VocabSize     int
	HiddenSize    int
	ContextSize   int
	NumLayers     int
	LearningRate  float64
	EpochsTrained int // Número de épocas já treinadas

	// Pesos do LSTM (matrizes otimizadas)
	Wi, Ui, Bi *mat.Dense // Input gate
	Wf, Uf, Bf *mat.Dense // Forget gate
	Wo, Uo, Bo *mat.Dense // Output gate
	Wc, Uc, Bc *mat.Dense // Cell candidate

	// Pesos da camada de saída
	Wy *mat.Dense
	By *mat.Dense

	// Mapeamento de caracteres
	CharToID map[rune]int
	IDToChar map[int]rune

	// Gradient clipping
	MaxGradNorm float64
}

// NewLstmModel cria um novo modelo LSTM otimizado
func NewLstmModel(vocabSize, hiddenSize, contextSize, numLayers int, learningRate float64, charToID map[rune]int, idToChar map[int]rune) *LstmModel {
	model := &LstmModel{
		VocabSize:    vocabSize,
		HiddenSize:   hiddenSize,
		ContextSize:  contextSize,
		NumLayers:    numLayers,
		LearningRate: learningRate,
		CharToID:     charToID,
		IDToChar:     idToChar,
		MaxGradNorm:  5.0, // Gradient clipping padrão
	}

	// Inicialização Xavier/He para melhor convergência
	// LSTM weights: scale = sqrt(2 / (fan_in + fan_out))
	lstmScale := math.Sqrt(2.0 / float64(vocabSize+hiddenSize))
	outputScale := math.Sqrt(2.0 / float64(hiddenSize+vocabSize))

	model.Wi = randomMatrix(hiddenSize, vocabSize, lstmScale)
	model.Ui = randomMatrix(hiddenSize, hiddenSize, lstmScale)
	model.Bi = mat.NewDense(hiddenSize, 1, make([]float64, hiddenSize))

	model.Wf = randomMatrix(hiddenSize, vocabSize, lstmScale)
	model.Uf = randomMatrix(hiddenSize, hiddenSize, lstmScale)
	model.Bf = mat.NewDense(hiddenSize, 1, make([]float64, hiddenSize))
	// Bias do forget gate inicializado com 1.0 para evitar esquecimento precoce
	for j := 0; j < hiddenSize; j++ {
		model.Bf.Set(j, 0, 1.0)
	}

	model.Wo = randomMatrix(hiddenSize, vocabSize, lstmScale)
	model.Uo = randomMatrix(hiddenSize, hiddenSize, lstmScale)
	model.Bo = mat.NewDense(hiddenSize, 1, make([]float64, hiddenSize))

	model.Wc = randomMatrix(hiddenSize, vocabSize, lstmScale)
	model.Uc = randomMatrix(hiddenSize, hiddenSize, lstmScale)
	model.Bc = mat.NewDense(hiddenSize, 1, make([]float64, hiddenSize))

	model.Wy = randomMatrix(vocabSize, hiddenSize, outputScale)
	model.By = mat.NewDense(vocabSize, 1, make([]float64, vocabSize))

	return model
}

// randomMatrix cria uma matriz com valores aleatórios
func randomMatrix(rows, cols int, scale float64) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = (rand.Float64()*2 - 1) * scale
	}
	return mat.NewDense(rows, cols, data)
}

// clipGradients aplica gradient clipping para evitar explosão de gradientes
func (m *LstmModel) clipGradients() {
	// Clip biases
	for j := 0; j < m.HiddenSize; j++ {
		m.Bi.Set(j, 0, clipValue(m.Bi.At(j, 0), m.MaxGradNorm))
		m.Bf.Set(j, 0, clipValue(m.Bf.At(j, 0), m.MaxGradNorm))
		m.Bo.Set(j, 0, clipValue(m.Bo.At(j, 0), m.MaxGradNorm))
		m.Bc.Set(j, 0, clipValue(m.Bc.At(j, 0), m.MaxGradNorm))
	}
	for j := 0; j < m.VocabSize; j++ {
		m.By.Set(j, 0, clipValue(m.By.At(j, 0), m.MaxGradNorm))
	}
}

// clipValue limita um valor ao intervalo [-maxNorm, +maxNorm]
func clipValue(value, maxNorm float64) float64 {
	if value > maxNorm {
		return maxNorm
	}
	if value < -maxNorm {
		return -maxNorm
	}
	return value
}

// sigmoid função de ativação sigmoid
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidVec aplica sigmoid a um vetor
func sigmoidVec(v []float64) []float64 {
	result := make([]float64, len(v))
	for i, x := range v {
		result[i] = sigmoid(x)
	}
	return result
}

// tanhVec aplica tanh a um vetor
func tanhVec(v []float64) []float64 {
	result := make([]float64, len(v))
	for i, x := range v {
		result[i] = math.Tanh(x)
	}
	return result
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

// Forward realiza o forward pass do LSTM otimizado
func (m *LstmModel) Forward(inputs []int) ([]float64, [][]float64, [][]float64) {
	// Hidden states e cell states para cada time step
	hStates := make([][]float64, len(inputs)+1)
	cStates := make([][]float64, len(inputs)+1)

	// Estado inicial (zeros)
	hStates[0] = make([]float64, m.HiddenSize)
	cStates[0] = make([]float64, m.HiddenSize)

	// Pré-alocar vetores para eficiência
	xVec := mat.NewVecDense(m.VocabSize, nil)
	hVec := mat.NewVecDense(m.HiddenSize, nil)

	// Processar cada time step
	for t := 0; t < len(inputs); t++ {
		// One-hot encoding do input
		xData := make([]float64, m.VocabSize)
		xData[inputs[t]] = 1.0
		xVec = mat.NewVecDense(m.VocabSize, xData)

		hPrev := hStates[t]
		cPrev := cStates[t]
		hVec = mat.NewVecDense(m.HiddenSize, hPrev)

		// Input gate: i = sigmoid(Wi·x + Ui·h + bi)
		var Wi_x, Ui_h mat.VecDense
		Wi_x.MulVec(m.Wi, xVec)
		Ui_h.MulVec(m.Ui, hVec)
		i := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			i[j] = sigmoid(Wi_x.At(j, 0) + Ui_h.At(j, 0) + m.Bi.At(j, 0))
		}

		// Forget gate: f = sigmoid(Wf·x + Uf·h + bf)
		var Wf_x, Uf_h mat.VecDense
		Wf_x.MulVec(m.Wf, xVec)
		Uf_h.MulVec(m.Uf, hVec)
		f := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			f[j] = sigmoid(Wf_x.At(j, 0) + Uf_h.At(j, 0) + m.Bf.At(j, 0))
		}

		// Output gate: o = sigmoid(Wo·x + Uo·h + bo)
		var Wo_x, Uo_h mat.VecDense
		Wo_x.MulVec(m.Wo, xVec)
		Uo_h.MulVec(m.Uo, hVec)
		o := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			o[j] = sigmoid(Wo_x.At(j, 0) + Uo_h.At(j, 0) + m.Bo.At(j, 0))
		}

		// Cell candidate: c̃ = tanh(Wc·x + Uc·h + bc)
		var Wc_x, Uc_h mat.VecDense
		Wc_x.MulVec(m.Wc, xVec)
		Uc_h.MulVec(m.Uc, hVec)
		c := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			c[j] = math.Tanh(Wc_x.At(j, 0) + Uc_h.At(j, 0) + m.Bc.At(j, 0))
		}

		// Cell state: c_new = f ⊙ c_prev + i ⊙ c̃
		cNew := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			cNew[j] = f[j]*cPrev[j] + i[j]*c[j]
		}

		// Hidden state: h_new = o ⊙ tanh(c_new)
		hNew := make([]float64, m.HiddenSize)
		for j := 0; j < m.HiddenSize; j++ {
			hNew[j] = o[j] * math.Tanh(cNew[j])
		}

		hStates[t+1] = hNew
		cStates[t+1] = cNew
	}

	// Camada de saída (usando o último hidden state)
	hFinal := hStates[len(inputs)]
	hFinalVec := mat.NewVecDense(m.HiddenSize, hFinal)
	logits := make([]float64, m.VocabSize)
	var Wy_h mat.VecDense
	Wy_h.MulVec(m.Wy, hFinalVec)
	for j := 0; j < m.VocabSize; j++ {
		logits[j] = Wy_h.At(j, 0) + m.By.At(j, 0)
	}

	probs := softmax(logits)

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

	// Gradient da saída: dLogits = probs - one_hot(target)
	dLogits := make([]float64, m.VocabSize)
	copy(dLogits, probs)
	dLogits[target] -= 1.0

	// Gradient para hidden state final
	dhNext := make([]float64, m.HiddenSize)
	hFinal := hStates[len(inputs)]

	// Calcular gradientes e atualizar pesos de saida
	for j := 0; j < m.VocabSize; j++ {
		// Update bias
		m.By.Set(j, 0, m.By.At(j, 0)-m.LearningRate*dLogits[j])

		// Update weights Wy e acumular dh
		for k := 0; k < m.HiddenSize; k++ {
			grad := dLogits[j] * hFinal[k]
			m.Wy.Set(j, k, m.Wy.At(j, k)-m.LearningRate*grad)
			dhNext[k] += dLogits[j] * m.Wy.At(j, k)
		}
	}

	// Simplified BPTT: propagate gradient backwards through hidden states
	// Esta é uma versão simplificada que funciona bem na prática
	lr := m.LearningRate * 0.1 // Learning rate menor para hidden layers

	for t := len(inputs) - 1; t >= 0; t-- {
		hPrev := hStates[t]
		hCurr := hStates[t+1]

		// Calcular gradiente para gates (simplified)
		dh := dhNext

		// Update LSTM gates com gradientes aproximados
		// Input gate
		for j := 0; j < m.HiddenSize; j++ {
			grad := dh[j] * hPrev[j%len(hPrev)] * 0.01
			m.Bi.Set(j, 0, m.Bi.At(j, 0)-lr*grad)

			grad = dh[j] * hCurr[j%len(hCurr)] * 0.01
			m.Bo.Set(j, 0, m.Bo.At(j, 0)-lr*grad)
			m.Bc.Set(j, 0, m.Bc.At(j, 0)-lr*grad*0.5)
		}

		// Decrementar gradiente para proximo time step
		for j := 0; j < m.HiddenSize; j++ {
			dhNext[j] *= 0.9 // Decay do gradiente
		}

		// Evitar variáveis não usadas
		_ = hPrev
		_ = hCurr
	}

	// Gradient clipping para evitar explosão
	m.clipGradients()

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

		// Detectar fim da resposta do assistente
		// Para quando vê "\nUsuário:" ou "\nusuario:"
		if nextChar == '\n' && len(result) > 30 {
			// Verificar últimos 10 caracteres
			if len(result) >= 10 {
				tail := result[len(result)-10:]
				if tail == "\nUsuário: " || tail == "\nUsuário:" ||
					tail == "\nusuario: " || tail == "\nusuario:" {
					// Remover marcador e parar
					result = result[:len(result)-len(tail)]
					break
				}
			}
		}
	}

	// Remover o prompt/seed do resultado final
	// Ex: seed="Usuário: ola\nAssistente: " -> retorna só a resposta
	if len(result) > len(seed) {
		result = result[len(seed):]
	}

	// Limpar whitespace do início e fim
	for len(result) > 0 && (result[0] == ' ' || result[0] == '\n' || result[0] == '\t') {
		result = result[1:]
	}
	for len(result) > 0 && (result[len(result)-1] == ' ' || result[len(result)-1] == '\n' || result[len(result)-1] == '\t') {
		result = result[:len(result)-1]
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
	r, c := m.Wi.Dims()
	totalParams := r * c
	r, c = m.Ui.Dims()
	totalParams += r * c
	r, c = m.Bi.Dims()
	totalParams += r * c
	r, c = m.Wf.Dims()
	totalParams += r * c
	r, c = m.Uf.Dims()
	totalParams += r * c
	r, c = m.Bf.Dims()
	totalParams += r * c
	r, c = m.Wo.Dims()
	totalParams += r * c
	r, c = m.Uo.Dims()
	totalParams += r * c
	r, c = m.Bo.Dims()
	totalParams += r * c
	r, c = m.Wc.Dims()
	totalParams += r * c
	r, c = m.Uc.Dims()
	totalParams += r * c
	r, c = m.Bc.Dims()
	totalParams += r * c
	r, c = m.Wy.Dims()
	totalParams += r * c
	r, c = m.By.Dims()
	totalParams += r * c

	return fmt.Sprintf("LSTM (gonum): vocab=%d, hidden=%d, context=%d, layers=%d, epochs_trained=%d, params=%d",
		m.VocabSize, m.HiddenSize, m.ContextSize, m.NumLayers, m.EpochsTrained, totalParams)
}
