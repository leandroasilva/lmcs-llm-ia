package model

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// TransformerModel representa um modelo Transformer small para conversação
type TransformerModel struct {
	VocabSize     int
	DModel        int // Dimension do modelo
	NHeads        int // Número de attention heads
	NLayers       int // Número de transformer layers
	MaxSeqLen     int // Tamanho máximo da sequência
	FFHidden      int // Hidden size do feed-forward
	LearningRate  float64
	EpochsTrained int

	// Embeddings
	TokenEmbedding    *mat.Dense // [vocab_size, d_model]
	PositionEmbedding *mat.Dense // [max_seq_len, d_model]

	// Transformer layers
	TransformerLayers []TransformerLayer

	// Output layer
	Layernorm *mat.Dense // [d_model, d_model]
	WOut      *mat.Dense // [vocab_size, d_model]
	BOut      *mat.Dense // [vocab_size, 1]

	// Tokenizer
	Vocab         []string       // Vocabulary (word-level)
	WordToID      map[string]int // Word to ID mapping
	IDToWord      map[int]string // ID to word mapping
	SpecialTokens map[string]int // Special tokens (<PAD>, <UNK>, etc.)
}

// TransformerLayer representa uma camada do Transformer
type TransformerLayer struct {
	// Multi-head attention
	WQ, WK, WV *mat.Dense // [d_model, d_model]
	WO         *mat.Dense // [d_model, d_model]

	// Feed-forward
	W1 *mat.Dense // [ff_hidden, d_model]
	B1 *mat.Dense // [ff_hidden, 1]
	W2 *mat.Dense // [d_model, ff_hidden]
	B2 *mat.Dense // [d_model, 1]

	// Layer normalization
	LN1Weight, LN1Bias *mat.Dense // [d_model, 1]
	LN2Weight, LN2Bias *mat.Dense // [d_model, 1]
}

// NewTransformerModel cria um novo modelo Transformer
func NewTransformerModel(vocabSize, dModel, nHeads, nLayers, maxSeqLen, ffHidden int, learningRate float64) *TransformerModel {
	model := &TransformerModel{
		VocabSize:         vocabSize,
		DModel:            dModel,
		NHeads:            nHeads,
		NLayers:           nLayers,
		MaxSeqLen:         maxSeqLen,
		FFHidden:          ffHidden,
		LearningRate:      learningRate,
		TransformerLayers: make([]TransformerLayer, nLayers),
	}

	// Inicialização Xavier
	scale := math.Sqrt(2.0 / float64(dModel))

	// Token embeddings
	model.TokenEmbedding = transformerRandomMatrix(vocabSize, dModel, scale)
	model.PositionEmbedding = transformerRandomMatrix(maxSeqLen, dModel, scale)

	// Criar transformer layers
	for i := 0; i < nLayers; i++ {
		layer := &model.TransformerLayers[i]

		// Attention weights
		attnScale := math.Sqrt(2.0 / float64(dModel+dModel))
		layer.WQ = transformerRandomMatrix(dModel, dModel, attnScale)
		layer.WK = transformerRandomMatrix(dModel, dModel, attnScale)
		layer.WV = transformerRandomMatrix(dModel, dModel, attnScale)
		layer.WO = transformerRandomMatrix(dModel, dModel, attnScale)

		// Feed-forward weights
		ffScale := math.Sqrt(2.0 / float64(dModel+ffHidden))
		layer.W1 = transformerRandomMatrix(ffHidden, dModel, ffScale)
		layer.B1 = mat.NewDense(ffHidden, 1, make([]float64, ffHidden))
		layer.W2 = transformerRandomMatrix(dModel, ffHidden, ffScale)
		layer.B2 = mat.NewDense(dModel, 1, make([]float64, dModel))

		// Layer norm (init com 1 e 0)
		layer.LN1Weight = mat.NewDense(dModel, 1, onesVector(dModel))
		layer.LN1Bias = mat.NewDense(dModel, 1, make([]float64, dModel))
		layer.LN2Weight = mat.NewDense(dModel, 1, onesVector(dModel))
		layer.LN2Bias = mat.NewDense(dModel, 1, make([]float64, dModel))
	}

	// Output layer
	outScale := math.Sqrt(2.0 / float64(dModel+vocabSize))
	model.WOut = transformerRandomMatrix(vocabSize, dModel, outScale)
	model.BOut = mat.NewDense(vocabSize, 1, make([]float64, vocabSize))

	// Layer norm final
	model.Layernorm = mat.NewDense(dModel, dModel, identityMatrix(dModel))

	// Special tokens
	model.SpecialTokens = map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
		"<BOS>": 2,
		"<EOS>": 3,
	}

	return model
}

// BuildVocabTransformer constrói vocabulário word-level a partir do texto
func BuildVocabTransformer(text string, maxVocab int) ([]string, map[string]int, map[int]string) {
	// Tokenização simples por espaço
	words := strings.Fields(text)

	// Contar frequência
	wordFreq := make(map[string]int)
	for _, word := range words {
		word = strings.ToLower(word)
		// Remover pontuação básica
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		if len(word) > 0 {
			wordFreq[word]++
		}
	}

	// Ordenar por frequência
	type wordCount struct {
		word  string
		count int
	}
	var sorted []wordCount
	for word, count := range wordFreq {
		sorted = append(sorted, wordCount{word, count})
	}
	// Bubble sort simples (adequado para vocab pequeno)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].count > sorted[i].count {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	// Limitar vocabulário
	if len(sorted) > maxVocab-4 { // -4 para special tokens
		sorted = sorted[:maxVocab-4]
	}

	// Criar mappings
	vocab := []string{"<PAD>", "<UNK>", "<BOS>", "<EOS>"}
	wordToID := map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
		"<BOS>": 2,
		"<EOS>": 3,
	}
	idToWord := map[int]string{
		0: "<PAD>",
		1: "<UNK>",
		2: "<BOS>",
		3: "<EOS>",
	}

	for i, wc := range sorted {
		id := i + 4
		vocab = append(vocab, wc.word)
		wordToID[wc.word] = id
		idToWord[id] = wc.word
	}

	return vocab, wordToID, idToWord
}

// Tokenize converte texto para IDs
func (m *TransformerModel) Tokenize(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	tokens := []int{m.SpecialTokens["<BOS>"]} // Begin of sequence

	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		if id, ok := m.WordToID[word]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, m.SpecialTokens["<UNK>"])
		}
	}

	tokens = append(tokens, m.SpecialTokens["<EOS>"]) // End of sequence
	return tokens
}

// Detokenize converte IDs para texto
func (m *TransformerModel) Detokenize(tokens []int) string {
	var words []string
	for _, id := range tokens {
		if id == m.SpecialTokens["<PAD>"] || id == m.SpecialTokens["<BOS>"] || id == m.SpecialTokens["<EOS>"] {
			continue
		}
		if word, ok := m.IDToWord[id]; ok {
			words = append(words, word)
		}
	}
	return strings.Join(words, " ")
}

// Forward realiza forward pass do Transformer
func (m *TransformerModel) Forward(inputTokens []int) *mat.Dense {
	seqLen := len(inputTokens)
	if seqLen > m.MaxSeqLen {
		seqLen = m.MaxSeqLen
		inputTokens = inputTokens[:seqLen]
	}

	// Criar input embeddings + positional encoding
	// X [seq_len, d_model]
	X := mat.NewDense(seqLen, m.DModel, nil)
	for i, tokenID := range inputTokens {
		if i >= seqLen {
			break
		}

		// Token embedding
		for j := 0; j < m.DModel; j++ {
			val := m.TokenEmbedding.At(tokenID, j) + m.PositionEmbedding.At(i, j)
			X.Set(i, j, val)
		}
	}

	// Passar por transformer layers
	for i := 0; i < m.NLayers; i++ {
		X = transformerLayerForward(&m.TransformerLayers[i], X, seqLen, m.DModel, m.NHeads)
	}

	// Layer normalization final
	X = applyLayerNorm(X, seqLen, m.DModel)

	return X
}

// Generate gera texto a partir de um prompt
func (m *TransformerModel) Generate(prompt string, maxTokens int, temperature float64, topK int) string {
	// Tokenizar prompt
	inputTokens := m.Tokenize(prompt)

	// Gerar tokens um por um
	for i := 0; i < maxTokens; i++ {
		// Forward pass
		output := m.Forward(inputTokens)

		// Pegar última posição
		seqLen := output.RawMatrix().Rows
		lastRow := mat.NewDense(1, m.DModel, nil)
		for j := 0; j < m.DModel; j++ {
			lastRow.Set(0, j, output.At(seqLen-1, j))
		}

		// Calcular logits para próximo token
		logits := make([]float64, m.VocabSize)
		for v := 0; v < m.VocabSize; v++ {
			logits[v] = m.BOut.At(v, 0)
			for j := 0; j < m.DModel; j++ {
				logits[v] += m.WOut.At(v, j) * lastRow.At(0, j)
			}
		}

		// Aplicar softmax com temperatura
		probs := softmaxWithTemperature(logits, temperature)

		// Sample
		nextToken := sampleToken(probs, topK)

		// Verificar se é EOS
		if nextToken == m.SpecialTokens["<EOS>"] {
			break
		}

		inputTokens = append(inputTokens, nextToken)
	}

	// Converter para texto (remover prompt)
	promptTokens := m.Tokenize(prompt)
	generatedTokens := inputTokens[len(promptTokens):]
	return m.Detokenize(generatedTokens)
}

// SaveModel salva o modelo
func (m *TransformerModel) SaveModel(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(m)
}

// LoadTransformerModel carrega o modelo
func LoadTransformerModel(path string) (*TransformerModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var model TransformerModel
	decoder := gob.NewDecoder(file)
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// transformerRandomMatrix cria uma matriz com valores aleatórios
func transformerRandomMatrix(rows, cols int, scale float64) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = (rand.Float64()*2 - 1) * scale
	}
	return mat.NewDense(rows, cols, data)
}

func onesVector(size int) []float64 {
	v := make([]float64, size)
	for i := range v {
		v[i] = 1.0
	}
	return v
}

func identityMatrix(size int) []float64 {
	data := make([]float64, size*size)
	for i := 0; i < size; i++ {
		data[i*size+i] = 1.0
	}
	return data
}

func softmaxWithTemperature(logits []float64, temperature float64) []float64 {
	probs := make([]float64, len(logits))
	maxVal := -math.MaxFloat64

	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for i, v := range logits {
		probs[i] = math.Exp((v - maxVal) / temperature)
		sum += probs[i]
	}

	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func sampleToken(probs []float64, topK int) int {
	if topK > 0 && topK < len(probs) {
		// Top-K sampling
		type probIdx struct {
			prob float64
			idx  int
		}
		var sorted []probIdx
		for i, p := range probs {
			sorted = append(sorted, probIdx{p, i})
		}
		// Sort descending
		for i := 0; i < len(sorted); i++ {
			for j := i + 1; j < len(sorted); j++ {
				if sorted[j].prob > sorted[i].prob {
					sorted[i], sorted[j] = sorted[j], sorted[i]
				}
			}
		}

		// Keep only top-K
		topKProbs := make([]float64, topK)
		topKIdx := make([]int, topK)
		sum := 0.0
		for i := 0; i < topK; i++ {
			topKProbs[i] = sorted[i].prob
			topKIdx[i] = sorted[i].idx
			sum += topKProbs[i]
		}
		// Renormalize
		for i := range topKProbs {
			topKProbs[i] /= sum
		}

		// Sample
		r := rand.Float64()
		cumSum := 0.0
		for i, p := range topKProbs {
			cumSum += p
			if r <= cumSum {
				return topKIdx[i]
			}
		}
		return topKIdx[len(topKIdx)-1]
	}

	// Sample from full distribution
	r := rand.Float64()
	cumSum := 0.0
	for i, p := range probs {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}
	return len(probs) - 1
}

// transformerLayerForward realiza forward pass de uma camada
func transformerLayerForward(layer *TransformerLayer, X *mat.Dense, seqLen, dModel, nHeads int) *mat.Dense {
	// Multi-head self-attention
	X = multiHeadAttention(layer, X, seqLen, dModel, nHeads)

	// Add & Norm
	X = applyLayerNorm(X, seqLen, dModel)

	// Feed-forward
	X = feedForward(layer, X, seqLen, dModel)

	// Add & Norm
	X = applyLayerNorm(X, seqLen, dModel)

	return X
}

// multiHeadAttention implementa attention mechanism
func multiHeadAttention(layer *TransformerLayer, X *mat.Dense, seqLen, dModel, nHeads int) *mat.Dense {
	headDim := dModel / nHeads

	// Q, K, V projections
	Q := mat.NewDense(seqLen, dModel, nil)
	K := mat.NewDense(seqLen, dModel, nil)
	V := mat.NewDense(seqLen, dModel, nil)

	Q.Mul(X, layer.WQ)
	K.Mul(X, layer.WK)
	V.Mul(X, layer.WV)

	// Scaled dot-product attention (simplified single-head)
	// Attention scores: Q * K^T / sqrt(d_k)
	KT := K.T()

	scores := mat.NewDense(seqLen, seqLen, nil)
	scores.Mul(Q, KT)

	// Scale
	scale := 1.0 / math.Sqrt(float64(headDim))
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			scores.Set(i, j, scores.At(i, j)*scale)
		}
	}

	// Softmax por linha
	for i := 0; i < seqLen; i++ {
		row := make([]float64, seqLen)
		for j := 0; j < seqLen; j++ {
			row[j] = scores.At(i, j)
		}
		row = transformerSoftmax(row)
		for j := 0; j < seqLen; j++ {
			scores.Set(i, j, row[j])
		}
	}

	// Weighted sum: scores * V
	output := mat.NewDense(seqLen, dModel, nil)
	output.Mul(scores, V)

	// Output projection
	result := mat.NewDense(seqLen, dModel, nil)
	result.Mul(output, layer.WO)

	// Residual connection
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			result.Set(i, j, result.At(i, j)+X.At(i, j))
		}
	}

	return result
}

// feedForward implementa FFN
func feedForward(layer *TransformerLayer, X *mat.Dense, seqLen, dModel int) *mat.Dense {
	// X * W1^T + B1
	hidden := mat.NewDense(seqLen, layer.W1.RawMatrix().Rows, nil)
	hidden.Mul(X, layer.W1.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < hidden.RawMatrix().Cols; j++ {
			hidden.Set(i, j, hidden.At(i, j)+layer.B1.At(j, 0))
		}
	}

	// ReLU activation
	for i := 0; i < seqLen; i++ {
		for j := 0; j < hidden.RawMatrix().Cols; j++ {
			val := hidden.At(i, j)
			if val < 0 {
				hidden.Set(i, j, 0)
			}
		}
	}

	// Hidden * W2^T + B2
	output := mat.NewDense(seqLen, dModel, nil)
	output.Mul(hidden, layer.W2.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			output.Set(i, j, output.At(i, j)+layer.B2.At(j, 0))
		}
	}

	// Residual connection
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			output.Set(i, j, output.At(i, j)+X.At(i, j))
		}
	}

	return output
}

// applyLayerNorm aplica layer normalization
func applyLayerNorm(X *mat.Dense, seqLen, dModel int) *mat.Dense {
	result := mat.NewDense(seqLen, dModel, nil)

	for i := 0; i < seqLen; i++ {
		// Calcular mean e variance
		sum := 0.0
		for j := 0; j < dModel; j++ {
			sum += X.At(i, j)
		}
		mean := sum / float64(dModel)

		varSum := 0.0
		for j := 0; j < dModel; j++ {
			diff := X.At(i, j) - mean
			varSum += diff * diff
		}
		variance := varSum / float64(dModel)
		std := math.Sqrt(variance + 1e-5)

		// Normalizar
		for j := 0; j < dModel; j++ {
			norm := (X.At(i, j) - mean) / std
			result.Set(i, j, norm)
		}
	}

	return result
}

// transformerSoftmax calcula softmax
func transformerSoftmax(logits []float64) []float64 {
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
