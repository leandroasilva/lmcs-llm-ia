package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"gonum.org/v1/gonum/mat"
)

// ModelConfig representa a configuração do modelo
type ModelConfig struct {
	VocabSize int     `json:"vocab_size"`
	DModel    int     `json:"d_model"`
	NHeads    int     `json:"n_heads"`
	NLayers   int     `json:"n_layers"`
	MaxSeqLen int     `json:"max_seq_len"`
	FFHidden  int     `json:"ff_hidden"`
	Dropout   float64 `json:"dropout"`
}

// ModelMetadata representa metadados do modelo
type ModelMetadata struct {
	Framework         string `json:"framework"`
	TrainingCompleted bool   `json:"training_completed"`
	FinalEpoch        int    `json:"final_epoch"`
	TotalParameters   int    `json:"total_parameters"`
}

// TrainedModel representa o modelo treinado
type TrainedModel struct {
	Metadata ModelMetadata          `json:"metadata"`
	Config   ModelConfig            `json:"config"`
	Weights  map[string]interface{} `json:"weights"`
}

// Vocabulary representa o vocabulário
type Vocabulary struct {
	VocabSize int            `json:"vocab_size"`
	WordToID  map[string]int `json:"word_to_id"`
	IDToWord  map[int]string `json:"-"` // Ignorado no JSON, populado manualmente
}

func main() {
	rand.Seed(time.Now().UnixNano())

	log.Println("=== Teste de Geração com Modelo Treinado (PyTorch) ===")

	modelPath := "config.trained.json"
	if len(os.Args) > 1 {
		modelPath = os.Args[1]
	}

	// Verificar se arquivo existe
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		log.Fatalf("Arquivo não encontrado: %s\n", modelPath)
	}

	// Obter tamanho do arquivo
	fileInfo, _ := os.Stat(modelPath)
	log.Printf("Arquivo: %s (%.2f MB)\n", modelPath, float64(fileInfo.Size())/1e6)

	// Carregar modelo
	log.Println("Carregando modelo...")
	startTime := time.Now()

	model, err := loadModelStreaming(modelPath)
	if err != nil {
		log.Fatalf("Erro ao carregar modelo: %v\n", err)
	}

	loadTime := time.Since(startTime)
	log.Printf("Modelo carregado em %.2fs\n", loadTime.Seconds())
	log.Printf("Config: vocab=%d, d_model=%d, heads=%d, layers=%d, seq_len=%d\n",
		model.Config.VocabSize, model.Config.DModel, model.Config.NHeads,
		model.Config.NLayers, model.Config.MaxSeqLen)
	log.Printf("Parâmetros: %d\n", model.Metadata.TotalParameters)

	// Carregar vocabulário
	vocabPath := "training_gpu/vocab_trained.json"
	vocab, err := loadVocabulary(vocabPath)
	if err != nil {
		log.Printf("Aviso: Não foi possível carregar vocabulário: %v\n", err)
		log.Println("Usando tokenização fallback...")
		vocab = nil
	} else {
		log.Printf("Vocabulário carregado: %d palavras\n", vocab.VocabSize)
	}

	// Testar geração
	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("TESTES DE GERAÇÃO DE TEXTO")
	fmt.Println(strings.Repeat("=", 70))

	// Prompts de teste
	prompts := []string{
		"Olá como posso ajudar",
		"O que é inteligência artificial",
		"Por favor me explique",
	}

	for i, prompt := range prompts {
		fmt.Printf("\n[Teste %d]\n", i+1)
		fmt.Printf("Prompt: %s\n", prompt)

		genStart := time.Now()
		generated := generateWithVocab(model, vocab, prompt, 20, 0.8)
		genTime := time.Since(genStart)

		fmt.Printf("Gerado: %s\n", generated)
		fmt.Printf("Tempo: %.2fs\n", genTime.Seconds())
	}

	fmt.Println("\n" + strings.Repeat("=", 70))
	fmt.Println("TESTES CONCLUÍDOS")
	fmt.Println(strings.Repeat("=", 70))
}

// loadModelStreaming carrega o modelo usando streaming para evitar consumo excessivo de memória
func loadModelStreaming(path string) (*TrainedModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Usar decoder streaming
	decoder := json.NewDecoder(bufio.NewReaderSize(file, 1024*1024*50)) // 50MB buffer

	var model TrainedModel
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("erro ao decodificar JSON: %w", err)
	}

	return &model, nil
}

// loadVocabulary carrega o vocabulário do JSON
func loadVocabulary(path string) (*Vocabulary, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var vocab Vocabulary
	if err := json.Unmarshal(data, &vocab); err != nil {
		return nil, err
	}

	// Popular IDToWord
	vocab.IDToWord = make(map[int]string)
	for word, id := range vocab.WordToID {
		vocab.IDToWord[id] = word
	}

	return &vocab, nil
}

// generateWithVocab gera texto usando vocabulário
func generateWithVocab(model *TrainedModel, vocab *Vocabulary, prompt string, maxTokens int, temperature float64) string {
	if vocab == nil {
		// Fallback para tokenização simples
		return generate(model, prompt, maxTokens, temperature)
	}

	config := model.Config

	// Tokenizar prompt usando vocabulário
	tokens := tokenizeWithVocab(prompt, vocab)

	// Gerar tokens
	for i := 0; i < maxTokens; i++ {
		// Forward pass
		hidden := forward(model, tokens)

		// Pegar última posição
		seqLen := len(tokens)
		lastHidden := make([]float64, config.DModel)
		for j := 0; j < config.DModel; j++ {
			lastHidden[j] = hidden.At(seqLen-1, j)
		}

		// Calcular logits
		logits := computeLogits(model, lastHidden)

		// Aplicar softmax com temperatura
		probs := softmaxWithTemperature(logits, temperature)

		// Sample token
		nextToken := sampleToken(probs)

		// Adicionar aos tokens
		tokens = append(tokens, nextToken)

		// Verificar token de fim
		if nextToken == 3 { // <EOS>
			break
		}
	}

	// Detokenizar usando vocabulário
	return detokenizeWithVocab(tokens, vocab)
}

// tokenizeWithVocab tokeniza usando vocabulário
func tokenizeWithVocab(text string, vocab *Vocabulary) []int {
	words := strings.Fields(text)
	tokens := []int{2} // <BOS>

	for _, word := range words {
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		word = strings.ToLower(word)

		if len(word) == 0 {
			continue
		}

		// Buscar no vocabulário
		if id, ok := vocab.WordToID[word]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, 1) // <UNK>
		}
	}

	tokens = append(tokens, 3) // <EOS>
	return tokens
}

// detokenizeWithVocab detokeniza usando vocabulário
func detokenizeWithVocab(tokens []int, vocab *Vocabulary) string {
	var words []string
	for _, t := range tokens {
		// Pular tokens especiais
		if t <= 3 {
			continue
		}

		// Converter ID para palavra
		if word, ok := vocab.IDToWord[t]; ok {
			words = append(words, word)
		} else {
			words = append(words, fmt.Sprintf("[UNK:%d]", t))
		}
	}
	return strings.Join(words, " ")
}

// generate gera texto a partir de um prompt
func generate(model *TrainedModel, prompt string, maxTokens int, temperature float64) string {
	config := model.Config

	// Tokenizar prompt
	tokens := simpleTokenize(prompt, config.VocabSize)

	// Gerar tokens
	for i := 0; i < maxTokens; i++ {
		// Forward pass
		hidden := forward(model, tokens)

		// Pegar última posição
		seqLen := len(tokens)
		lastHidden := make([]float64, config.DModel)
		for j := 0; j < config.DModel; j++ {
			lastHidden[j] = hidden.At(seqLen-1, j)
		}

		// Calcular logits para próximo token
		// Usar token_embedding como peso de output (técnica common em language models)
		logits := computeLogits(model, lastHidden)

		// Aplicar softmax com temperatura
		probs := softmaxWithTemperature(logits, temperature)

		// Sample token
		nextToken := sampleToken(probs)

		// Adicionar aos tokens
		tokens = append(tokens, nextToken)

		// Verificar token de fim
		if nextToken == 3 { // <EOS>
			break
		}
	}

	// Detokenizar
	return simpleDetokenize(tokens)
}

// forward realiza forward pass
func forward(model *TrainedModel, tokens []int) *mat.Dense {
	config := model.Config
	weights := model.Weights

	seqLen := len(tokens)
	if seqLen > config.MaxSeqLen {
		seqLen = config.MaxSeqLen
		tokens = tokens[:seqLen]
	}

	// Input embeddings + positional encoding
	X := mat.NewDense(seqLen, config.DModel, nil)
	for i, tokenID := range tokens {
		if tokenID >= config.VocabSize {
			tokenID = 1 // <UNK>
		}

		// Token embedding
		tokenEmb := getMatrixRow(weights, "token_embedding.weight", tokenID, config.DModel)
		// Position embedding
		posEmb := getMatrixRow(weights, "position_embedding.weight", i, config.DModel)

		for j := 0; j < config.DModel; j++ {
			X.Set(i, j, tokenEmb[j]+posEmb[j])
		}
	}

	// Transformer layers
	for i := 0; i < config.NLayers; i++ {
		X = transformerLayerForward(model, X, i, seqLen, config.DModel, config.NHeads, config.FFHidden)
	}

	return X
}

// transformerLayerForward forward pass de uma camada
func transformerLayerForward(model *TrainedModel, X *mat.Dense, layerIdx, seqLen, dModel, nHeads, ffHidden int) *mat.Dense {
	weights := model.Weights
	prefix := fmt.Sprintf("transformer.layers.%d.", layerIdx)

	// Guardar residual
	XResidual := mat.NewDense(seqLen, dModel, nil)
	XResidual.CloneFrom(X)

	// Multi-head self-attention
	X = multiHeadAttentionPyTorch(weights, X, prefix, seqLen, dModel, nHeads)

	// Add residual
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			X.Set(i, j, X.At(i, j)+XResidual.At(i, j))
		}
	}

	// Layer norm 1
	X = layerNorm(X, seqLen, dModel, weights, prefix+"norm1.")

	// Guardar residual para FFN
	XResidual2 := mat.NewDense(seqLen, dModel, nil)
	XResidual2.CloneFrom(X)

	// Feed-forward
	X = feedForwardPyTorch(weights, X, prefix, seqLen, dModel, ffHidden)

	// Add residual
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			X.Set(i, j, X.At(i, j)+XResidual2.At(i, j))
		}
	}

	// Layer norm 2
	X = layerNorm(X, seqLen, dModel, weights, prefix+"norm2.")

	return X
}

// multiHeadAttentionPyTorch implementa attention no formato PyTorch
func multiHeadAttentionPyTorch(weights map[string]interface{}, X *mat.Dense, prefix string, seqLen, dModel, nHeads int) *mat.Dense {
	headDim := dModel / nHeads

	// PyTorch in_proj_weight: [3*d_model, d_model] - Q, K, V concatenados
	inProjWeight := getMatrix2D(weights, prefix+"self_attn.in_proj_weight", dModel*3, dModel)
	inProjBias := getVector1D(weights, prefix+"self_attn.in_proj_bias", dModel*3)

	// Calcular Q, K, V = X * in_proj_weight^T + bias
	// Separar Q, K, V do resultado
	QKV := mat.NewDense(seqLen, dModel*3, nil)
	QKV.Mul(X, inProjWeight.T())

	// Adicionar bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel*3; j++ {
			QKV.Set(i, j, QKV.At(i, j)+inProjBias[j])
		}
	}

	// Extrair Q, K, V
	Q := mat.NewDense(seqLen, dModel, nil)
	K := mat.NewDense(seqLen, dModel, nil)
	V := mat.NewDense(seqLen, dModel, nil)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			Q.Set(i, j, QKV.At(i, j))
			K.Set(i, j, QKV.At(i, j+dModel))
			V.Set(i, j, QKV.At(i, j+2*dModel))
		}
	}

	// Multi-head attention
	output := mat.NewDense(seqLen, dModel, nil)

	for h := 0; h < nHeads; h++ {
		headStart := h * headDim

		// Extrair head
		QHead := mat.NewDense(seqLen, headDim, nil)
		KHead := mat.NewDense(seqLen, headDim, nil)
		VHead := mat.NewDense(seqLen, headDim, nil)

		for i := 0; i < seqLen; i++ {
			for j := 0; j < headDim; j++ {
				QHead.Set(i, j, Q.At(i, headStart+j))
				KHead.Set(i, j, K.At(i, headStart+j))
				VHead.Set(i, j, V.At(i, headStart+j))
			}
		}

		// Scaled dot-product attention
		KTHead := KHead.T()
		scores := mat.NewDense(seqLen, seqLen, nil)
		scores.Mul(QHead, KTHead)

		scale := 1.0 / math.Sqrt(float64(headDim))
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				scores.Set(i, j, scores.At(i, j)*scale)
			}
		}

		// Softmax
		for i := 0; i < seqLen; i++ {
			row := make([]float64, seqLen)
			for j := 0; j < seqLen; j++ {
				row[j] = scores.At(i, j)
			}
			row = softmax(row)
			for j := 0; j < seqLen; j++ {
				scores.Set(i, j, row[j])
			}
		}

		// Weighted sum
		headOutput := mat.NewDense(seqLen, headDim, nil)
		headOutput.Mul(scores, VHead)

		// Colocar no output
		for i := 0; i < seqLen; i++ {
			for j := 0; j < headDim; j++ {
				output.Set(i, headStart+j, headOutput.At(i, j))
			}
		}
	}

	// Output projection
	outProjWeight := getMatrix2D(weights, prefix+"self_attn.out_proj.weight", dModel, dModel)
	outProjBias := getVector1D(weights, prefix+"self_attn.out_proj.bias", dModel)

	result := mat.NewDense(seqLen, dModel, nil)
	result.Mul(output, outProjWeight.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			result.Set(i, j, result.At(i, j)+outProjBias[j])
		}
	}

	return result
}

// feedForwardPyTorch implementa FFN no formato PyTorch
func feedForwardPyTorch(weights map[string]interface{}, X *mat.Dense, prefix string, seqLen, dModel, ffHidden int) *mat.Dense {
	// Linear 1
	w1 := getMatrix2D(weights, prefix+"linear1.weight", ffHidden, dModel)
	b1 := getVector1D(weights, prefix+"linear1.bias", ffHidden)

	hidden := mat.NewDense(seqLen, ffHidden, nil)
	hidden.Mul(X, w1.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < ffHidden; j++ {
			hidden.Set(i, j, hidden.At(i, j)+b1[j])
		}
	}

	// ReLU
	for i := 0; i < seqLen; i++ {
		for j := 0; j < ffHidden; j++ {
			val := hidden.At(i, j)
			if val < 0 {
				hidden.Set(i, j, 0)
			}
		}
	}

	// Linear 2
	w2 := getMatrix2D(weights, prefix+"linear2.weight", dModel, ffHidden)
	b2 := getVector1D(weights, prefix+"linear2.bias", dModel)

	output := mat.NewDense(seqLen, dModel, nil)
	output.Mul(hidden, w2.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			output.Set(i, j, output.At(i, j)+b2[j])
		}
	}

	return output
}

// layerNorm aplica layer normalization
func layerNorm(X *mat.Dense, seqLen, dModel int, weights map[string]interface{}, prefix string) *mat.Dense {
	weight := getVector1D(weights, prefix+"weight", dModel)
	bias := getVector1D(weights, prefix+"bias", dModel)

	result := mat.NewDense(seqLen, dModel, nil)

	for i := 0; i < seqLen; i++ {
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

		for j := 0; j < dModel; j++ {
			norm := (X.At(i, j) - mean) / std
			result.Set(i, j, norm*weight[j]+bias[j])
		}
	}

	return result
}

// computeLogits calcula logits para próximo token
func computeLogits(model *TrainedModel, hidden []float64) []float64 {
	config := model.Config
	weights := model.Weights

	// Usar token_embedding transpose como peso de output (tied weights)
	tokenEmb := getMatrix2D(weights, "token_embedding.weight", config.VocabSize, config.DModel)

	logits := make([]float64, config.VocabSize)
	for v := 0; v < config.VocabSize; v++ {
		logit := 0.0
		for j := 0; j < config.DModel; j++ {
			logit += tokenEmb.At(v, j) * hidden[j]
		}
		logits[v] = logit
	}

	return logits
}

// Helper functions para extrair dados do JSON

func getMatrixRow(weights map[string]interface{}, key string, row, cols int) []float64 {
	data, ok := weights[key]
	if !ok {
		return make([]float64, cols)
	}

	matrix, ok := data.([]interface{})
	if !ok || row >= len(matrix) {
		return make([]float64, cols)
	}

	rowData, ok := matrix[row].([]interface{})
	if !ok {
		return make([]float64, cols)
	}

	result := make([]float64, cols)
	for j := 0; j < cols && j < len(rowData); j++ {
		if val, ok := rowData[j].(float64); ok {
			result[j] = val
		}
	}

	return result
}

func getMatrix2D(weights map[string]interface{}, key string, rows, cols int) *mat.Dense {
	data, ok := weights[key]
	if !ok {
		return mat.NewDense(rows, cols, make([]float64, rows*cols))
	}

	matrix, ok := data.([]interface{})
	if !ok {
		return mat.NewDense(rows, cols, make([]float64, rows*cols))
	}

	flatData := make([]float64, rows*cols)
	for i := 0; i < rows && i < len(matrix); i++ {
		rowData, ok := matrix[i].([]interface{})
		if !ok {
			continue
		}
		for j := 0; j < cols && j < len(rowData); j++ {
			if val, ok := rowData[j].(float64); ok {
				flatData[i*cols+j] = val
			}
		}
	}

	return mat.NewDense(rows, cols, flatData)
}

func getVector1D(weights map[string]interface{}, key string, size int) []float64 {
	data, ok := weights[key]
	if !ok {
		return make([]float64, size)
	}

	vector, ok := data.([]interface{})
	if !ok {
		return make([]float64, size)
	}

	result := make([]float64, size)
	for i := 0; i < size && i < len(vector); i++ {
		if val, ok := vector[i].(float64); ok {
			result[i] = val
		}
	}

	return result
}

// softmax calcula softmax
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

// softmaxWithTemperature aplica softmax com temperatura
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

// sampleToken sampleia um token
func sampleToken(probs []float64) int {
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

// simpleTokenize tokenização por palavras (compatível com training)
func simpleTokenize(text string, vocabSize int) []int {
	// Tokenização por espaços (igual ao treinamento Python)
	words := strings.Fields(text)
	tokens := []int{2} // <BOS>

	for _, word := range words {
		// Remover pontuação básica
		word = strings.Trim(word, ".,!?;:\"'()[]{}")
		word = strings.ToLower(word)

		if len(word) == 0 {
			continue
		}

		// Hash simples da palavra para ID
		hash := 0
		for _, c := range word {
			hash = hash*31 + int(c)
		}
		tokenID := (hash % (vocabSize - 4)) + 4 // Reservar 0-3 para tokens especiais
		tokens = append(tokens, tokenID)
	}

	tokens = append(tokens, 3) // <EOS>
	return tokens
}

// simpleDetokenize detokenização por palavras
func simpleDetokenize(tokens []int) string {
	var words []string
	for _, t := range tokens {
		// Pular tokens especiais
		if t <= 3 {
			continue
		}

		// Reverter o hash (aproximado) - na prática precisamos de um vocabulário
		// Para demonstração, usar placeholder
		words = append(words, fmt.Sprintf("[token:%d]", t))
	}
	return strings.Join(words, " ")
}
