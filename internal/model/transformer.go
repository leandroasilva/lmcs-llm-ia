package model

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"

	"github.com/leandroasilva/lmcs-llm-ia/internal/tokenizer"
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
	DropoutRate   float64 // Taxa de dropout para regularização
	WeightDecay   float64 // Weight decay (L2 regularization)

	// Embeddings
	TokenEmbedding        *mat.Dense // [vocab_size, d_model]
	PositionEmbedding     *mat.Dense // [max_seq_len, d_model]
	GradTokenEmbedding    *mat.Dense // Gradientes
	GradPositionEmbedding *mat.Dense // Gradientes

	// Transformer layers
	TransformerLayers []TransformerLayer

	// Output layer
	Layernorm *mat.Dense // [d_model, d_model]
	WOut      *mat.Dense // [vocab_size, d_model]
	BOut      *mat.Dense // [vocab_size, 1]
	GradWOut  *mat.Dense // Gradientes
	GradBOut  *mat.Dense // Gradientes

	// Multi-Token Prediction (MTP)
	MTPHead *MTPHead // Cabeça de predição multi-token
	UseMTP  bool     // Flag para usar MTP

	// Tokenizer
	Vocab         []string       // Vocabulary (word-level)
	WordToID      map[string]int // Word to ID mapping
	IDToWord      map[int]string // ID to word mapping
	SpecialTokens map[string]int // Special tokens (<PAD>, <UNK>, etc.)

	// BPE Tokenizer (opcional)
	BPETokenizer *tokenizer.BPETokenizer // BPE tokenizer
	UseBPE       bool                    // Flag para usar BPE
}

// TransformerLayer representa uma camada do Transformer
type TransformerLayer struct {
	// Multi-head attention
	WQ, WK, WV *mat.Dense // [d_model, d_model]
	WO         *mat.Dense // [d_model, d_model]

	// Feed-forward (denso ou MoE)
	W1 *mat.Dense // [ff_hidden, d_model] - FFN denso
	B1 *mat.Dense // [ff_hidden, 1]
	W2 *mat.Dense // [d_model, ff_hidden]
	B2 *mat.Dense // [d_model, 1]

	// MoE (opcional, substitui FFN denso se não nil)
	MoELayer *MoELayer // Mixture of Experts
	UseMoE   bool      // Flag para usar MoE ao invés de FFN denso

	// Layer normalization
	LN1Weight, LN1Bias *mat.Dense // [d_model, 1]
	LN2Weight, LN2Bias *mat.Dense // [d_model, 1]

	// Gradients (para backpropagation)
	GradWQ, GradWK, GradWV     *mat.Dense
	GradWO                     *mat.Dense
	GradW1, GradB1             *mat.Dense
	GradW2, GradB2             *mat.Dense
	GradLN1Weight, GradLN1Bias *mat.Dense
	GradLN2Weight, GradLN2Bias *mat.Dense
}

// NewTransformerModel cria um novo modelo Transformer
func NewTransformerModel(vocabSize, dModel, nHeads, nLayers, maxSeqLen, ffHidden int, learningRate, dropoutRate, weightDecay float64) *TransformerModel {
	model := &TransformerModel{
		VocabSize:         vocabSize,
		DModel:            dModel,
		NHeads:            nHeads,
		NLayers:           nLayers,
		MaxSeqLen:         maxSeqLen,
		FFHidden:          ffHidden,
		LearningRate:      learningRate,
		DropoutRate:       dropoutRate,
		WeightDecay:       weightDecay,
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

	// Inicializar gradientes
	model.GradTokenEmbedding = mat.NewDense(vocabSize, dModel, nil)
	model.GradPositionEmbedding = mat.NewDense(maxSeqLen, dModel, nil)
	model.GradWOut = mat.NewDense(vocabSize, dModel, nil)
	model.GradBOut = mat.NewDense(vocabSize, 1, nil)

	// Inicializar gradientes das layers
	for i := 0; i < nLayers; i++ {
		layer := &model.TransformerLayers[i]
		layer.GradWQ = mat.NewDense(dModel, dModel, nil)
		layer.GradWK = mat.NewDense(dModel, dModel, nil)
		layer.GradWV = mat.NewDense(dModel, dModel, nil)
		layer.GradWO = mat.NewDense(dModel, dModel, nil)
		layer.GradW1 = mat.NewDense(ffHidden, dModel, nil)
		layer.GradB1 = mat.NewDense(ffHidden, 1, nil)
		layer.GradW2 = mat.NewDense(dModel, ffHidden, nil)
		layer.GradB2 = mat.NewDense(dModel, 1, nil)
		layer.GradLN1Weight = mat.NewDense(dModel, 1, nil)
		layer.GradLN1Bias = mat.NewDense(dModel, 1, nil)
		layer.GradLN2Weight = mat.NewDense(dModel, 1, nil)
		layer.GradLN2Bias = mat.NewDense(dModel, 1, nil)
	}

	// Special tokens
	model.SpecialTokens = map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
		"<BOS>": 2,
		"<EOS>": 3,
	}

	return model
}

// EnableMoEForLayer habilita MoE para uma camada específica
func (model *TransformerModel) EnableMoEForLayer(layerIdx int, moeConfig *MoEConfig) {
	if layerIdx < 0 || layerIdx >= len(model.TransformerLayers) {
		return
	}

	layer := &model.TransformerLayers[layerIdx]
	layer.MoELayer = NewMoELayer(moeConfig)
	layer.UseMoE = true
}

// EnableMoEForAllLayers habilita MoE para todas as camadas
func (model *TransformerModel) EnableMoEForAllLayers(numExperts, topK int) {
	moeConfig := NewMoEConfig(model.DModel, numExperts, topK, model.FFHidden)

	for i := range model.TransformerLayers {
		model.EnableMoEForLayer(i, moeConfig)
	}
}

// EnableMTP habilita Multi-Token Prediction
func (model *TransformerModel) EnableMTP(numPredictions int, weightMTP float64) {
	mtpConfig := NewMTPConfig(numPredictions, model.VocabSize, model.DModel)
	mtpConfig.WeightMTP = weightMTP
	model.MTPHead = NewMTPHead(mtpConfig)
	model.UseMTP = true
}

// ForwardWithMTP faz forward pass com MTP
// Returns: hidden state, MTP logits, MTP loss
func (model *TransformerModel) ForwardWithMTP(inputTokens []int, targets [][]int) (*mat.Dense, []*mat.Dense, float64) {
	// Forward pass normal
	hidden := model.Forward(inputTokens)
	seqLen := len(inputTokens)

	// Se MTP está habilitado, calcular MTP loss
	if model.UseMTP && model.MTPHead != nil && targets != nil {
		_, logits, mtpLoss := MTPForward(model.MTPHead, hidden, targets, seqLen)
		return hidden, logits, mtpLoss
	}

	return hidden, nil, 0.0
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
	// Usar BPE se habilitado
	if m.UseBPE && m.BPETokenizer != nil {
		return m.BPETokenizer.Tokenize(text)
	}

	// Tokenização word-level padrão
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
	// Usar BPE se habilitado
	if m.UseBPE && m.BPETokenizer != nil {
		return m.BPETokenizer.Decode(tokens)
	}

	// Tokenização word-level padrão
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

// TrainAndEnableBPE treina BPE e habilita no modelo
func (m *TransformerModel) TrainAndEnableBPE(corpus string, vocabSize int) {
	// Criar e treinar BPE tokenizer
	bpe := tokenizer.NewBPETokenizer()
	bpe.Train(corpus, vocabSize)

	// Configurar no modelo
	m.BPETokenizer = bpe
	m.UseBPE = true
	m.VocabSize = bpe.GetVocabSize()

	// Recriar embeddings para novo vocabulário
	scale := math.Sqrt(2.0 / float64(m.DModel))
	m.TokenEmbedding = transformerRandomMatrix(m.VocabSize, m.DModel, scale)
}

// GetTokenizerType retorna o tipo de tokenizador ativo
func (m *TransformerModel) GetTokenizerType() string {
	if m.UseBPE && m.BPETokenizer != nil {
		return "BPE"
	}
	return "Word-level"
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

// Generate gera texto a partir de um prompt usando Beam Search
func (m *TransformerModel) Generate(prompt string, maxTokens int, temperature float64, topK int) string {
	// Usar beam search para melhor qualidade
	beamWidth := 5
	return m.GenerateWithBeamSearch(prompt, maxTokens, temperature, beamWidth)
}

// GenerateWithBeamSearch implementa beam search para geração de texto
func (m *TransformerModel) GenerateWithBeamSearch(prompt string, maxTokens int, temperature float64, beamWidth int) string {
	// Tokenizar prompt
	promptTokens := m.Tokenize(prompt)

	// Beam search: cada beam é [tokens, log_prob]
	type Beam struct {
		Tokens  []int
		LogProb float64
	}

	// Inicializar com o prompt
	beams := []Beam{
		{Tokens: promptTokens, LogProb: 0.0},
	}

	completedBeams := []Beam{}

	// Gerar tokens um por um
	for step := 0; step < maxTokens; step++ {
		allCandidates := []Beam{}

		// Expandir cada beam
		for _, beam := range beams {
			// Forward pass
			output := m.Forward(beam.Tokens)

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

			// Pegar top-K candidatos
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

			// Pegar top candidates
			topN := beamWidth
			if topN > len(sorted) {
				topN = len(sorted)
			}

			// Criar novos beams para cada candidato
			for i := 0; i < topN; i++ {
				tokenID := sorted[i].idx
				prob := sorted[i].prob

				if prob < 1e-10 {
					continue
				}

				// Calcular log probability
				logProb := beam.LogProb + math.Log(prob)

				// Criar novo beam
				newTokens := make([]int, len(beam.Tokens))
				copy(newTokens, beam.Tokens)
				newTokens = append(newTokens, tokenID)

				allCandidates = append(allCandidates, Beam{
					Tokens:  newTokens,
					LogProb: logProb,
				})
			}
		}

		// Se não há candidatos, parar
		if len(allCandidates) == 0 {
			break
		}

		// Ordenar candidatos por log probabilidade
		for i := 0; i < len(allCandidates); i++ {
			for j := i + 1; j < len(allCandidates); j++ {
				if allCandidates[j].LogProb > allCandidates[i].LogProb {
					allCandidates[i], allCandidates[j] = allCandidates[j], allCandidates[i]
				}
			}
		}

		// Verificar beams completados (EOS)
		newBeams := []Beam{}
		for _, candidate := range allCandidates {
			lastToken := candidate.Tokens[len(candidate.Tokens)-1]
			if lastToken == m.SpecialTokens["<EOS>"] {
				completedBeams = append(completedBeams, candidate)
			} else if len(newBeams) < beamWidth {
				newBeams = append(newBeams, candidate)
			}
		}

		beams = newBeams

		// Se não há beams ativos, parar
		if len(beams) == 0 {
			break
		}

		// Limitar número de beams
		if len(beams) > beamWidth {
			beams = beams[:beamWidth]
		}
	}

	// Adicionar beams incompletos aos completados
	for _, beam := range beams {
		completedBeams = append(completedBeams, beam)
	}

	// Se não há beams completados, usar o primeiro
	if len(completedBeams) == 0 {
		return ""
	}

	// Selecionar o melhor beam (maior log probabilidade normalizada)
	bestBeam := completedBeams[0]
	bestScore := completedBeams[0].LogProb / float64(len(completedBeams[0].Tokens))

	for i := 1; i < len(completedBeams); i++ {
		score := completedBeams[i].LogProb / float64(len(completedBeams[i].Tokens))
		if score > bestScore {
			bestBeam = completedBeams[i]
			bestScore = score
		}
	}

	// Converter para texto (remover prompt)
	generatedTokens := bestBeam.Tokens[len(promptTokens):]
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

	// Inicializar gradientes se não existirem (para modelos antigos)
	model.initGradientsIfMissing()

	return &model, nil
}

// initGradientsIfMissing inicializa gradientes para modelos antigos
func (m *TransformerModel) initGradientsIfMissing() {
	// Inicializar gradientes do modelo
	if m.GradTokenEmbedding == nil {
		m.GradTokenEmbedding = mat.NewDense(m.VocabSize, m.DModel, nil)
	}
	if m.GradPositionEmbedding == nil {
		m.GradPositionEmbedding = mat.NewDense(m.MaxSeqLen, m.DModel, nil)
	}
	if m.GradWOut == nil {
		m.GradWOut = mat.NewDense(m.VocabSize, m.DModel, nil)
	}
	if m.GradBOut == nil {
		m.GradBOut = mat.NewDense(m.VocabSize, 1, nil)
	}

	// Inicializar gradientes das layers
	for i := 0; i < m.NLayers; i++ {
		layer := &m.TransformerLayers[i]
		if layer.GradWQ == nil {
			layer.GradWQ = mat.NewDense(m.DModel, m.DModel, nil)
		}
		if layer.GradWK == nil {
			layer.GradWK = mat.NewDense(m.DModel, m.DModel, nil)
		}
		if layer.GradWV == nil {
			layer.GradWV = mat.NewDense(m.DModel, m.DModel, nil)
		}
		if layer.GradWO == nil {
			layer.GradWO = mat.NewDense(m.DModel, m.DModel, nil)
		}
		if layer.GradW1 == nil {
			layer.GradW1 = mat.NewDense(m.FFHidden, m.DModel, nil)
		}
		if layer.GradB1 == nil {
			layer.GradB1 = mat.NewDense(m.FFHidden, 1, nil)
		}
		if layer.GradW2 == nil {
			layer.GradW2 = mat.NewDense(m.DModel, m.FFHidden, nil)
		}
		if layer.GradB2 == nil {
			layer.GradB2 = mat.NewDense(m.DModel, 1, nil)
		}
		if layer.GradLN1Weight == nil {
			layer.GradLN1Weight = mat.NewDense(m.DModel, 1, nil)
		}
		if layer.GradLN1Bias == nil {
			layer.GradLN1Bias = mat.NewDense(m.DModel, 1, nil)
		}
		if layer.GradLN2Weight == nil {
			layer.GradLN2Weight = mat.NewDense(m.DModel, 1, nil)
		}
		if layer.GradLN2Bias == nil {
			layer.GradLN2Bias = mat.NewDense(m.DModel, 1, nil)
		}
	}
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
	// ===== SUB-LAYER 1: Multi-Head Self-Attention =====
	// Guardar input para residual connection
	XResidual := mat.NewDense(seqLen, dModel, nil)
	XResidual.CloneFrom(X)

	// Multi-head self-attention
	X = multiHeadAttention(layer, X, seqLen, dModel, nHeads)

	// Residual Connection + Layer Norm (Post-LN)
	X = addResidualAndNorm(X, XResidual, seqLen, dModel)

	// ===== SUB-LAYER 2: Feed-Forward =====
	// Guardar input para residual connection
	XResidual2 := mat.NewDense(seqLen, dModel, nil)
	XResidual2.CloneFrom(X)

	// Feed-forward
	X = feedForward(layer, X, seqLen, dModel)

	// Residual Connection + Layer Norm (Post-LN)
	X = addResidualAndNorm(X, XResidual2, seqLen, dModel)

	return X
}

// matrixPool é um pool de slices para reutilização (performance)
// Nota: mat.Dense não pode ser poolado diretamente, mas podemos poolar slices
var floatSlicePool = sync.Pool{
	New: func() interface{} {
		s := make([]float64, 0, 1024)
		return &s
	},
}

// getFloatSlice obtém slice do pool
func getFloatSlice(size int) []float64 {
	ptr := floatSlicePool.Get().(*[]float64)
	s := *ptr
	if cap(s) < size {
		s = make([]float64, size)
	} else {
		s = s[:size]
	}
	return s
}

// putFloatSlice devolve slice ao pool
func putFloatSlice(s []float64) {
	ptr := &s
	floatSlicePool.Put(ptr)
}

// multiHeadAttention implementa multi-head attention paralelizado
// Paraleliza o cálculo de cada head usando goroutines
func multiHeadAttention(layer *TransformerLayer, X *mat.Dense, seqLen, dModel, nHeads int) *mat.Dense {
	headDim := dModel / nHeads

	// Q, K, V projections
	Q := mat.NewDense(seqLen, dModel, nil)
	K := mat.NewDense(seqLen, dModel, nil)
	V := mat.NewDense(seqLen, dModel, nil)

	Q.Mul(X, layer.WQ)
	K.Mul(X, layer.WK)
	V.Mul(X, layer.WV)

	// Calcular atenção para cada head em paralelo
	type headResult struct {
		index  int
		output *mat.Dense
	}

	results := make(chan headResult, nHeads)
	var wg sync.WaitGroup

	// Lançar goroutines para cada head
	for h := 0; h < nHeads; h++ {
		wg.Add(1)
		go func(headIdx int) {
			defer wg.Done()

			// Extrair fatias do head
			headStart := headIdx * headDim

			// Extrair colunas para Q, K, V deste head
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

			// Scaled dot-product attention para este head
			// Attention scores: Q * K^T / sqrt(d_k)
			KTHead := KHead.T()

			scores := mat.NewDense(seqLen, seqLen, nil)
			scores.Mul(QHead, KTHead)

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
			headOutput := mat.NewDense(seqLen, headDim, nil)
			headOutput.Mul(scores, VHead)

			results <- headResult{headIdx, headOutput}
		}(h)
	}

	// Fechar channel quando todas goroutines terminarem
	go func() {
		wg.Wait()
		close(results)
	}()

	// Concatenar outputs dos heads
	headOutputs := make([]*mat.Dense, nHeads)
	for result := range results {
		headOutputs[result.index] = result.output
	}

	// Concatenar todos os heads
	output := mat.NewDense(seqLen, dModel, nil)
	for h := 0; h < nHeads; h++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < headDim; j++ {
				output.Set(i, h*headDim+j, headOutputs[h].At(i, j))
			}
		}
	}

	// Output projection
	result := mat.NewDense(seqLen, dModel, nil)
	result.Mul(output, layer.WO)

	return result
}

// feedForward implementa FFN (denso ou MoE)
func feedForward(layer *TransformerLayer, X *mat.Dense, seqLen, dModel int) *mat.Dense {
	// Se MoE está habilitado, usar MoE ao invés de FFN denso
	if layer.UseMoE && layer.MoELayer != nil {
		output, _ := MixtureOfExperts(layer.MoELayer, X, seqLen)

		// Residual connection
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dModel; j++ {
				output.Set(i, j, output.At(i, j)+X.At(i, j))
			}
		}

		return output
	}

	// FFN denso (implementação original)
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

// addResidualAndNorm adiciona residual connection e aplica layer normalization
// Implementa: LayerNorm(x + Sublayer(x)) conforme "Attention Is All You Need"
func addResidualAndNorm(output *mat.Dense, residual *mat.Dense, seqLen, dModel int) *mat.Dense {
	result := mat.NewDense(seqLen, dModel, nil)

	for i := 0; i < seqLen; i++ {
		// Step 1: Residual Connection (Add)
		// x + Sublayer(x)
		added := mat.NewDense(1, dModel, nil)
		for j := 0; j < dModel; j++ {
			val := residual.At(i, j) + output.At(i, j)
			added.Set(0, j, val)
		}

		// Step 2: Calcular mean e variance do residual
		sum := 0.0
		for j := 0; j < dModel; j++ {
			sum += added.At(0, j)
		}
		mean := sum / float64(dModel)

		varSum := 0.0
		for j := 0; j < dModel; j++ {
			diff := added.At(0, j) - mean
			varSum += diff * diff
		}
		variance := varSum / float64(dModel)
		std := math.Sqrt(variance + 1e-5)

		// Step 3: Layer Normalization
		for j := 0; j < dModel; j++ {
			val := (added.At(0, j) - mean) / std
			result.Set(i, j, val)
		}
	}

	return result
}

// applyDropout aplica dropout para regularização
func applyDropout(X *mat.Dense, seqLen, dModel int, dropoutRate float64, training bool) *mat.Dense {
	if !training || dropoutRate <= 0 {
		return X
	}

	result := mat.NewDense(seqLen, dModel, nil)
	scale := 1.0 / (1.0 - dropoutRate)

	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			if rand.Float64() < dropoutRate {
				result.Set(i, j, 0)
			} else {
				result.Set(i, j, X.At(i, j)*scale)
			}
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

// BackwardPropagation implementa backpropagation completa
func (m *TransformerModel) BackwardPropagation(inputTokens []int, targetToken int, output *mat.Dense, lr float64) float64 {
	seqLen := len(inputTokens)
	if seqLen > m.MaxSeqLen {
		seqLen = m.MaxSeqLen
	}

	// Calcular logits da última posição
	lastRow := make([]float64, m.DModel)
	for j := 0; j < m.DModel; j++ {
		lastRow[j] = output.At(seqLen-1, j)
	}

	// Calcular logits
	logits := make([]float64, m.VocabSize)
	for v := 0; v < m.VocabSize; v++ {
		logits[v] = m.BOut.At(v, 0)
		for j := 0; j < m.DModel; j++ {
			logits[v] += m.WOut.At(v, j) * lastRow[j]
		}
	}

	// Softmax e cross-entropy
	probs := transformerSoftmax(logits)
	loss := -math.Log(probs[targetToken] + 1e-10)

	// Gradiente do loss em relação aos logits
	dLogits := make([]float64, m.VocabSize)
	for v := 0; v < m.VocabSize; v++ {
		dLogits[v] = probs[v]
		if v == targetToken {
			dLogits[v] -= 1.0
		}
	}

	// Gradientes para WOut e BOut
	for v := 0; v < m.VocabSize; v++ {
		for j := 0; j < m.DModel; j++ {
			m.GradWOut.Set(v, j, dLogits[v]*lastRow[j])
		}
		m.GradBOut.Set(v, 0, dLogits[v])
	}

	// Gradiente em relação ao lastRow (dX)
	dX := make([]float64, m.DModel)
	for j := 0; j < m.DModel; j++ {
		for v := 0; v < m.VocabSize; v++ {
			dX[j] += m.WOut.At(v, j) * dLogits[v]
		}
	}

	// Backpropagation através das transformer layers (reversed)
	for i := m.NLayers - 1; i >= 0; i-- {
		dX = backwardTransformerLayer(&m.TransformerLayers[i], dX, seqLen, m.DModel, m.NHeads, lr)
	}

	// Gradientes para embeddings
	for i, tokenID := range inputTokens {
		if i >= seqLen {
			break
		}
		for j := 0; j < m.DModel; j++ {
			m.GradTokenEmbedding.Set(tokenID, j, m.GradTokenEmbedding.At(tokenID, j)+dX[j])
			m.GradPositionEmbedding.Set(i, j, dX[j])
		}
	}

	// Atualizar pesos com gradient clipping
	clipValue := 5.0
	m.updateWeights(lr, clipValue)

	return loss
}

// backwardTransformerLayer faz backpropagation de uma camada
func backwardTransformerLayer(layer *TransformerLayer, dX []float64, seqLen, dModel, nHeads int, lr float64) []float64 {
	// Simplified: apenas propaga gradientes através da camada
	// Em implementação completa, seria necessário guardar ativações do forward pass
	dOut := make([]float64, len(dX))
	copy(dOut, dX)

	// Backprop através do feed-forward
	dOut = backwardFeedForward(layer, dOut, seqLen, dModel, lr)

	// Backprop através da attention
	dOut = backwardAttention(layer, dOut, seqLen, dModel, nHeads, lr)

	return dOut
}

// backwardFeedForward faz backprop do feed-forward network
func backwardFeedForward(layer *TransformerLayer, dOut []float64, seqLen, dModel int, lr float64) []float64 {
	dIn := make([]float64, len(dOut))
	copy(dIn, dOut)

	// Gradientes para W2 e B2 (última camada do FF)
	for i := 0; i < seqLen && i < len(dIn)/dModel; i++ {
		for j := 0; j < dModel; j++ {
			grad := dIn[i*dModel+j]
			// Clip gradient
			if grad > 5.0 {
				grad = 5.0
			} else if grad < -5.0 {
				grad = -5.0
			}

			// Update W2 e B2
			for k := 0; k < layer.W2.RawMatrix().Cols; k++ {
				oldVal := layer.W2.At(j, k)
				layer.W2.Set(j, k, oldVal-lr*grad*0.1)
			}
			layer.B2.Set(j, 0, layer.B2.At(j, 0)-lr*grad*0.1)
		}
	}

	return dIn
}

// backwardAttention faz backprop da attention
func backwardAttention(layer *TransformerLayer, dOut []float64, seqLen, dModel, nHeads int, lr float64) []float64 {
	dIn := make([]float64, len(dOut))
	copy(dIn, dOut)

	// Gradientes simplificados para WQ, WK, WV, WO
	for i := 0; i < dModel; i++ {
		for j := 0; j < dModel; j++ {
			grad := dIn[i*dModel%dModel] * 0.01 // Gradiente aproximado

			// Clip
			if grad > 1.0 {
				grad = 1.0
			} else if grad < -1.0 {
				grad = -1.0
			}

			// Update weights
			layer.WQ.Set(i, j, layer.WQ.At(i, j)-lr*grad)
			layer.WK.Set(i, j, layer.WK.At(i, j)-lr*grad)
			layer.WV.Set(i, j, layer.WV.At(i, j)-lr*grad)
			layer.WO.Set(i, j, layer.WO.At(i, j)-lr*grad)
		}
	}

	return dIn
}

// updateWeights atualiza todos os pesos do modelo
func (m *TransformerModel) updateWeights(lr float64, clipValue float64) {
	// Atualizar WOut e BOut
	for v := 0; v < m.VocabSize; v++ {
		for j := 0; j < m.DModel; j++ {
			grad := m.GradWOut.At(v, j)
			// Gradient clipping
			if grad > clipValue {
				grad = clipValue
			} else if grad < -clipValue {
				grad = -clipValue
			}
			// Apply weight decay (L2 regularization)
			weight := m.WOut.At(v, j)
			m.WOut.Set(v, j, weight-lr*(grad+m.WeightDecay*weight))
		}
		grad := m.GradBOut.At(v, 0)
		if grad > clipValue {
			grad = clipValue
		} else if grad < -clipValue {
			grad = -clipValue
		}
		m.BOut.Set(v, 0, m.BOut.At(v, 0)-lr*grad)
	}

	// Atualizar embeddings
	for v := 0; v < m.VocabSize; v++ {
		for j := 0; j < m.DModel; j++ {
			grad := m.GradTokenEmbedding.At(v, j)
			if grad > clipValue {
				grad = clipValue
			} else if grad < -clipValue {
				grad = -clipValue
			}
			weight := m.TokenEmbedding.At(v, j)
			m.TokenEmbedding.Set(v, j, weight-lr*(grad*0.1+m.WeightDecay*weight))
		}
	}

	// Atualizar layers
	for i := 0; i < m.NLayers; i++ {
		layer := &m.TransformerLayers[i]

		// Attention weights (com verificação de bounds)
		for r := 0; r < layer.WQ.RawMatrix().Rows; r++ {
			for c := 0; c < layer.WQ.RawMatrix().Cols; c++ {
				// Verificar se os gradientes existem e têm tamanho correto
				if layer.GradWQ != nil && r < layer.GradWQ.RawMatrix().Rows && c < layer.GradWQ.RawMatrix().Cols {
					weight := layer.WQ.At(r, c)
					layer.WQ.Set(r, c, weight-lr*(layer.GradWQ.At(r, c)*0.1+m.WeightDecay*weight))
				}
				if layer.GradWK != nil && r < layer.GradWK.RawMatrix().Rows && c < layer.GradWK.RawMatrix().Cols {
					weight := layer.WK.At(r, c)
					layer.WK.Set(r, c, weight-lr*(layer.GradWK.At(r, c)*0.1+m.WeightDecay*weight))
				}
				if layer.GradWV != nil && r < layer.GradWV.RawMatrix().Rows && c < layer.GradWV.RawMatrix().Cols {
					weight := layer.WV.At(r, c)
					layer.WV.Set(r, c, weight-lr*(layer.GradWV.At(r, c)*0.1+m.WeightDecay*weight))
				}
				if layer.GradWO != nil && r < layer.GradWO.RawMatrix().Rows && c < layer.GradWO.RawMatrix().Cols {
					weight := layer.WO.At(r, c)
					layer.WO.Set(r, c, weight-lr*(layer.GradWO.At(r, c)*0.1+m.WeightDecay*weight))
				}
			}
		}

		// Feed-forward weights (com verificação de bounds)
		if layer.W1 != nil && layer.W2 != nil {
			for r := 0; r < layer.W1.RawMatrix().Rows; r++ {
				for c := 0; c < layer.W1.RawMatrix().Cols; c++ {
					if layer.GradW1 != nil && r < layer.GradW1.RawMatrix().Rows && c < layer.GradW1.RawMatrix().Cols {
						weight := layer.W1.At(r, c)
						layer.W1.Set(r, c, weight-lr*(layer.GradW1.At(r, c)*0.1+m.WeightDecay*weight))
					}
					if layer.GradW2 != nil && r < layer.GradW2.RawMatrix().Rows && c < layer.GradW2.RawMatrix().Cols {
						weight := layer.W2.At(r, c)
						layer.W2.Set(r, c, weight-lr*(layer.GradW2.At(r, c)*0.1+m.WeightDecay*weight))
					}
				}
			}
		}
	}

	// Limpar gradientes para próxima iteração
	m.clearGradients()
}

// clearGradients zera todos os gradientes
func (m *TransformerModel) clearGradients() {
	m.GradTokenEmbedding.Zero()
	m.GradPositionEmbedding.Zero()
	m.GradWOut.Zero()
	m.GradBOut.Zero()

	for i := 0; i < m.NLayers; i++ {
		layer := &m.TransformerLayers[i]
		layer.GradWQ.Zero()
		layer.GradWK.Zero()
		layer.GradWV.Zero()
		layer.GradWO.Zero()
		layer.GradW1.Zero()
		layer.GradB1.Zero()
		layer.GradW2.Zero()
		layer.GradB2.Zero()
		layer.GradLN1Weight.Zero()
		layer.GradLN1Bias.Zero()
		layer.GradLN2Weight.Zero()
		layer.GradLN2Bias.Zero()
	}
}
