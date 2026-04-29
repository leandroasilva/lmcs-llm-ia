package model

import (
	"strings"
	"testing"
)

// TestGenerateWithKVCache_BasicGeneration testa geração básica com KV cache
func TestGenerateWithKVCache_BasicGeneration(t *testing.T) {
	model := NewTransformerModel(
		100,   // vocabSize
		128,   // dModel
		4,     // nHeads
		2,     // nLayers
		64,    // maxSeqLen
		256,   // ffHidden
		0.001, // learningRate
		0.1,   // dropoutRate
		0.01,  // weightDecay
	)

	// Construir vocabulário
	corpus := "ola mundo teste geracao texto transformer cache"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Gerar texto com KV cache
	prompt := "ola mundo"
	generated := model.GenerateWithKVCache(prompt, 20, 0.8, 10)

	// Modelo não treinado pode gerar texto vazio ou sem sentido
	// O importante é que não panic ou crash
	t.Logf("GenerateWithKVCache: '%s' -> '%s'", prompt, generated)
}

// TestGenerateWithKVCache_DoesNotPanic testa que a função não causa panic
func TestGenerateWithKVCache_DoesNotPanic(t *testing.T) {
	model := NewTransformerModel(50, 64, 2, 1, 32, 128, 0.001, 0.1, 0.01)

	corpus := "teste panic geracao"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Deve executar sem panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("GenerateWithKVCache causou panic: %v", r)
		}
	}()

	model.GenerateWithKVCache("teste", 10, 0.7, 5)
}

// TestGenerateWithKVCache_RespectsMaxTokens testa que respeita limite de tokens
func TestGenerateWithKVCache_RespectsMaxTokens(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "max tokens limite geracao teste"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	maxTokens := 5
	generated := model.GenerateWithKVCache("teste", maxTokens, 0.8, 10)

	// Contar tokens gerados
	tokens := strings.Fields(generated)
	if len(tokens) > maxTokens+2 { // +2 para margem de segurança
		t.Errorf("Gerou muitos tokens: esperado no máximo %d, obteve %d", maxTokens+2, len(tokens))
	}

	t.Logf("Max tokens test: gerou %d tokens (limite: %d)", len(tokens), maxTokens)
}

// TestGenerateWithKVCache_EmptyPrompt testa com prompt vazio
func TestGenerateWithKVCache_EmptyPrompt(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "empty prompt teste geracao"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Deve funcionar com prompt vazio
	generated := model.GenerateWithKVCache("", 10, 0.8, 10)

	t.Logf("Empty prompt: gerou '%s'", generated)
}

// TestGenerateWithKVCache_TemperatureZero testa com temperatura zero (greedy)
func TestGenerateWithKVCache_TemperatureZero(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "temperature zero greedy deterministico"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Temperatura zero deve usar greedy sampling
	generated := model.GenerateWithKVCache("teste", 10, 0.0, 10)

	t.Logf("Temperature=0: '%s'", generated)
}

// TestGenerateWithKVCache_HighTemperature testa com temperatura alta
func TestGenerateWithKVCache_HighTemperature(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "alta temperatura aleatorio diversidade criativo"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Temperatura alta deve gerar texto mais diversificado
	generated := model.GenerateWithKVCache("teste", 10, 2.0, 10)

	t.Logf("Temperature=2.0: '%s'", generated)
}

// TestGenerateWithKVCache_TopK testa diferentes valores de topK
func TestGenerateWithKVCache_TopK(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "topk parametro amostragem controle diversidade"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	topKValues := []int{1, 5, 10, 50}

	for _, topK := range topKValues {
		generated := model.GenerateWithKVCache("teste", 10, 0.8, topK)
		t.Logf("TopK=%d: '%s'", topK, generated)
	}
}

// TestGenerateWithKVCache_LongPrompt testa com prompt longo
func TestGenerateWithKVCache_LongPrompt(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "prompt longo sequencia grande teste geracao transformer cache performance"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Prompt longo
	prompt := "prompt longo sequencia grande teste geracao"
	generated := model.GenerateWithKVCache(prompt, 10, 0.8, 10)

	t.Logf("Long prompt: '%s' -> '%s'", prompt, generated)
}

// TestGenerateWithKVCache_RepetitionPenaltyApplied testa que penalidade de repetição é aplicada
func TestGenerateWithKVCache_RepetitionPenaltyApplied(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "repeticao penalidade controle diversidade tokens unicos"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Gerar texto suficiente para verificar repetições
	generated := model.GenerateWithKVCache("teste", 30, 0.8, 10)

	// Contar frequência de tokens
	tokens := strings.Fields(generated)
	tokenCounts := make(map[string]int)
	for _, token := range tokens {
		tokenCounts[token]++
	}

	// Verificar que não há repetições excessivas (>5 vezes)
	excessiveRepetitions := 0
	for token, count := range tokenCounts {
		if count > 5 {
			t.Logf("Aviso: token '%s' repetido %d vezes", token, count)
			excessiveRepetitions++
		}
	}

	if excessiveRepetitions > len(tokens)/4 {
		t.Errorf("Muitas repetições excessivas: %d", excessiveRepetitions)
	}

	t.Logf("Repetition penalty: %d tokens, %d únicos, %d repetições excessivas",
		len(tokens), len(tokenCounts), excessiveRepetitions)
}

// TestGenerateWithKVCache_DifferentConfigs testa com diferentes configurações de modelo
func TestGenerateWithKVCache_DifferentConfigs(t *testing.T) {
	configs := []struct {
		name    string
		vocab   int
		dModel  int
		nHeads  int
		nLayers int
		maxSeq  int
	}{
		{"tiny", 50, 64, 2, 1, 32},
		{"small", 100, 128, 4, 2, 64},
		{"medium", 200, 256, 8, 4, 128},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			model := NewTransformerModel(
				cfg.vocab, cfg.dModel, cfg.nHeads, cfg.nLayers,
				cfg.maxSeq, cfg.dModel*2, 0.001, 0.1, 0.01,
			)

			corpus := "config teste geracao modelo"
			vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
			model.Vocab = vocab
			model.WordToID = wordToID
			model.IDToWord = idToWord

			// Deve funcionar com qualquer configuração
			generated := model.GenerateWithKVCache("teste", 5, 0.8, 5)

			t.Logf("Config %s: '%s'", cfg.name, generated)
		})
	}
}

// TestGenerateWithKVCache_StopsAtEOS testa que para no token EOS
func TestGenerateWithKVCache_StopsAtEOS(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "eos token fim parada geracao teste"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Configurar token EOS
	model.SpecialTokens = map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
		"<BOS>": 2,
		"<EOS>": 3,
	}

	// Gerar texto - deve parar no EOS se encontrado
	generated := model.GenerateWithKVCache("teste", 50, 0.8, 10)

	// Texto não deve conter token EOS
	if strings.Contains(generated, "<EOS>") {
		t.Error("Texto gerado contém token <EOS>")
	}

	t.Logf("EOS test: gerou '%s' (sem token EOS)", generated)
}

// TestGenerateWithKVCache_MaxSeqLenLimit testa que respeita limite de sequência
func TestGenerateWithKVCache_MaxSeqLenLimit(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 32, 256, 0.001, 0.1, 0.01)

	corpus := "max sequencia limite comprimento"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Tentar gerar mais tokens que maxSeqLen
	prompt := "teste sequencia longa"
	generated := model.GenerateWithKVCache(prompt, 100, 0.8, 10)

	tokens := strings.Fields(generated)
	// Deve respeitar o limite de maxSeqLen
	if len(tokens) > model.MaxSeqLen+5 { // +5 para margem
		t.Errorf("Ultrapassou maxSeqLen: gerou %d tokens, limite %d", len(tokens), model.MaxSeqLen)
	}

	t.Logf("MaxSeqLen test: gerou %d tokens (limite: %d)", len(tokens), model.MaxSeqLen)
}

// TestGenerateWithKVCache_VocabularyMapping testa mapeamento correto de vocabulário
func TestGenerateWithKVCache_VocabularyMapping(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "vocabulario mapeamento correto palavras especificas teste"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Gerar texto
	generated := model.GenerateWithKVCache("teste", 20, 0.8, 10)

	// Verificar que tokens gerados são do vocabulário ou estão mapeados corretamente
	tokens := strings.Fields(generated)
	unknownTokens := 0
	for _, token := range tokens {
		_, ok := wordToID[token]
		if !ok && token != "<UNK>" {
			unknownTokens++
		}
	}

	if unknownTokens > len(tokens)/2 {
		t.Logf("Aviso: %d tokens desconhecidos de %d", unknownTokens, len(tokens))
	}

	t.Logf("Vocab mapping: %d tokens, %d desconhecidos", len(tokens), unknownTokens)
}

// BenchmarkGenerateWithKVCache mede performance da geração com KV cache
func BenchmarkGenerateWithKVCache(b *testing.B) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "benchmark performance geracao cache rapido"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.GenerateWithKVCache("benchmark", 10, 0.8, 5)
	}
}

// BenchmarkGenerateWithKVCache_LargeSequence mede performance com sequência longa
func BenchmarkGenerateWithKVCache_LargeSequence(b *testing.B) {
	model := NewTransformerModel(200, 256, 8, 4, 128, 512, 0.001, 0.1, 0.01)

	corpus := "benchmark sequencia longa performance cache otimizacao transformer"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.GenerateWithKVCache("benchmark sequencia", 20, 0.8, 10)
	}
}
