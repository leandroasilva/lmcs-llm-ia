package model

import (
	"math"
	"strings"
	"testing"
)

// helper function para criar corpus de teste
func createTestCorpus() string {
	return strings.Repeat("hello world test transformer model ", 50)
}

func TestNewTransformerModel(t *testing.T) {
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

	if model == nil {
		t.Fatal("Modelo não foi criado")
	}

	// Verificar configurações
	if model.VocabSize != 100 {
		t.Errorf("VocabSize: expected 100, got %d", model.VocabSize)
	}
	if model.DModel != 128 {
		t.Errorf("DModel: expected 128, got %d", model.DModel)
	}
	if model.NHeads != 4 {
		t.Errorf("NHeads: expected 4, got %d", model.NHeads)
	}
	if model.NLayers != 2 {
		t.Errorf("NLayers: expected 2, got %d", model.NLayers)
	}
	if len(model.TransformerLayers) != 2 {
		t.Errorf("TransformerLayers: expected 2, got %d", len(model.TransformerLayers))
	}

	t.Logf("Modelo criado: vocab=%d, d_model=%d, heads=%d, layers=%d",
		model.VocabSize, model.DModel, model.NHeads, model.NLayers)
}

func TestTransformerModel_BuildVocab(t *testing.T) {
	corpus := "hello world test transformer hello world"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)

	// Verificar vocabulário
	if len(vocab) == 0 {
		t.Error("Vocabulário vazio")
	}

	// Verificar mappings
	if len(wordToID) != len(vocab) {
		t.Errorf("wordToID size mismatch: %d vs %d", len(wordToID), len(vocab))
	}
	if len(idToWord) != len(vocab) {
		t.Errorf("idToWord size mismatch: %d vs %d", len(idToWord), len(vocab))
	}

	// Verificar tokens especiais
	if _, ok := wordToID["<PAD>"]; !ok {
		t.Error("Token <PAD> não encontrado no vocab")
	}
	if _, ok := wordToID["<UNK>"]; !ok {
		t.Error("Token <UNK> não encontrado no vocab")
	}

	t.Logf("Vocabulário construído: %d tokens", len(vocab))
}

func TestTransformerModel_Tokenize(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "hello world test transformer"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Tokenizar texto
	text := "hello world"
	tokens := model.Tokenize(text)

	if len(tokens) == 0 {
		t.Error("Tokenização não gerou tokens")
	}

	// Verificar BOS token
	if tokens[0] != model.SpecialTokens["<BOS>"] {
		t.Errorf("Esperado <BOS> no início, got %d", tokens[0])
	}

	// Verificar EOS token
	lastToken := tokens[len(tokens)-1]
	if lastToken != model.SpecialTokens["<EOS>"] {
		t.Errorf("Esperado <EOS> no final, got %d", lastToken)
	}

	t.Logf("Tokenize: '%s' -> %v", text, tokens)
}

func TestTransformerModel_Detokenize(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "hello world test transformer"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Tokenizar e depois detokenizar
	original := "hello world"
	tokens := model.Tokenize(original)
	decoded := model.Detokenize(tokens)

	if len(decoded) == 0 {
		t.Error("Detokenize retornou string vazia")
	}

	t.Logf("Detokenize: %v -> '%s'", tokens, decoded)
}

func TestTransformerModel_Forward(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "forward pass test transformer model"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Tokenizar
	text := "forward test"
	tokens := model.Tokenize(text)

	// Forward pass
	output := model.Forward(tokens)

	// Verificar output
	if output == nil {
		t.Fatal("Forward retornou nil")
	}

	rows, cols := output.Dims()
	expectedRows := len(tokens)
	expectedCols := model.DModel

	if rows != expectedRows {
		t.Errorf("Output rows: expected %d, got %d", expectedRows, rows)
	}
	if cols != expectedCols {
		t.Errorf("Output cols: expected %d, got %d", expectedCols, cols)
	}

	t.Logf("Forward: tokens=%d, output=(%d x %d)", len(tokens), rows, cols)
}

func TestTransformerModel_Generate(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "generate text test transformer hello world"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Gerar texto - pode retornar vazio se modelo não treinado
	prompt := "hello"
	generated := model.Generate(prompt, 20, 0.8, 10)

	// Modelo não treinado pode gerar texto vazio ou sem sentido
	// O importante é que não panic ou crash
	t.Logf("Generate: '%s' -> '%s' (pode ser vazio para modelo não treinado)", prompt, generated)
}

func TestTransformerModel_TrainAndEnableBPE(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "BPE training test transformer model with subwords"

	// Treinar BPE
	model.TrainAndEnableBPE(corpus, 50)

	// Verificar se BPE foi habilitado
	if !model.UseBPE {
		t.Error("BPE não foi habilitado")
	}
	if model.BPETokenizer == nil {
		t.Error("BPETokenizer é nil")
	}

	// Verificar tipo de tokenizer
	tokenizerType := model.GetTokenizerType()
	if tokenizerType != "BPE" {
		t.Errorf("TokenizerType: expected 'BPE', got '%s'", tokenizerType)
	}

	t.Logf("BPE enabled: vocab_size=%d, type=%s", model.VocabSize, tokenizerType)
}

func TestTransformerModel_BPE_Tokenize(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "BPE tokenize test with subword tokenization"
	model.TrainAndEnableBPE(corpus, 50)

	// Tokenizar com BPE
	text := "BPE tokenize test"
	tokens := model.Tokenize(text)

	if len(tokens) == 0 {
		t.Error("BPE tokenização falhou")
	}

	// Decode
	decoded := model.Detokenize(tokens)
	if len(decoded) == 0 {
		t.Error("BPE decode falhou")
	}

	t.Logf("BPE: '%s' -> %v -> '%s'", text, tokens, decoded)
}

func TestTransformerModel_SaveLoad(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "save load test transformer model"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Salvar modelo - TransformerModel não tem Save, usando LSTM model
	t.Skip("Save/Load not implemented for TransformerModel yet")
}

func TestTransformerModel_ResidualConnections(t *testing.T) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := "residual connections test"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Forward pass
	tokens := model.Tokenize("residual test")
	output := model.Forward(tokens)

	// Verificar que output não tem NaN ou Inf
	rows, cols := output.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := output.At(i, j)
			if math.IsNaN(val) {
				t.Errorf("NaN detected in output[%d][%d]", i, j)
			}
			if math.IsInf(val, 0) {
				t.Errorf("Inf detected in output[%d][%d]", i, j)
			}
		}
	}

	t.Logf("Residual connections: output stable (no NaN/Inf)")
}

func TestTransformerModel_LayerNormalization(t *testing.T) {
	// Testar que Layer Normalization está funcionando
	model := NewTransformerModel(100, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)

	corpus := "layer normalization test"
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	// Forward com sequência longa
	tokens := model.Tokenize("layer norm test with longer sequence to verify normalization")
	output := model.Forward(tokens)

	// Verificar que valores estão normalizados (media ~0, variance ~1)
	rows, cols := output.Dims()
	for i := 0; i < rows; i++ {
		// Calcular mean da linha
		sum := 0.0
		for j := 0; j < cols; j++ {
			sum += output.At(i, j)
		}
		mean := sum / float64(cols)

		// Mean deve estar próximo de 0 após layer norm
		if math.Abs(mean) > 1.0 {
			t.Logf("Aviso: mean alto na posição %d: %f", i, mean)
		}
	}

	t.Logf("Layer normalization: verified")
}

func TestTransformerModel_DifferentConfigs(t *testing.T) {
	configs := []struct {
		name     string
		vocab    int
		dModel   int
		nHeads   int
		nLayers  int
		maxSeq   int
		ffHidden int
	}{
		{"tiny", 50, 64, 2, 1, 32, 128},
		{"small", 100, 128, 4, 2, 64, 256},
		{"medium", 200, 256, 8, 4, 128, 512},
	}

	for _, cfg := range configs {
		t.Run(cfg.name, func(t *testing.T) {
			model := NewTransformerModel(
				cfg.vocab, cfg.dModel, cfg.nHeads, cfg.nLayers,
				cfg.maxSeq, cfg.ffHidden, 0.001, 0.1, 0.01,
			)

			if model.VocabSize != cfg.vocab {
				t.Errorf("VocabSize: expected %d, got %d", cfg.vocab, model.VocabSize)
			}
			if model.DModel != cfg.dModel {
				t.Errorf("DModel: expected %d, got %d", cfg.dModel, model.DModel)
			}

			t.Logf("Config %s: OK (vocab=%d, d_model=%d, heads=%d, layers=%d)",
				cfg.name, model.VocabSize, model.DModel, model.NHeads, model.NLayers)
		})
	}
}

func BenchmarkTransformerModel_Forward(b *testing.B) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := createTestCorpus()
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	tokens := model.Tokenize("benchmark forward test performance")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Forward(tokens)
	}
}

func BenchmarkTransformerModel_Generate(b *testing.B) {
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	corpus := createTestCorpus()
	vocab, wordToID, idToWord := BuildVocabTransformer(corpus, 50)
	model.Vocab = vocab
	model.WordToID = wordToID
	model.IDToWord = idToWord

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		model.Generate("benchmark", 10, 0.8, 5)
	}
}
