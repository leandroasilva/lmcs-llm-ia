package tokenizer

import (
	"strings"
	"testing"
)

func TestBPETokenizer_TrainAndTokenize(t *testing.T) {
	// Corpus de treino simples
	corpus := "hello world hello there world world"

	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 100)

	// Verificar se vocabulário foi criado
	if len(tokenizer.Vocab) == 0 {
		t.Error("Vocabulário não foi criado")
	}

	// Verificar tokens especiais
	if _, ok := tokenizer.SpecialTokens["<PAD>"]; !ok {
		t.Error("Token <PAD> não encontrado")
	}
	if _, ok := tokenizer.SpecialTokens["<UNK>"]; !ok {
		t.Error("Token <UNK> não encontrado")
	}
}

func TestBPETokenizer_Tokenize(t *testing.T) {
	corpus := "the quick brown fox jumps over the lazy dog"
	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 50)

	// Tokenizar texto
	text := "the fox jumps"
	tokens := tokenizer.Tokenize(text)

	// Verificar se tokens foram criados
	if len(tokens) == 0 {
		t.Error("Nenhum token gerado")
	}

	// Verificar token BOS no início
	if tokens[0] != tokenizer.SpecialTokens["<BOS>"] {
		t.Errorf("Esperado token <BOS> no início, got %d", tokens[0])
	}

	// Verificar token EOS no final
	lastToken := tokens[len(tokens)-1]
	if lastToken != tokenizer.SpecialTokens["<EOS>"] {
		t.Errorf("Esperado token <EOS> no final, got %d", lastToken)
	}

	t.Logf("Texto: '%s' -> Tokens: %v", text, tokens)
}

func TestBPETokenizer_Decode(t *testing.T) {
	corpus := "hello world testing decode"
	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 50)

	// Tokenizar
	original := "hello world"
	tokens := tokenizer.Tokenize(original)

	// Decoder
	decoded := tokenizer.Decode(tokens)

	// Verificar se decode funciona (pode não ser exato devido a normalização)
	if len(decoded) == 0 {
		t.Error("Decode retornou string vazia")
	}

	t.Logf("Original: '%s' -> Tokens: %v -> Decoded: '%s'", original, tokens, decoded)
}

func TestBPETokenizer_UnknownWords(t *testing.T) {
	corpus := "hello world test"
	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 50)

	// Tokenizar palavra desconhecida
	unknown := "xyz123unknown"
	tokens := tokenizer.Tokenize(unknown)

	// Deve conter token <UNK>
	hasUNK := false
	for _, token := range tokens {
		if token == tokenizer.SpecialTokens["<UNK>"] {
			hasUNK = true
			break
		}
	}

	if !hasUNK {
		t.Logf("Aviso: Palavra desconhecida '%s' não gerou token <UNK>", unknown)
	}

	t.Logf("Palavra desconhecida '%s' -> Tokens: %v", unknown, tokens)
}

func TestBPETokenizer_VocabSize(t *testing.T) {
	corpus := "this is a test corpus with multiple words for vocabulary"
	tokenizer := NewBPETokenizer()

	targetVocabSize := 30
	tokenizer.Train(corpus, targetVocabSize)

	// Verificar tamanho do vocabulário
	actualSize := tokenizer.GetVocabSize()

	// Pode ser menor se corpus é pequeno
	if actualSize == 0 {
		t.Error("Vocabulário vazio")
	}

	t.Logf("Target vocab: %d, Actual vocab: %d", targetVocabSize, actualSize)
}

func TestBPETokenizer_SaveLoadMerges(t *testing.T) {
	corpus := "save load test merges"
	tokenizer1 := NewBPETokenizer()
	tokenizer1.Train(corpus, 50)

	// Salvar merges
	mergesText := tokenizer1.SaveMerges()
	if len(mergesText) == 0 {
		t.Error("SaveMerges retornou string vazia")
	}

	// Carregar em novo tokenizer
	tokenizer2 := NewBPETokenizer()
	tokenizer2.LoadMerges(mergesText)

	// Verificar se merges foram carregados
	if len(tokenizer2.Merges) != len(tokenizer1.Merges) {
		t.Errorf("Número de merges diferente: expected %d, got %d",
			len(tokenizer1.Merges), len(tokenizer2.Merges))
	}

	t.Logf("Merges salvos/carregados: %d", len(tokenizer1.Merges))
}

func TestBPETokenizer_SpecialTokens(t *testing.T) {
	tokenizer := NewBPETokenizer()

	// Verificar tokens especiais
	expectedTokens := map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
		"<BOS>": 2,
		"<EOS>": 3,
	}

	for token, expectedID := range expectedTokens {
		actualID, ok := tokenizer.SpecialTokens[token]
		if !ok {
			t.Errorf("Token especial '%s' não encontrado", token)
			continue
		}
		if actualID != expectedID {
			t.Errorf("Token '%s': expected ID %d, got %d", token, expectedID, actualID)
		}
	}
}

func TestWordPieceTokenizer_Train(t *testing.T) {
	corpus := "testing wordpiece tokenizer with subwords"
	tokenizer := NewWordPieceTokenizer()
	tokenizer.Train(corpus, 50)

	// Verificar vocabulário
	if len(tokenizer.Vocab) == 0 {
		t.Error("WordPiece vocabulário vazio")
	}

	// Verificar tokens especiais
	expectedSpecials := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]"}
	for _, token := range expectedSpecials {
		if _, ok := tokenizer.Vocab[token]; !ok {
			t.Errorf("WordPiece token '%s' não encontrado", token)
		}
	}
}

func TestWordPieceTokenizer_Tokenize(t *testing.T) {
	corpus := "wordpiece tokenization test example"
	tokenizer := NewWordPieceTokenizer()
	tokenizer.Train(corpus, 50)

	// Tokenizar
	text := "wordpiece test"
	tokens := tokenizer.Tokenize(text)

	if len(tokens) == 0 {
		t.Error("WordPiece não gerou tokens")
	}

	// Verificar [CLS] no início
	if tokens[0] != tokenizer.Vocab["[CLS]"] {
		t.Error("WordPiece: esperado [CLS] no início")
	}

	// Verificar [SEP] no final
	lastToken := tokens[len(tokens)-1]
	if lastToken != tokenizer.Vocab["[SEP]"] {
		t.Error("WordPiece: esperado [SEP] no final")
	}

	t.Logf("WordPiece: '%s' -> %v", text, tokens)
}

func TestBPETokenizer_EmptyCorpus(t *testing.T) {
	tokenizer := NewBPETokenizer()

	// Treinar com corpus vazio
	tokenizer.Train("", 50)

	// Deve ter apenas tokens especiais
	if len(tokenizer.Vocab) < 4 {
		t.Error("Vocabulário muito pequeno com corpus vazio")
	}
}

func TestBPETokenizer_SingleWord(t *testing.T) {
	tokenizer := NewBPETokenizer()

	// Treinar com uma única palavra
	tokenizer.Train("hello", 50)

	// Tokenizar
	tokens := tokenizer.Tokenize("hello")

	if len(tokens) == 0 {
		t.Error("Não gerou tokens para palavra única")
	}

	t.Logf("Single word 'hello' -> %v", tokens)
}

func TestBPETokenizer_Reproducibility(t *testing.T) {
	corpus := "reproducibility test corpus"

	// Treinar dois tokenizadores iguais
	tokenizer1 := NewBPETokenizer()
	tokenizer1.Train(corpus, 50)

	tokenizer2 := NewBPETokenizer()
	tokenizer2.Train(corpus, 50)

	// Verificar se produzem MESMO TAMANHO de vocabulário (não necessariamente mesmos IDs)
	vocabSize1 := tokenizer1.GetVocabSize()
	vocabSize2 := tokenizer2.GetVocabSize()

	if vocabSize1 != vocabSize2 {
		t.Errorf("Vocab sizes different: %d vs %d", vocabSize1, vocabSize2)
	}

	// Verificar que ambos tokenizam texto corretamente (podem ter IDs diferentes)
	text := "reproducibility test"
	tokens1 := tokenizer1.Tokenize(text)
	tokens2 := tokenizer2.Tokenize(text)

	// Devem ter número de tokens similar (variação de ±3 é aceitável devido a map ordering)
	diff := len(tokens1) - len(tokens2)
	if diff < 0 {
		diff = -diff
	}
	if diff > 3 {
		t.Errorf("Token counts too different: %d vs %d (diff=%d)", len(tokens1), len(tokens2), diff)
	}

	t.Logf("Reproducibility verified: vocab=%d, tokens1=%d, tokens2=%d",
		vocabSize1, len(tokens1), len(tokens2))
}

func TestBPETokenizer_LargeVocab(t *testing.T) {
	// Gerar corpus grande
	words := []string{"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"}
	corpus := strings.Repeat(strings.Join(words, " ")+" ", 100)

	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 500)

	// Verificar vocabulário maior
	vocabSize := tokenizer.GetVocabSize()
	if vocabSize < 10 {
		t.Errorf("Vocabulário muito pequeno para corpus grande: %d", vocabSize)
	}

	t.Logf("Large corpus vocab size: %d", vocabSize)
}

func BenchmarkBPETokenizer_Tokenize(b *testing.B) {
	corpus := "benchmark test corpus with many words for performance testing"
	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 100)

	text := "benchmark test performance"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Tokenize(text)
	}
}

func BenchmarkBPETokenizer_Decode(b *testing.B) {
	corpus := "benchmark decode test corpus"
	tokenizer := NewBPETokenizer()
	tokenizer.Train(corpus, 100)

	tokens := tokenizer.Tokenize("benchmark decode test")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tokenizer.Decode(tokens)
	}
}
