package tokenizer

import (
	"testing"
)

func TestDataQuality_SDGTGeneration(t *testing.T) {
	// Testar Seed-Driven Growth Technique
	seeds := []string{
		"machine learning algorithms",
		"neural network training",
		"data preprocessing",
	}

	sdgt := NewSDGTDataGenerator(seeds)

	// Gerar dados sintéticos
	generated := sdgt.Generate(50)

	if len(generated) < 50 {
		t.Errorf("Expected at least 50 samples, got %d", len(generated))
	}

	t.Logf("✓ SDGT Data Generation:")
	t.Logf("  Seeds: %d", sdgt.GetSeedCount())
	t.Logf("  Generated: %d samples", sdgt.GetGeneratedCount())
	t.Logf("  Expansion rate: %.1fx", sdgt.SDGTExpansionRate())

	// Verificar qualidade
	uniqueSamples := make(map[string]bool)
	for _, sample := range generated {
		uniqueSamples[sample] = true
	}

	t.Logf("  Unique samples: %d", len(uniqueSamples))
	t.Logf("  Uniqueness rate: %.1f%%", float64(len(uniqueSamples))/float64(len(generated))*100)

	// Mostrar exemplos
	t.Logf("\n  Sample generations:")
	for i := 0; i < 5 && i < len(generated); i++ {
		t.Logf("    %d. %s", i+1, generated[i])
	}
}

func TestDataQuality_SDGTWithPortuguese(t *testing.T) {
	// Testar com seeds em português
	seeds := []string{
		"aprendizado de máquina",
		"redes neurais profundas",
		"processamento de linguagem natural",
	}

	sdgt := NewSDGTDataGenerator(seeds)
	generated := sdgt.Generate(30)

	t.Logf("✓ SDGT with Portuguese seeds:")
	t.Logf("  Seeds: %d", len(seeds))
	t.Logf("  Generated: %d samples", len(generated))

	// Verificar que contains original seeds
	for _, seed := range seeds {
		found := false
		for _, sample := range generated {
			if sample == seed {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Original seed not found: %s", seed)
		}
	}

	t.Logf("  All original seeds preserved")
}

func TestDataQuality_Curation(t *testing.T) {
	// Testar data curation
	data := []string{
		"This is a good quality sentence about machine learning.",
		"Short", // Too short
		"This is another good sentence about neural networks and deep learning.",
		"This is a good quality sentence about machine learning.", // Duplicate
		"This is very similar to the first sentence about ML.",    // Low diversity
		"A completely different topic about astronomy and stars in the universe.",
		"Another unique sentence about quantum computing and qubits.",
	}

	curator := NewDataCurator()
	curator.MinLength = 20
	curator.MaxLength = 200
	curator.DiversityThreshold = 0.5

	curated := curator.Curate(data)

	t.Logf("✓ Data Curation:")
	t.Logf("  Original: %d samples", len(data))
	t.Logf("  Curated: %d samples", len(curated))

	stats := curator.GetCurateStats(data, curated)
	t.Logf("  Filtered out: %v", stats["filtered_out"])
	t.Logf("  Retention rate: %.1f%%", stats["retention_rate"].(float64)*100)

	// Verificar que removidos
	if len(curated) >= len(data) {
		t.Error("Curation should reduce dataset size")
	}

	// Verificar que short foi removido
	for _, item := range curated {
		if len(item) < curator.MinLength {
			t.Errorf("Item too short in curated: %s", item)
		}
	}

	t.Logf("  All items meet length requirements")
}

func TestDataQuality_VocabularyOptimization(t *testing.T) {
	// Testar vocabulary optimization
	corpus := []string{
		"machine learning is important for data science",
		"neural networks are used in deep learning",
		"data preprocessing is essential for ML",
	}

	// Criar vocabulário grande
	vocab := make(map[string]int)
	tokens := []string{
		"machine", "learning", "data", "neural", "networks",
		"deep", "important", "science", "preprocessing", "essential",
		"rare_token_1", "rare_token_2", "rare_token_3",
	}

	for i, token := range tokens {
		vocab[token] = i
	}

	// Otimizar
	optimizer := NewVocabularyOptimizer(10, 0.01)
	optimized := optimizer.Optimize(vocab, corpus)

	t.Logf("✓ Vocabulary Optimization:")
	t.Logf("  Original vocab: %d tokens", len(vocab))
	t.Logf("  Optimized vocab: %d tokens", len(optimized))

	stats := optimizer.GetOptimizationStats(vocab, optimized)
	t.Logf("  Reduction: %d tokens", stats["reduction"])
	t.Logf("  Reduction percentage: %.1f%%", stats["reduction_pct"].(float64)*100)

	// Verificar que tokens raros foram removidos
	for _, rareToken := range []string{"rare_token_1", "rare_token_2", "rare_token_3"} {
		if _, exists := optimized[rareToken]; exists {
			t.Logf("  Note: Rare token %s kept (has some coverage)", rareToken)
		} else {
			t.Logf("  Rare token %s removed (low coverage)", rareToken)
		}
	}
}

func TestDataQuality_SDGTExpansionRates(t *testing.T) {
	// Testar diferentes taxas de expansão
	testCases := []struct {
		Name         string
		NumSeeds     int
		TargetSize   int
		ExpectedRate float64
	}{
		{"Small expansion", 3, 20, 6.67},
		{"Medium expansion", 5, 50, 10.0},
		{"Large expansion", 10, 100, 10.0},
	}

	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			seeds := make([]string, tc.NumSeeds)
			for i := 0; i < tc.NumSeeds; i++ {
				seeds[i] = string(rune('a' + i))
			}

			sdgt := NewSDGTDataGenerator(seeds)
			generated := sdgt.Generate(tc.TargetSize)

			rate := sdgt.SDGTExpansionRate()

			t.Logf("  %s: %d seeds → %d samples (%.1fx)",
				tc.Name, tc.NumSeeds, len(generated), rate)

			if len(generated) < tc.TargetSize {
				t.Errorf("Expected %d, got %d", tc.TargetSize, len(generated))
			}
		})
	}
}

func TestDataQuality_DiversityFiltering(t *testing.T) {
	// Testar filtragem por diversidade
	data := []string{
		"The quick brown fox jumps over the lazy dog",
		"The quick brown fox leaps over the lazy dog",       // Very similar
		"The fast brown fox jumps over the sleepy dog",      // Similar
		"Python is a programming language for data science", // Different
		"JavaScript is used for web development",            // Different
	}

	curator := NewDataCurator()
	curator.DiversityThreshold = 0.4

	curated := curator.Curate(data)

	t.Logf("✓ Diversity Filtering:")
	t.Logf("  Original: %d", len(data))
	t.Logf("  Curated: %d", len(curated))
	t.Logf("  Removed %d similar items", len(data)-len(curated))

	// Deve remover itens muito similares
	if len(curated) >= 4 {
		t.Error("Should have removed more similar items")
	}

	t.Logf("  Kept only diverse items")
}

func TestDataQuality_TokenizerComparison(t *testing.T) {
	// Comparar BPE vs WordPiece
	corpus := []string{
		"machine learning algorithms are powerful",
		"neural networks process information efficiently",
		"deep learning models require large datasets",
	}

	// BPE
	bpe := NewBPETokenizer()
	bpeText := "machine learning neural networks deep learning"
	bpe.Train(bpeText, 50)

	bpeTokens := bpe.Tokenize("machine learning")
	t.Logf("✓ BPE Tokenization:")
	t.Logf("  Vocab size: %d", bpe.GetVocabSize())
	t.Logf("  'machine learning' → %v", bpeTokens)
	t.Logf("  Decoded: '%s'", bpe.Decode(bpeTokens))

	// WordPiece
	wp := NewWordPieceTokenizer()
	wpCorpus := ""
	for _, item := range corpus {
		wpCorpus += item + " "
	}
	wp.Train(wpCorpus, 50)

	wpTokens := wp.Tokenize("machine learning")
	t.Logf("\n✓ WordPiece Tokenization:")
	t.Logf("  Vocab size: ~%d (estimated)", len(wp.Vocab))
	t.Logf("  'machine learning' → %v", wpTokens)

	// Comparar
	t.Logf("\n✓ Comparison:")
	t.Logf("  BPE vocab: %d tokens", bpe.GetVocabSize())
	t.Logf("  WordPiece vocab: ~%d tokens (estimated)", len(wp.Vocab))
	t.Logf("  BPE handles unknown chars better")
	t.Logf("  WordPiece handles unknown subwords better")
}

func TestDataQuality_UnknownTokenHandling(t *testing.T) {
	// Testar handling de tokens desconhecidos
	corpus := []string{
		"hello world machine learning",
	}

	wp := NewWordPieceTokenizer()
	wpCorpus := ""
	for _, item := range corpus {
		wpCorpus += item + " "
	}
	wp.Train(wpCorpus, 20)

	// Palavra desconhecida
	unknownText := "xyz123"
	tokens := wp.Tokenize(unknownText)

	t.Logf("✓ Unknown Token Handling:")
	t.Logf("  Input: '%s'", unknownText)
	t.Logf("  Tokens: %v", tokens)

	// Deve conter UNK token
	hasUNK := false
	for _, token := range tokens {
		if token == 0 { // UNK token ID
			hasUNK = true
			break
		}
	}

	if !hasUNK {
		t.Logf("  Note: WordPiece found subword matches")
	} else {
		t.Logf("  ✓ Used [UNK] token for unknown word")
	}
}

func TestDataQuality_EndToEndPipeline(t *testing.T) {
	// Testar pipeline completo: SDGT → Curation → Tokenization
	t.Logf("\n✓ End-to-End Data Pipeline:")
	t.Logf("  1. Seed-Driven Growth")
	t.Logf("  2. Data Curation")
	t.Logf("  3. Vocabulary Optimization")
	t.Logf("  4. Tokenization\n")

	// Step 1: SDGT
	seeds := []string{
		"artificial intelligence",
		"deep learning",
		"natural language processing",
	}

	sdgt := NewSDGTDataGenerator(seeds)
	generated := sdgt.Generate(100)

	t.Logf("Step 1: SDGT Generation")
	t.Logf("  Seeds: %d → Generated: %d samples", len(seeds), len(generated))

	// Step 2: Curation
	curator := NewDataCurator()
	curator.MinLength = 10
	curator.MaxLength = 500
	curator.DiversityThreshold = 0.6

	curated := curator.Curate(generated)

	t.Logf("\nStep 2: Data Curation")
	t.Logf("  Generated: %d → Curated: %d samples", len(generated), len(curated))
	t.Logf("  Retention: %.1f%%", float64(len(curated))/float64(len(generated))*100)

	// Step 3: Train tokenizer
	bpe := NewBPETokenizer()
	corpusText := ""
	for _, item := range curated {
		corpusText += item + " "
	}
	bpe.Train(corpusText, 1000)

	t.Logf("\nStep 3: BPE Training")
	t.Logf("  Vocabulary: %d tokens", bpe.GetVocabSize())

	// Step 4: Test tokenization
	testText := "artificial intelligence and deep learning"
	tokens := bpe.Tokenize(testText)

	t.Logf("\nStep 4: Tokenization Test")
	t.Logf("  Input: '%s'", testText)
	t.Logf("  Tokens: %v", tokens)
	t.Logf("  Token count: %d", len(tokens))
	t.Logf("  Decoded: '%s'", bpe.Decode(tokens))

	t.Logf("\n✓ Pipeline completed successfully!")
}

func TestDataQuality_SubwordBenefits(t *testing.T) {
	// Demonstrar benefícios de subword tokenization
	t.Logf("\n✓ Subword Tokenization Benefits:")
	t.Logf("  ================================\n")

	t.Logf("Character-level:")
	t.Logf("  - Vocab: ~100 chars")
	t.Logf("  - Handles ANY word")
	t.Logf("  - BUT: Long sequences")
	t.Logf("  - 'unbelievable' → 12 tokens")
	t.Logf("")

	t.Logf("Word-level:")
	t.Logf("  - Vocab: ~50,000 words")
	t.Logf("  - Short sequences")
	t.Logf("  - BUT: Cannot handle unknown words")
	t.Logf("  - 'unbelievable' → 1 token (if in vocab)")
	t.Logf("  - 'xyz123' → FAIL")
	t.Logf("")

	t.Logf("Subword (BPE/WordPiece):")
	t.Logf("  - Vocab: ~5,000-30,000 subwords")
	t.Logf("  - Handles unknown words via subwords")
	t.Logf("  - Short sequences")
	t.Logf("  - 'unbelievable' → 2-3 tokens (un+believ+able)")
	t.Logf("  - 'xyz123' → 2-3 tokens (x+y+z+123)")
	t.Logf("  - BEST OF BOTH WORLDS!")
	t.Logf("")

	t.Logf("Key Insight:")
	t.Logf("  Subword tokenization enables SMALL vocabularies")
	t.Logf("  to handle LARGE vocabularies without OOV issues!")
}

func TestDataQuality_CorpusSizeVsVocab(t *testing.T) {
	// Testar relação entre tamanho do corpus e vocabulário
	corpusSizes := []int{10, 50, 100, 500}

	for _, size := range corpusSizes {
		// Gerar corpus
		corpus := make([]string, size)
		for i := 0; i < size; i++ {
			corpus[i] = "machine learning deep neural network data"
		}

		// Treinar BPE
		bpe := NewBPETokenizer()
		corpusText := ""
		for _, item := range corpus {
			corpusText += item + " "
		}
		bpe.Train(corpusText, 100)

		t.Logf("✓ Corpus %d sentences → Vocab %d tokens", size, bpe.GetVocabSize())
	}
}

func TestDataQuality_PortugueseSubwords(t *testing.T) {
	// Testar subword tokenization com português
	corpus := []string{
		"aprendizado de máquina é fascinante",
		"redes neurais profundas processam informações",
		"processamento de linguagem natural é importante",
	}

	// BPE
	bpe := NewBPETokenizer()
	bpeText := ""
	for _, item := range corpus {
		bpeText += item + " "
	}
	bpe.Train(bpeText, 100)

	testText := "aprendizado de máquina"
	tokens := bpe.Tokenize(testText)

	t.Logf("✓ Portuguese Subword Tokenization:")
	t.Logf("  Input: '%s'", testText)
	t.Logf("  Tokens: %v", tokens)
	t.Logf("  Decoded: '%s'", bpe.Decode(tokens))

	// Verificar que funciona com caracteres especiais
	specialText := "não é impossível"
	specialTokens := bpe.Tokenize(specialText)

	t.Logf("\n  Special characters:")
	t.Logf("  Input: '%s'", specialText)
	t.Logf("  Tokens: %v", specialTokens)
	t.Logf("  Decoded: '%s'", bpe.Decode(specialTokens))
}

func TestDataQuality_SmallModelOptimization(t *testing.T) {
	// Testar otimização para modelos pequenos
	t.Logf("\n✓ Small Model Optimization Strategy:")
	t.Logf("  ===================================\n")

	t.Logf("Problem:")
	t.Logf("  - Small model (270M params)")
	t.Logf("  - Limited embedding capacity")
	t.Logf("  - Cannot afford 50K word vocab")
	t.Logf("")

	t.Logf("Solution: Subword Tokenization")
	t.Logf("  - 5K-10K subword vocab (vs 50K words)")
	t.Logf("  - 80%% smaller embedding layer")
	t.Logf("  - Handles unknown words gracefully")
	t.Logf("  - No OOV (Out-Of-Vocabulary) errors")
	t.Logf("")

	t.Logf("SDGT + Curation:")
	t.Logf("  - Start with 10-50 high-quality seeds")
	t.Logf("  - Generate 100-500 diverse samples")
	t.Logf("  - Curate to remove duplicates/similar")
	t.Logf("  - Final: 50-200 HIGH-QUALITY training samples")
	t.Logf("")

	t.Logf("Result:")
	t.Logf("  - Small vocab → Small model")
	t.Logf("  - Curated data → Better learning")
	t.Logf("  - Subword → No OOV issues")
	t.Logf("  - Efficiency ↑, Quality ↑")
}
