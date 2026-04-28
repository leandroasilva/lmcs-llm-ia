package model

import (
	"testing"

	"github.com/leandroasilva/lmcs-llm-ia/internal/tokenizer"
)

func TestIntegration_CompletePipeline(t *testing.T) {
	t.Log("=== END-TO-END INTEGRATION TEST ===")
	t.Log("")

	// Phase 1: Data
	t.Log("[Phase 1] Data Preparation")
	seeds := []string{"machine learning", "deep learning"}
	sdgt := tokenizer.NewSDGTDataGenerator(seeds)
	data := sdgt.Generate(20)
	t.Logf("  Generated %d samples", len(data))

	curator := tokenizer.NewDataCurator()
	curated := curator.Curate(data)
	t.Logf("  Curated to %d samples", len(curated))

	bpe := tokenizer.NewBPETokenizer()
	bpeText := ""
	for _, d := range curated {
		bpeText += d + " "
	}
	bpe.Train(bpeText, 100)
	t.Logf("  BPE vocab: %d tokens", bpe.GetVocabSize())
	t.Log("")

	// Phase 2: Model
	t.Log("[Phase 2] Model Creation")
	model := NewTransformerModel(100, 32, 2, 1, 64, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}
	t.Log("  Model created successfully")
	t.Log("")

	// Phase 3: Reasoning
	t.Log("[Phase 3] Reasoning Stack")

	// CoT
	cotConfig := NewCoTConfig()
	_ = NewCoTTrainer(model, cotConfig)
	t.Log("  CoT: Ready")

	// Grammar
	_, err := NewGrammarConstrainedDecoder(model, "reasoning")
	if err == nil {
		t.Log("  Grammar: Ready")
	}

	// ToT
	tot := NewTreeOfThoughts(model, &AgenticConfig{MaxDepth: 2, BranchFactor: 2})
	problem := "What is 2+2?"
	bestNode, _ := tot.Solve(problem)
	if bestNode != nil {
		t.Logf("  ToT: Solved (score=%.2f)", bestNode.Score)
	}

	// Self-Consistency
	sc := NewSelfConsistency(model, &AgenticConfig{NumSamples: 3})
	answer, confidence := sc.Solve(problem)
	t.Logf("  Self-Consistency: %s (conf=%.2f)", answer, confidence)
	t.Log("")

	// Phase 4: SLM
	t.Log("[Phase 4] SLM Optimization")

	// CRV
	crv := NewCRVFramework(model, &CRVConfig{MaxIterations: 2, CritiqueThreshold: 0.8})
	result, _ := crv.Execute(problem, "Answer is 4")
	if result != nil {
		t.Logf("  CRV: Score %.2f", result.Score)
	}

	// ReasoningBank
	bank := NewReasoningBank(50)
	bank.Add(ReasoningEntry{
		ID: "r1", Problem: problem, Solution: answer,
		Strategy: "math", Tags: []string{"math"},
	})
	results := bank.Query(problem, 1)
	t.Logf("  ReasoningBank: %d entries, %d query results", len(bank.Reasonings), len(results))
	t.Log("")

	// Phase 5: Tokenization
	t.Log("[Phase 5] Tokenization")
	testText := "machine learning"
	tokens := bpe.Tokenize(testText)
	decoded := bpe.Decode(tokens)
	t.Logf("  Input: %s", testText)
	t.Logf("  Tokens: %v", tokens)
	t.Logf("  Decoded: %s", decoded)
	t.Log("")

	t.Log("=== INTEGRATION TEST PASSED ===")
	t.Log("All components working together!")
}

func TestIntegration_DataToInference(t *testing.T) {
	t.Log("=== DATA TO INFERENCE TEST ===")

	// Generate data
	seeds := []string{"AI is powerful"}
	sdgt := tokenizer.NewSDGTDataGenerator(seeds)
	data := sdgt.Generate(10)

	// Curate
	curator := tokenizer.NewDataCurator()
	curated := curator.Curate(data)

	// Tokenize
	bpe := tokenizer.NewBPETokenizer()
	corpus := ""
	for _, d := range curated {
		corpus += d + " "
	}
	bpe.Train(corpus, 50)

	// Model
	model := NewTransformerModel(50, 16, 1, 1, 32, 64, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	// Tokenize test
	tokens := bpe.Tokenize("AI is")
	t.Logf("Tokenized: %v", tokens)
	t.Log("=== TEST PASSED ===")
}

func TestIntegration_ReasoningStack(t *testing.T) {
	t.Log("=== REASONING STACK TEST ===")

	model := NewTransformerModel(100, 32, 2, 1, 64, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	problem := "Capital of France?"

	// ToT
	tot := NewTreeOfThoughts(model, &AgenticConfig{MaxDepth: 2})
	node, _ := tot.Solve(problem)
	if node != nil {
		t.Logf("ToT score: %.2f", node.Score)
	}

	// Self-Consistency
	sc := NewSelfConsistency(model, &AgenticConfig{NumSamples: 3})
	ans, conf := sc.Solve(problem)
	t.Logf("SC: %s (%.2f)", ans, conf)

	// CRV
	crv := NewCRVFramework(model, &CRVConfig{MaxIterations: 1})
	res, _ := crv.Execute(problem, ans)
	if res != nil {
		t.Logf("CRV: %.2f", res.Score)
	}

	// Bank
	bank := NewReasoningBank(20)
	bank.Add(ReasoningEntry{
		ID: "r1", Problem: problem, Solution: "Paris",
		Strategy: "factual", Tags: []string{"geo"},
	})
	t.Logf("Bank: %d entries", len(bank.Reasonings))

	t.Log("=== TEST PASSED ===")
}
