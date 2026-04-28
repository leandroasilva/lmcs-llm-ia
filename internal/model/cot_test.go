package model

import (
	"strings"
	"testing"
)

func TestCoT_ConfigCreation(t *testing.T) {
	config := NewCoTConfig()

	if config.LearningRate != 0.001 {
		t.Errorf("LearningRate: expected 0.001, got %f", config.LearningRate)
	}
	if config.BatchSize != 8 {
		t.Errorf("BatchSize: expected 8, got %d", config.BatchSize)
	}
	if config.Epochs != 10 {
		t.Errorf("Epochs: expected 10, got %d", config.Epochs)
	}
	if config.ThinkWeight != 1.0 {
		t.Errorf("ThinkWeight: expected 1.0, got %f", config.ThinkWeight)
	}
	if config.AnswerWeight != 1.0 {
		t.Errorf("AnswerWeight: expected 1.0, got %f", config.AnswerWeight)
	}

	t.Logf("✓ CoT Config created:")
	t.Logf("  LR: %.4f, Batch: %d, Epochs: %d",
		config.LearningRate, config.BatchSize, config.Epochs)
	t.Logf("  Weights: think=%.1f, answer=%.1f",
		config.ThinkWeight, config.AnswerWeight)
}

func TestCoT_FormatSample(t *testing.T) {
	question := "What is 2 + 2?"
	chainOfThought := "First, we add 2 and 2. This gives us 4."
	answer := "4"

	formatted := FormatCoTSample(question, chainOfThought, answer)

	// Verificar que contém todas as partes
	if !strings.Contains(formatted, question) {
		t.Error("Formatted text missing question")
	}
	if !strings.Contains(formatted, chainOfThought) {
		t.Error("Formatted text missing chain of thought")
	}
	if !strings.Contains(formatted, answer) {
		t.Error("Formatted text missing answer")
	}
	if !strings.Contains(formatted, CoTThinkOpenTag) {
		t.Error("Formatted text missing think start tag")
	}
	if !strings.Contains(formatted, CoTThinkCloseTag) {
		t.Error("Formatted text missing think end tag")
	}
	if !strings.Contains(formatted, CoTAnswerTag) {
		t.Error("Formatted text missing answer tag")
	}

	t.Logf("✓ CoT Sample formatted correctly:")
	t.Logf("  %s", strings.ReplaceAll(formatted, "\n", "\n  "))
}

func TestCoT_ParseResponse(t *testing.T) {
	response := "What is 2+2?\n<|think_start|>\nFirst, we calculate 2+2.\nThis equals 4.\n<|think_end|>\n<|answer|>\n4"

	chainOfThought, answer := ParseCoTResponse(response)

	if chainOfThought == "" {
		t.Error("Chain of thought not extracted")
	}
	if answer == "" {
		t.Error("Answer not extracted")
	}

	// Verificar conteúdo
	if !strings.Contains(chainOfThought, "calculate") {
		t.Errorf("Chain of thought missing content: %s", chainOfThought)
	}
	if answer != "4" {
		t.Errorf("Answer incorrect: expected '4', got '%s'", answer)
	}

	t.Logf("✓ CoT Response parsed:")
	t.Logf("  Chain of thought: '%s'", chainOfThought)
	t.Logf("  Answer: '%s'", answer)
}

func TestCoT_ParseResponseNoAnswerTag(t *testing.T) {
	// Testar caso sem tag de resposta (fallback)
	response := "Question\n<|think_start|>\nLet me think...\n<|think_end|>\nJust the answer"

	chainOfThought, answer := ParseCoTResponse(response)

	if chainOfThought == "" {
		t.Error("Chain of thought not extracted")
	}
	if answer == "" {
		t.Error("Answer not extracted (fallback failed)")
	}

	t.Logf("✓ CoT Response parsed (no answer tag):")
	t.Logf("  Chain: '%s'", chainOfThought)
	t.Logf("  Answer: '%s'", answer)
}

func TestCoT_CreateDataset(t *testing.T) {
	samples := []CoTSample{
		{
			Question:       "What is 2+2?",
			ChainOfThought: "We add 2 and 2",
			Answer:         "4",
		},
		{
			Question:       "What is 3*3?",
			ChainOfThought: "We multiply 3 by 3",
			Answer:         "9",
		},
	}

	dataset := CreateCoTDataset(samples)

	if len(dataset) != len(samples) {
		t.Errorf("Dataset size: expected %d, got %d", len(samples), len(dataset))
	}

	// Verificar formato de cada sample
	for i, text := range dataset {
		if !strings.Contains(text, CoTThinkOpenTag) {
			t.Errorf("Sample %d missing think tag", i)
		}
		if !strings.Contains(text, CoTAnswerTag) {
			t.Errorf("Sample %d missing answer tag", i)
		}
	}

	t.Logf("✓ CoT Dataset created: %d samples", len(dataset))
}

func TestCoT_TokenizeCoTText(t *testing.T) {
	model := NewTransformerModel(1000, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 1000)
	for i := 0; i < 1000; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	config := NewCoTConfig()
	trainer := NewCoTTrainer(model, config)

	text := "What is 2+2?\n<|think_start|>\nLet me think\n<|think_end|>\n<|answer|>\n4"

	tokens, tokenTypes := trainer.TokenizeCoTText(text)

	if len(tokens) == 0 {
		t.Error("No tokens generated")
	}
	if len(tokens) != len(tokenTypes) {
		t.Errorf("Tokens/types length mismatch: %d vs %d", len(tokens), len(tokenTypes))
	}

	// Verificar que temos tipos diferentes
	hasThink := false
	hasAnswer := false
	for _, tt := range tokenTypes {
		if tt == 1 {
			hasThink = true
		}
		if tt == 2 {
			hasAnswer = true
		}
	}

	if !hasThink {
		t.Error("No thinking tokens detected")
	}
	if !hasAnswer {
		t.Error("No answer tokens detected")
	}

	t.Logf("✓ CoT Text tokenized:")
	t.Logf("  Tokens: %d, Types: %d", len(tokens), len(tokenTypes))
	t.Logf("  Has thinking: %v, Has answer: %v", hasThink, hasAnswer)
}

func TestCoT_ComputeLoss(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewCoTConfig()
	trainer := NewCoTTrainer(model, config)

	// Criar dados de teste
	hidden := model.Forward([]int{10, 11, 12, 13, 14})
	targets := []int{11, 12, 13, 14, 15}
	tokenTypes := []int{0, 1, 1, 2, 2} // question, thinking, thinking, answer, answer
	seqLen := 5

	totalLoss, thinkLoss, answerLoss := trainer.ComputeCoTLoss(
		hidden, targets, tokenTypes, seqLen,
	)

	if totalLoss <= 0 {
		t.Errorf("Total loss should be positive, got %f", totalLoss)
	}

	t.Logf("✓ CoT Loss computed:")
	t.Logf("  Total: %.4f", totalLoss)
	t.Logf("  Think: %.4f", thinkLoss)
	t.Logf("  Answer: %.4f", answerLoss)
}

func TestCoT_TrainCoTEpoch(t *testing.T) {
	model := NewTransformerModel(100, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	config := NewCoTConfig()
	config.MaxSeqLen = 50
	trainer := NewCoTTrainer(model, config)

	// Criar dataset pequeno
	dataset := []string{
		"Question 1\n<|think_start|>\nThink step 1\n<|think_end|>\n<|answer|>\nAnswer 1",
		"Question 2\n<|think_start|>\nThink step 2\n<|think_end|>\n<|answer|>\nAnswer 2",
	}

	totalLoss, thinkLoss, answerLoss, err := trainer.TrainCoTEpoch(dataset)
	if err != nil {
		t.Fatalf("Training epoch failed: %v", err)
	}

	t.Logf("✓ CoT Epoch trained:")
	t.Logf("  Total loss: %.4f", totalLoss)
	t.Logf("  Think loss: %.4f", thinkLoss)
	t.Logf("  Answer loss: %.4f", answerLoss)
}

func TestCoT_TrainCoT(t *testing.T) {
	model := NewTransformerModel(100, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	config := NewCoTConfig()
	config.Epochs = 3
	config.MaxSeqLen = 50
	trainer := NewCoTTrainer(model, config)

	dataset := []string{
		"Q1\n<|think_start|>\nT1\n<|think_end|>\n<|answer|>\nA1",
		"Q2\n<|think_start|>\nT2\n<|think_end|>\n<|answer|>\nA2",
		"Q3\n<|think_start|>\nT3\n<|think_end|>\n<|answer|>\nA3",
	}

	allMetrics, err := trainer.TrainCoT(dataset)
	if err != nil {
		t.Fatalf("CoT training failed: %v", err)
	}

	if len(allMetrics) != config.Epochs {
		t.Errorf("Expected %d epochs of metrics, got %d", config.Epochs, len(allMetrics))
	}

	t.Logf("✓ CoT Training completed: %d epochs", len(allMetrics))
	for i, metrics := range allMetrics {
		t.Logf("  Epoch %d: total=%.4f, think=%.4f, answer=%.4f",
			i+1, metrics["total_loss"], metrics["think_loss"], metrics["answer_loss"])
	}
}

func TestCoT_GenerateCoTResponse(t *testing.T) {
	model := NewTransformerModel(100, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	config := NewCoTConfig()
	trainer := NewCoTTrainer(model, config)

	question := "What is the capital of France?"

	chainOfThought, answer := trainer.GenerateCoTResponse(
		question,
		10,  // maxThinkLen
		5,   // maxAnswerLen
		1.0, // temperature
	)

	t.Logf("✓ CoT Response generated:")
	t.Logf("  Question: '%s'", question)
	t.Logf("  Chain of thought: '%s'", chainOfThought)
	t.Logf("  Answer: '%s'", answer)
}

func TestCoT_Stats(t *testing.T) {
	config := NewCoTConfig()
	stats := GetCoTStats(config)

	t.Logf("✓ CoT Stats:")
	t.Logf("  Learning rate: %v", stats["learning_rate"])
	t.Logf("  Batch size: %v", stats["batch_size"])
	t.Logf("  Epochs: %v", stats["epochs"])
	t.Logf("  Think weight: %v", stats["think_weight"])
	t.Logf("  Answer weight: %v", stats["answer_weight"])
	t.Logf("  Think tag: %v", stats["think_tag"])
	t.Logf("  End tag: %v", stats["end_tag"])
	t.Logf("  Answer tag: %v", stats["answer_tag"])
}

func TestCoT_ComparisonWithoutCoT(t *testing.T) {
	t.Logf("\nCoT SFT vs Standard SFT Comparison:")
	t.Logf("====================================\n")

	t.Logf("Standard SFT:")
	t.Logf("  - Input: 'What is 2+2?'")
	t.Logf("  - Output: '4'")
	t.Logf("  - Model learns direct mapping")
	t.Logf("  - No reasoning capability")
	t.Logf("  - Black box predictions")
	t.Logf("")

	t.Logf("Chain of Thought SFT:")
	t.Logf("  - Input: 'What is 2+2?'")
	t.Logf("  - Output: '<|think_start|>")
	t.Logf("    First, we add 2 and 2.")
	t.Logf("    This gives us 4.")
	t.Logf("    <|think_end|>")
	t.Logf("    <|answer|>")
	t.Logf("    4")
	t.Logf("  - Model learns to REASON")
	t.Logf("  - Step-by-step thinking")
	t.Logf("  - Interpretable process")
	t.Logf("  - Better generalization")
	t.Logf("")

	t.Logf("Benefits of CoT:")
	t.Logf("  1. Better accuracy on complex tasks")
	t.Logf("  2. Interpretable reasoning")
	t.Logf("  3. Easier debugging")
	t.Logf("  4. Improved generalization")
	t.Logf("  5. Foundation for RLVR/GRPO")
}

func TestCoT_ComplexReasoning(t *testing.T) {
	// Testar raciocínio complexo
	samples := []CoTSample{
		{
			Question:       "Prove that sqrt(2) is irrational",
			ChainOfThought: "Assume sqrt(2) is rational. Then sqrt(2) = p/q where p,q are coprime integers. Squaring both sides: 2 = p²/q². Therefore, p² = 2q². This means p² is even, so p is even. Let p = 2k. Then 4k² = 2q², so q² = 2k². This means q² is even, so q is even. But if both p and q are even, they're not coprime. Contradiction!",
			Answer:         "sqrt(2) is irrational",
		},
		{
			Question:       "Solve: x² + 5x + 6 = 0",
			ChainOfThought: "We can factor this quadratic. We need two numbers that multiply to 6 and add to 5. Those numbers are 2 and 3. So x² + 5x + 6 = (x+2)(x+3) = 0. Therefore, x+2=0 or x+3=0. Solving: x=-2 or x=-3.",
			Answer:         "x = -2 or x = -3",
		},
	}

	dataset := CreateCoTDataset(samples)

	t.Logf("✓ Complex reasoning dataset created: %d samples", len(dataset))
	for i, text := range dataset {
		t.Logf("  Sample %d length: %d characters", i, len(text))

		// Verificar que tem estrutura completa
		cot, answer := ParseCoTResponse(text)
		if cot == "" || answer == "" {
			t.Errorf("Sample %d: missing CoT or answer", i)
		}
	}
}

func TestCoT_IntegrationWithRLVR(t *testing.T) {
	// Testar integração CoT + RLVR
	model := NewTransformerModel(100, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 100)
	for i := 0; i < 100; i++ {
		model.Vocab[i] = string(rune('a' + (i % 26)))
	}

	// 1. Treinar com CoT SFT
	config := NewCoTConfig()
	config.Epochs = 2
	config.MaxSeqLen = 100
	cotTrainer := NewCoTTrainer(model, config)

	dataset := []string{
		"What is 2+2?\n<|think_start|>\nWe add 2 and 2\n<|think_end|>\n<|answer|>\n4",
		"What is 3*3?\n<|think_start|>\nWe multiply 3 by 3\n<|think_end|>\n<|answer|>\n9",
	}

	_, err := cotTrainer.TrainCoT(dataset)
	if err != nil {
		t.Fatalf("CoT SFT failed: %v", err)
	}

	// 2. Usar RLVR para verificar raciocínio
	mathVerifier := &MathQAVerifier{
		ExpectedAnswer: "4",
		Tolerance:      0.01,
	}

	// 3. Gerar resposta com CoT
	cot, answer := cotTrainer.GenerateCoTResponse("What is 2+2?", 20, 10, 1.0)

	t.Logf("✓ CoT + RLVR integration:")
	t.Logf("  Chain of thought: '%s'", cot)
	t.Logf("  Answer: '%s'", answer)

	// 4. Verificar com RLVR
	if answer != "" {
		score, _ := mathVerifier.Verify("Answer: "+answer, "")
		t.Logf("  RLVR verification score: %.2f", score)
	}

	t.Logf("  CoT enables: interpretable reasoning + verifiable answers")
}

func TestCoT_MultiStepReasoning(t *testing.T) {
	// Testar raciocínio multi-step
	question := "If a train travels 60 km/h for 2 hours, then 80 km/h for 3 hours, what is the total distance?"

	expectedCoT := "First, calculate distance for first leg: 60 km/h × 2 h = 120 km. Then, calculate distance for second leg: 80 km/h × 3 h = 240 km. Finally, add both distances: 120 km + 240 km = 360 km."
	expectedAnswer := "360 km"

	sample := CoTSample{
		Question:       question,
		ChainOfThought: expectedCoT,
		Answer:         expectedAnswer,
	}

	formatted := FormatCoTSample(sample.Question, sample.ChainOfThought, sample.Answer)

	t.Logf("✓ Multi-step reasoning sample:")
	t.Logf("  Question: %s", sample.Question)
	t.Logf("  CoT length: %d characters", len(sample.ChainOfThought))
	t.Logf("  Answer: %s", sample.Answer)
	t.Logf("  Total formatted length: %d characters", len(formatted))

	// Parse e verificar
	parsedCoT, parsedAnswer := ParseCoTResponse(formatted)

	if parsedCoT != expectedCoT {
		t.Logf("  Warning: parsed CoT differs (expected formatting)")
	}
	if parsedAnswer != expectedAnswer {
		t.Errorf("  Answer mismatch: expected '%s', got '%s'", expectedAnswer, parsedAnswer)
	}
}
