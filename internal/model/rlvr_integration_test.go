package model

import (
	"fmt"
	"strings"
	"testing"
)

// TestRLVR_IntegrationWithGRPO demonstra integração completa RLVR + GRPO
func TestRLVR_IntegrationWithGRPO(t *testing.T) {
	// 1. Criar modelo Transformer
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// 2. Criar verificadores RLVR
	mathVerifier := &MathQAVerifier{
		ExpectedAnswer: "42",
		Tolerance:      0.5,
	}

	logicVerifier := &LogicPuzzleVerifier{
		ExpectedSteps: []string{"first", "calculate", "therefore"},
		Solution:      "42",
	}

	compositeVerifier := &CompositeVerifier{
		Verifiers: []RLVRVerifier{mathVerifier, logicVerifier},
		Weights:   []float64{0.5, 0.5},
	}

	// 3. Criar trainer GRPO com reward function baseada em RLVR
	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	grpoTrainer := NewGRPOTrainer(model, config)

	// 4. Reward function usando RLVR
	rlvrRewardFunc := func(response []int, prompt []int) float64 {
		// Converter tokens para string (simplificado)
		responseStr := fmt.Sprintf("%v", response)
		promptStr := fmt.Sprintf("%v", prompt)

		// Usar verificadores RLVR
		score, details := compositeVerifier.Verify(responseStr, promptStr)

		t.Logf("  RLVR verification: score=%.2f, details=%v", score, details["total_score"])

		return score
	}

	// 5. Preparar prompts
	prompts := [][]int{
		{1, 2, 3}, // "What is the answer?"
		{10, 11},  // "Solve this puzzle"
	}

	// 6. Treinar com GRPO + RLVR
	t.Logf("Starting GRPO training with RLVR rewards...")

	// Simular poucas iterações para teste
	numIterations := 2
	allMetrics, err := grpoTrainer.TrainGRPO(prompts, rlvrRewardFunc, numIterations)
	if err != nil {
		t.Fatalf("GRPO training failed: %v", err)
	}

	// 7. Verificar progresso
	t.Logf("\n✓ GRPO + RLVR Training completed: %d iterations", len(allMetrics))
	for i, metrics := range allMetrics {
		t.Logf("  Iteration %d: loss=%.4f, reward=%.2f",
			i+1, metrics["total_loss"], metrics["mean_reward"])
	}
}

// TestRLVR_DeepSeekR1Style demonstra abordagem DeepSeek-R1 (RL puro sem SFT)
func TestRLVR_DeepSeekR1Style(t *testing.T) {
	t.Logf("\nDeepSeek-R1 Approach: Pure RL without SFT")
	t.Logf("==========================================\n")

	t.Logf("Traditional approach:")
	t.Logf("  1. Supervised Fine-Tuning (SFT)")
	t.Logf("  2. Train reward model with human feedback")
	t.Logf("  3. RLHF with PPO")
	t.Logf("  4. Expensive, slow, subjective")
	t.Logf("")

	t.Logf("DeepSeek-R1 approach (RLVR):")
	t.Logf("  1. NO SFT needed")
	t.Logf("  2. Use verifiable rewards directly")
	t.Logf("  3. GRPO for efficient optimization")
	t.Logf("  4. Emergent reasoning from environment")
	t.Logf("")

	// Exemplo de problema verificável
	mathVerifier := &MathQAVerifier{
		ExpectedAnswer: "100",
		Tolerance:      1.0,
	}

	// Problema: "What is 10 * 10?"
	testCases := []string{
		"Answer: 100",      // Correto
		"Answer: 99",       // Quase
		"Answer: 50",       // Errado
		"I think it's 100", // Incerto mas correto
	}

	t.Logf("Testing verifiable rewards:\n")
	for _, response := range testCases {
		score, details := mathVerifier.Verify(response, "")
		t.Logf("  Response: '%s'", response)
		t.Logf("    → Score: %.2f, Expected: %v, Actual: %v",
			score, details["expected"], details["actual"])
	}

	t.Logf("\n✓ DeepSeek-R1 proved that:")
	t.Logf("  - Reasoning emerges from interaction")
	t.Logf("  - Verifiable rewards are sufficient")
	t.Logf("  - No need for human annotation")
}

// TestRLVR_EmergentReasoning demonstra como raciocínio emerge
func TestRLVR_EmergentReasoning(t *testing.T) {
	t.Logf("\nEmergent Reasoning through RLVR")
	t.Logf("================================\n")

	// Criar verificador de lógica complexo
	complexVerifier := &LogicPuzzleVerifier{
		ExpectedSteps: []string{
			"assume",
			"if",
			"then",
			"contradiction",
			"therefore",
		},
		Solution: "the statement is true",
	}

	// Diferentes níveis de raciocínio
	responses := []struct {
		name     string
		response string
	}{
		{
			"No reasoning",
			"The statement is true",
		},
		{
			"Basic reasoning",
			"If we assume X, then Y. Therefore, the statement is true",
		},
		{
			"Complete reasoning",
			"Assume the statement is false. If X is true, then we have a contradiction. Therefore, the statement must be true.",
		},
		{
			"Deep reasoning",
			"Assume the opposite. If this were true, then by logic we derive a contradiction. Therefore, the original statement is true.",
		},
	}

	t.Logf("Testing different reasoning levels:\n")
	for _, tc := range responses {
		score, details := complexVerifier.Verify(tc.response, "")
		t.Logf("  %s:", tc.name)
		t.Logf("    Score: %.2f (solution: %v, steps: %d/%d)",
			score,
			details["solution_correct"],
			details["steps_found"],
			details["total_steps"])
	}

	t.Logf("\n✓ Key insight:")
	t.Logf("  - Model learns to reason because it gets HIGHER rewards")
	t.Logf("  - No explicit instruction to 'show your work'")
	t.Logf("  - Reasoning emerges naturally from reward structure")
}

// TestRLVR_ComparisonTable cria tabela comparativa
func TestRLVR_ComparisonTable(t *testing.T) {
	separator := strings.Repeat("=", 70)

	t.Logf("\n%s", separator)
	t.Logf("RLVR vs Traditional RLHF Comparison")
	t.Logf("%s\n", separator)

	t.Logf("%-30s | %-20s | %-20s", "Aspect", "Traditional RLHF", "RLVR")
	t.Logf("%-30s-+-%-20s-+-%-20s", "------------------------------", "--------------------", "--------------------")
	t.Logf("%-30s | %-20s | %-20s", "Reward Source", "Human annotators", "Verifiable rules")
	t.Logf("%-30s | %-20s | %-20s", "Cost", "High ($$$)", "Low ($)")
	t.Logf("%-30s | %-20s | %-20s", "Scalability", "Limited", "Infinite")
	t.Logf("%-30s | %-20s | %-20s", "Speed", "Slow (days/weeks)", "Fast (hours)")
	t.Logf("%-30s | %-20s | %-20s", "Objectivity", "Subjective", "Objective")
	t.Logf("%-30s | %-20s | %-20s", "Use Cases", "General tasks", "Code, math, logic")
	t.Logf("%-30s | %-20s | %-20s", "DeepSeek-R1", "No", "YES ✓")
	t.Logf("")

	t.Logf("%s", separator)
	t.Logf("✓ RLVR enables scalable, objective, efficient RL training")
	t.Logf("%s", separator)
}
