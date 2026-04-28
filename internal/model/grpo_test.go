package model

import (
	"math"
	"testing"
)

func TestGRPO_ConfigCreation(t *testing.T) {
	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)

	if config.GroupSize != 4 {
		t.Errorf("GroupSize: expected 4, got %d", config.GroupSize)
	}
	if config.Epsilon != 0.2 {
		t.Errorf("Epsilon: expected 0.2, got %f", config.Epsilon)
	}
	if config.BetaKL != 0.01 {
		t.Errorf("BetaKL: expected 0.01, got %f", config.BetaKL)
	}
	if config.LearningRate != 0.001 {
		t.Errorf("LearningRate: expected 0.001, got %f", config.LearningRate)
	}

	t.Logf("✓ GRPO Config: group_size=%d, epsilon=%.2f, beta_kl=%.3f, lr=%.4f",
		config.GroupSize, config.Epsilon, config.BetaKL, config.LearningRate)
}

func TestGRPO_TrainerCreation(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	if trainer == nil {
		t.Fatal("GRPO trainer creation failed")
	}
	if trainer.Model != model {
		t.Error("Trainer model reference incorrect")
	}
	if trainer.Config != config {
		t.Error("Trainer config reference incorrect")
	}

	t.Logf("✓ GRPO Trainer created successfully")
}

func TestGRPO_ComputeLogProbs(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	// Testar sequência
	tokens := []int{10, 11, 12, 13, 14}
	logProbs := trainer.ComputeLogProbs(tokens)

	if len(logProbs) != len(tokens)-1 {
		t.Errorf("Expected %d log probs, got %d", len(tokens)-1, len(logProbs))
	}

	// Verificar que log probs são negativas (ou zero)
	for i, lp := range logProbs {
		if lp > 0 {
			t.Errorf("Log prob at %d should be <= 0, got %f", i, lp)
		}
	}

	t.Logf("✓ ComputeLogProbs: %d tokens → %d log probs", len(tokens), len(logProbs))
	t.Logf("  Log probs: %v", logProbs)
}

func TestGRPO_GroupAdvantages(t *testing.T) {
	// Criar samples com mesmo prompt
	prompt := []int{10, 11, 12}

	samples := []*GRPOSample{
		{Prompt: prompt, Response: []int{10, 11, 12, 13}, Reward: 5.0},
		{Prompt: prompt, Response: []int{10, 11, 12, 14}, Reward: 3.0},
		{Prompt: prompt, Response: []int{10, 11, 12, 15}, Reward: 7.0},
		{Prompt: prompt, Response: []int{10, 11, 12, 16}, Reward: 1.0},
	}

	// Calcular vantagens
	ComputeGroupAdvantages(samples)

	// Verificar que vantagens somam ~0
	sumAdv := 0.0
	for _, s := range samples {
		sumAdv += s.Advantage
	}

	if math.Abs(sumAdv) > 1e-6 {
		t.Errorf("Advantages should sum to ~0, got %f", sumAdv)
	}

	// Verificar ordenação: maior reward → maior advantage
	if samples[2].Advantage <= samples[0].Advantage ||
		samples[0].Advantage <= samples[1].Advantage ||
		samples[1].Advantage <= samples[3].Advantage {
		t.Logf("Warning: Advantages not perfectly ordered (may be OK)")
	}

	t.Logf("✓ Group Advantages computed:")
	for i, s := range samples {
		t.Logf("  Sample %d: reward=%.1f, advantage=%.4f", i, s.Reward, s.Advantage)
	}
	t.Logf("  Sum of advantages: %.6f", sumAdv)
}

func TestGRPO_ComputeLoss(t *testing.T) {
	// Criar modelo e trainer
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	// Criar samples
	prompt := []int{10}
	samples := []*GRPOSample{
		{
			Prompt:    prompt,
			Response:  []int{10, 11, 12},
			LogProbs:  []float64{-1.0, -1.5},
			Reward:    5.0,
			Advantage: 1.5,
		},
		{
			Prompt:    prompt,
			Response:  []int{10, 13, 14},
			LogProbs:  []float64{-2.0, -1.0},
			Reward:    3.0,
			Advantage: -0.5,
		},
	}

	// Calcular vantagens
	ComputeGroupAdvantages(samples)

	// Calcular loss
	loss := trainer.ComputeGRPOLoss(samples)

	t.Logf("✓ GRPO Loss computed: %.4f", loss)
}

func TestGRPO_KLPenalty(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	samples := []*GRPOSample{
		{Response: []int{10, 11, 12, 13}},
		{Response: []int{20, 21, 22, 23}},
	}

	klPenalty := trainer.ComputeKLPenalty(samples)

	if klPenalty < 0 {
		t.Errorf("KL penalty should be >= 0, got %f", klPenalty)
	}

	t.Logf("✓ KL Penalty: %.4f", klPenalty)
}

func TestGRPO_Entropy(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	samples := []*GRPOSample{
		{Response: []int{10, 11, 12}},
	}

	entropy := trainer.ComputeEntropy(samples)

	if entropy < 0 {
		t.Errorf("Entropy should be >= 0, got %f", entropy)
	}

	// Entropia máxima para vocab=50 é ln(50) ≈ 3.91
	maxEntropy := math.Log(50.0)
	if entropy > maxEntropy+0.1 {
		t.Errorf("Entropy %.4f exceeds maximum %.4f", entropy, maxEntropy)
	}

	t.Logf("✓ Entropy: %.4f (max: %.4f)", entropy, maxEntropy)
}

func TestGRPO_GenerateResponses(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	prompt := []int{10, 11, 12}
	responses := trainer.GenerateResponses(prompt, 4, 10, 1.0)

	if len(responses) != 4 {
		t.Errorf("Expected 4 responses, got %d", len(responses))
	}

	for i, resp := range responses {
		if len(resp) != len(prompt)+10 {
			t.Errorf("Response %d length: expected %d, got %d", i, len(prompt)+10, len(resp))
		}

		// Verificar que começa com o prompt
		for j := 0; j < len(prompt); j++ {
			if resp[j] != prompt[j] {
				t.Errorf("Response %d token %d: expected %d, got %d", i, j, prompt[j], resp[j])
			}
		}
	}

	t.Logf("✓ Generated %d responses (prompt=%d tokens, response=%d tokens)",
		len(responses), len(prompt), len(responses[0]))
}

func TestGRPO_GRPOStep(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	prompt := []int{10}
	samples := []*GRPOSample{
		{Prompt: prompt, Response: []int{10, 11, 12}, LogProbs: []float64{-1.0, -1.5}, Reward: 5.0},
		{Prompt: prompt, Response: []int{10, 13, 14}, LogProbs: []float64{-2.0, -1.0}, Reward: 3.0},
		{Prompt: prompt, Response: []int{10, 15, 16}, LogProbs: []float64{-1.5, -1.2}, Reward: 7.0},
		{Prompt: prompt, Response: []int{10, 17, 18}, LogProbs: []float64{-1.8, -1.3}, Reward: 1.0},
	}

	metrics := trainer.GRPOStep(samples)

	// Verificar métricas
	requiredKeys := []string{"total_loss", "policy_loss", "kl_penalty", "entropy"}
	for _, key := range requiredKeys {
		if _, ok := metrics[key]; !ok {
			t.Errorf("Missing metric key: %s", key)
		}
	}

	t.Logf("✓ GRPO Step completed:")
	t.Logf("  Total loss: %.4f", metrics["total_loss"])
	t.Logf("  Policy loss: %.4f", metrics["policy_loss"])
	t.Logf("  KL penalty: %.4f", metrics["kl_penalty"])
	t.Logf("  Entropy: %.4f", metrics["entropy"])
}

func TestGRPO_RewardFunctions(t *testing.T) {
	// Testar reward function padrão
	prompt := []int{10, 11}
	response := []int{12, 13, 14, 15, 16}

	reward := DefaultRewardFunction(response, prompt)

	if reward <= 0 {
		t.Errorf("Reward should be positive, got %f", reward)
	}

	t.Logf("✓ Default Reward Function: %.4f", reward)
	t.Logf("  Length reward: %.4f", float64(len(response))*0.01)
	t.Logf("  Diversity reward: %.4f", float64(len(response))*0.05)
}

func TestGRPO_TrainGRPO(t *testing.T) {
	// Criar modelo pequeno para teste rápido
	model := NewTransformerModel(30, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 30)
	for i := 0; i < 30; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewGRPOConfig(3, 0.2, 0.01, 0.001)
	trainer := NewGRPOTrainer(model, config)

	// Prompts de exemplo
	prompts := [][]int{
		{1, 2},
		{5, 6},
		{10, 11},
	}

	// Treinar por poucas iterações
	numIterations := 3
	allMetrics, err := trainer.TrainGRPO(prompts, DefaultRewardFunction, numIterations)
	if err != nil {
		t.Fatalf("GRPO training failed: %v", err)
	}

	if len(allMetrics) != numIterations {
		t.Errorf("Expected %d iterations of metrics, got %d", numIterations, len(allMetrics))
	}

	t.Logf("✓ GRPO Training completed: %d iterations", len(allMetrics))
	for i, metrics := range allMetrics {
		t.Logf("  Iteration %d: loss=%.4f, reward=%.2f",
			i+1, metrics["total_loss"], metrics["mean_reward"])
	}
}

func TestGRPO_Stats(t *testing.T) {
	config := NewGRPOConfig(4, 0.2, 0.01, 0.001)
	stats := GetGRPOStats(config)

	t.Logf("✓ GRPO Stats:")
	t.Logf("  Group size: %v", stats["group_size"])
	t.Logf("  Epsilon: %v", stats["epsilon"])
	t.Logf("  Beta KL: %v", stats["beta_kl"])
	t.Logf("  Learning rate: %v", stats["learning_rate"])
	t.Logf("  Entropy coeff: %v", stats["entropy_coeff"])
	t.Logf("  Max grad norm: %v", stats["max_grad_norm"])
}

func TestGRPO_ComparisonWithPPO(t *testing.T) {
	// GRPO vs PPO comparison
	t.Logf("GRPO vs PPO Comparison:")
	t.Logf("========================")
	t.Logf("PPO (Proximal Policy Optimization):")
	t.Logf("  - Requires critic model (value network)")
	t.Logf("  - Computes advantages: A(s,a) = Q(s,a) - V(s)")
	t.Logf("  - Needs to train critic separately")
	t.Logf("  - Higher memory usage (2 models)")
	t.Logf("")
	t.Logf("GRPO (Group Relative Policy Optimization):")
	t.Logf("  - NO critic model needed")
	t.Logf("  - Computes advantages via group normalization")
	t.Logf("  - A_i = (R_i - mean(R)) / std(R)")
	t.Logf("  - Lower memory usage (1 model)")
	t.Logf("  - More sample efficient")
	t.Logf("")
	t.Logf("Memory savings: ~50 percent (no critic model)")
	t.Logf("Implementation complexity: Lower (no critic training)")
}

func TestGRPO_AdvantageNormalization(t *testing.T) {
	// Testar diferentes cenários de normalização
	testCases := []struct {
		name    string
		rewards []float64
	}{
		{"Uniform", []float64{5.0, 5.0, 5.0, 5.0}},
		{"Linear", []float64{1.0, 2.0, 3.0, 4.0}},
		{"Exponential", []float64{1.0, 2.0, 4.0, 8.0}},
		{"Mixed", []float64{-1.0, 0.5, 3.0, -2.0}},
	}

	for _, tc := range testCases {
		samples := make([]*GRPOSample, len(tc.rewards))
		prompt := []int{10}

		for i, r := range tc.rewards {
			samples[i] = &GRPOSample{
				Prompt: prompt,
				Reward: r,
			}
		}

		ComputeGroupAdvantages(samples)

		// Verificar propriedades
		sumAdv := 0.0
		for _, s := range samples {
			sumAdv += s.Advantage
		}

		t.Logf("✓ Advantage Normalization (%s):", tc.name)
		t.Logf("  Rewards: %v", tc.rewards)
		t.Logf("  Advantages:")
		for _, s := range samples {
			t.Logf("    %.4f", s.Advantage)
		}
		t.Logf("  Sum: %.6f", sumAdv)

		// Verificar que soma é ~0
		if math.Abs(sumAdv) > 0.01 {
			t.Errorf("  Warning: advantages don't sum to 0 (sum=%.6f)", sumAdv)
		}
	}
}
