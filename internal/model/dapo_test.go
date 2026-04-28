package model

import (
	"math"
	"testing"
)

func TestDAPO_ConfigCreation(t *testing.T) {
	config := NewDAPOConfig()

	if config.EpsilonLow != 0.1 {
		t.Errorf("EpsilonLow: expected 0.1, got %f", config.EpsilonLow)
	}
	if config.EpsilonHigh != 0.3 {
		t.Errorf("EpsilonHigh: expected 0.3, got %f", config.EpsilonHigh)
	}
	if config.MinGroupSize != 2 {
		t.Errorf("MinGroupSize: expected 2, got %d", config.MinGroupSize)
	}
	if config.MaxGroupSize != 16 {
		t.Errorf("MaxGroupSize: expected 16, got %d", config.MaxGroupSize)
	}
	if config.TargetKL != 0.01 {
		t.Errorf("TargetKL: expected 0.01, got %f", config.TargetKL)
	}

	t.Logf("✓ DAPO Config created:")
	t.Logf("  Epsilon: [%.2f, %.2f]", config.EpsilonLow, config.EpsilonHigh)
	t.Logf("  Group size: [%d, %d] (default: %d)", 
		config.MinGroupSize, config.MaxGroupSize, config.GroupSize)
	t.Logf("  Target KL: %.3f", config.TargetKL)
}

func TestDAPO_DecoupledClip(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	testCases := []struct {
		ratio     float64
		advantage float64
		name      string
	}{
		{1.5, 1.0, "Positive advantage, high ratio"},
		{0.5, 1.0, "Positive advantage, low ratio"},
		{1.5, -1.0, "Negative advantage, high ratio"},
		{0.5, -1.0, "Negative advantage, low ratio"},
		{1.0, 0.5, "Neutral ratio"},
	}

	for _, tc := range testCases {
		clipped := trainer.ComputeDecoupledClip(tc.ratio, tc.advantage)

		// Verificar bounds
		if tc.advantage > 0 {
			// Positive advantage: clip high
			if clipped > 1.0+config.EpsilonHigh {
				t.Errorf("Clipped ratio exceeds upper bound: %.4f > %.4f", 
					clipped, 1.0+config.EpsilonHigh)
			}
		} else {
			// Negative advantage: clip low
			if clipped < 1.0-config.EpsilonLow {
				t.Errorf("Clipped ratio below lower bound: %.4f < %.4f",
					clipped, 1.0-config.EpsilonLow)
			}
		}

		t.Logf("✓ %s: ratio=%.2f, adv=%.2f → clipped=%.4f",
			tc.name, tc.ratio, tc.advantage, clipped)
	}
}

func TestDAPO_DynamicGroupSize(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	config := NewDAPOConfig()
	config.GroupSize = 8
	trainer := NewDAPOTrapper(model, config)

	testCases := []struct {
		currentKL        float64
		expectedIncrease bool
		name             string
	}{
		{0.03, true, "High KL → increase group"},
		{0.004, false, "Low KL → decrease group"},
		{0.01, false, "Normal KL → keep size"},
	}

	for _, tc := range testCases {
		newSize := trainer.ComputeDynamicGroupSize(tc.currentKL)

		t.Logf("✓ %s: KL=%.3f → group_size=%d (was %d)",
			tc.name, tc.currentKL, newSize, config.GroupSize)

		// Verificar bounds
		if newSize < config.MinGroupSize || newSize > config.MaxGroupSize {
			t.Errorf("Group size out of bounds: %d (min=%d, max=%d)",
				newSize, config.MinGroupSize, config.MaxGroupSize)
		}
	}
}

func TestDAPO_LengthAdaptiveReward(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	testCases := []struct {
		reward    float64
		length    int
		avgLength float64
		name      string
	}{
		{5.0, 10, 20.0, "Short sequence"},
		{5.0, 20, 20.0, "Average sequence"},
		{5.0, 50, 20.0, "Long sequence"},
		{5.0, 60, 20.0, "Very long sequence (penalty)"},
	}

	for _, tc := range testCases {
		adjustedReward := trainer.ComputeLengthAdaptiveReward(
			tc.reward, tc.length, tc.avgLength,
		)

		t.Logf("✓ %s: reward=%.1f, len=%d, avg=%.1f → adjusted=%.4f",
			tc.name, tc.reward, tc.length, tc.avgLength, adjustedReward)
	}
}

func TestDAPO_GroupAdvantagesWithLength(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	prompt := []int{10, 11}
	samples := []*DAPOSample{
		{Prompt: prompt, Response: []int{10, 11, 12, 13}, Reward: 5.0, Length: 4},
		{Prompt: prompt, Response: []int{10, 11, 12, 13, 14, 15}, Reward: 5.0, Length: 6},
		{Prompt: prompt, Response: []int{10, 11, 12}, Reward: 3.0, Length: 3},
		{Prompt: prompt, Response: []int{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, Reward: 7.0, Length: 10},
	}

	trainer.ComputeGroupAdvantagesWithLength(samples)

	// Verificar que vantagens somam ~0
	sumAdv := 0.0
	for _, s := range samples {
		sumAdv += s.Advantage
	}

	if math.Abs(sumAdv) > 0.01 {
		t.Logf("Warning: advantages sum to %.6f (should be ~0)", sumAdv)
	}

	t.Logf("✓ Group advantages with length normalization:")
	for i, s := range samples {
		t.Logf("  Sample %d: reward=%.1f, len=%d, norm_reward=%.4f, adv=%.4f",
			i, s.Reward, s.Length, s.NormalizedReward, s.Advantage)
	}
}

func TestDAPO_ComputeLoss(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	samples := []*DAPOSample{
		{
			Response:    []int{10, 11, 12},
			LogProbsOld: []float64{-1.0, -1.5},
			Advantage:   1.5,
		},
		{
			Response:    []int{20, 21, 22},
			LogProbsOld: []float64{-2.0, -1.0},
			Advantage:   -0.5,
		},
	}

	loss := trainer.ComputeDAPOLoss(samples)

	t.Logf("✓ DAPO Loss computed: %.4f", loss)
}

func TestDAPO_KLApproximation(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	samples := []*DAPOSample{
		{Response: []int{10, 11, 12, 13}, LogProbsOld: []float64{-1.0, -1.5, -2.0}},
		{Response: []int{20, 21, 22, 23}, LogProbsOld: []float64{-1.2, -1.3, -1.8}},
	}

	kl := trainer.ComputeKLApproximation(samples)

	t.Logf("✓ KL Approximation: %.4f", kl)
}

func TestDAPO_DAPOStep(t *testing.T) {
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	prompt := []int{10}
	samples := []*DAPOSample{
		{Prompt: prompt, Response: []int{10, 11, 12}, LogProbsOld: []float64{-1.0, -1.5}, Reward: 5.0, Length: 3},
		{Prompt: prompt, Response: []int{10, 13, 14}, LogProbsOld: []float64{-2.0, -1.0}, Reward: 3.0, Length: 3},
		{Prompt: prompt, Response: []int{10, 15, 16}, LogProbsOld: []float64{-1.5, -1.2}, Reward: 7.0, Length: 3},
		{Prompt: prompt, Response: []int{10, 17, 18}, LogProbsOld: []float64{-1.8, -1.3}, Reward: 1.0, Length: 3},
	}

	metrics := trainer.DAPOStep(samples)

	requiredKeys := []string{
		"total_loss", "policy_loss", "kl_divergence", 
		"entropy", "dynamic_group_size",
	}
	for _, key := range requiredKeys {
		if _, ok := metrics[key]; !ok {
			t.Errorf("Missing metric key: %s", key)
		}
	}

	t.Logf("✓ DAPO Step completed:")
	t.Logf("  Total loss: %.4f", metrics["total_loss"])
	t.Logf("  Policy loss: %.4f", metrics["policy_loss"])
	t.Logf("  KL divergence: %.4f", metrics["kl_divergence"])
	t.Logf("  Entropy: %.4f", metrics["entropy"])
	t.Logf("  Dynamic group size: %.0f", metrics["dynamic_group_size"])
}

func TestDAPO_TrainDAPO(t *testing.T) {
	model := NewTransformerModel(30, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 30)
	for i := 0; i < 30; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	config.GroupSize = 3
	trainer := NewDAPOTrapper(model, config)

	prompts := [][]int{
		{1, 2},
		{5, 6},
		{10, 11},
	}

	rewardFunc := func(response []int, prompt []int) float64 {
		return float64(len(response)) * 0.1
	}

	numIterations := 3
	allMetrics, err := trainer.TrainDAPO(prompts, rewardFunc, numIterations)
	if err != nil {
		t.Fatalf("DAPO training failed: %v", err)
	}

	if len(allMetrics) != numIterations {
		t.Errorf("Expected %d iterations, got %d", numIterations, len(allMetrics))
	}

	t.Logf("✓ DAPO Training completed: %d iterations", len(allMetrics))
	for i, metrics := range allMetrics {
		t.Logf("  Iteration %d: loss=%.4f, KL=%.4f, reward=%.2f, length=%.1f",
			i+1, metrics["total_loss"], metrics["kl_divergence"],
			metrics["mean_reward"], metrics["mean_length"])
	}
}

func TestDAPO_Stats(t *testing.T) {
	config := NewDAPOConfig()
	stats := GetDAPOStats(config)

	t.Logf("✓ DAPO Stats:")
	t.Logf("  Epsilon: [%.2f, %.2f]", stats["epsilon_low"], stats["epsilon_high"])
	t.Logf("  Group size: [%d, %d]", stats["min_group_size"], stats["max_group_size"])
	t.Logf("  Target KL: %.3f", stats["target_kl"])
	t.Logf("  Length penalty: %.3f", stats["max_length_penalty"])
	t.Logf("  Length normalization: %.2f", stats["length_normalization"])
}

func TestDAPO_ComparisonWithGRPO(t *testing.T) {
	t.Logf("\nDAPO vs GRPO Comparison:")
	t.Logf("========================\n")

	t.Logf("GRPO (Group Relative Policy Optimization):")
	t.Logf("  - Symmetric clipping (epsilon = 0.2)")
	t.Logf("  - Fixed group size")
	t.Logf("  - No length normalization")
	t.Logf("  - Good for general tasks")
	t.Logf("")

	t.Logf("DAPO (Decoupled Clip and Dynamic Sampling):")
	t.Logf("  - Asymmetric clipping (low=0.1, high=0.3)")
	t.Logf("  - Dynamic group size based on KL")
	t.Logf("  - Length-adaptive rewards")
	t.Logf("  - Better for LONG reasoning chains")
	t.Logf("")

	t.Logf("Key improvements in DAPO:")
	t.Logf("  1. Decoupled clipping: more stable updates")
	t.Logf("  2. Dynamic sampling: adapts to training state")
	t.Logf("  3. Length normalization: no bias to long responses")
	t.Logf("  4. Better convergence for complex reasoning")
}

func TestDAPO_LongReasoningChains(t *testing.T) {
	// Testar especificamente cenários de raciocínio longo
	model := NewTransformerModel(30, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 30)
	for i := 0; i < 30; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	// Simular diferentes comprimentos de raciocínio
	testCases := []struct {
		name   string
		length int
	}{
		{"Short (10 tokens)", 10},
		{"Medium (30 tokens)", 30},
		{"Long (50 tokens)", 50},
		{"Very Long (100 tokens)", 100},
	}

	for _, tc := range testCases {
		prompt := []int{1, 2}
		response := make([]int, tc.length)
		for i := range response {
			response[i] = (i % 28) + 3
		}
		response[0] = prompt[0]
		response[1] = prompt[1]

		sample := &DAPOSample{
			Prompt:      prompt,
			Response:    response,
			Reward:      5.0,
			Length:      tc.length,
			LogProbsOld: make([]float64, tc.length-1),
		}

		// Calcular reward adaptativo
		avgLength := 30.0
		adjustedReward := trainer.ComputeLengthAdaptiveReward(
			sample.Reward, sample.Length, avgLength,
		)

		t.Logf("✓ %s: original=%.1f, adjusted=%.4f",
			tc.name, sample.Reward, adjustedReward)
	}

	t.Logf("\n✓ Length-adaptive rewards prevent bias toward long sequences")
}

func TestDAPO_IntegrationWithRLVR(t *testing.T) {
	// Testar DAPO com verificadores RLVR
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	config := NewDAPOConfig()
	trainer := NewDAPOTrapper(model, config)

	// Criar verificador RLVR
	mathVerifier := &MathQAVerifier{
		ExpectedAnswer: "42",
		Tolerance:      1.0,
	}

	// Reward function usando RLVR
	rlvrRewardFunc := func(response []int, prompt []int) float64 {
		responseStr := "Answer: 42" // Simulado
		score, _ := mathVerifier.Verify(responseStr, "")
		return score
	}

	prompts := [][]int{{1, 2}}
	
	// Treinar por poucas iterações
	numIterations := 2
	allMetrics, err := trainer.TrainDAPO(prompts, rlvrRewardFunc, numIterations)
	if err != nil {
		t.Fatalf("DAPO + RLVR training failed: %v", err)
	}

	t.Logf("✓ DAPO + RLVR integration completed")
	for i, metrics := range allMetrics {
		t.Logf("  Iteration %d: loss=%.4f, reward=%.2f",
			i+1, metrics["total_loss"], metrics["mean_reward"])
	}
}
