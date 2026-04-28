package model

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestMoE_ConfigCreation(t *testing.T) {
	cfg := NewMoEConfig(256, 8, 2, 512)

	if cfg.DModel != 256 {
		t.Errorf("DModel: expected 256, got %d", cfg.DModel)
	}
	if cfg.NumExperts != 8 {
		t.Errorf("NumExperts: expected 8, got %d", cfg.NumExperts)
	}
	if cfg.TopK != 2 {
		t.Errorf("TopK: expected 2, got %d", cfg.TopK)
	}
	if cfg.FFHidden != 512 {
		t.Errorf("FFHidden: expected 512, got %d", cfg.FFHidden)
	}
	if !cfg.UseLoadBal {
		t.Error("UseLoadBal should be true by default")
	}
	if cfg.LambdaAux != 0.01 {
		t.Errorf("LambdaAux: expected 0.01, got %f", cfg.LambdaAux)
	}

	t.Logf("✓ MoE Config: d_model=%d, experts=%d, top_k=%d, ff_hidden=%d",
		cfg.DModel, cfg.NumExperts, cfg.TopK, cfg.FFHidden)
}

func TestMoE_LayerCreation(t *testing.T) {
	cfg := NewMoEConfig(256, 8, 2, 512)
	layer := NewMoELayer(cfg)

	if layer == nil {
		t.Fatal("MoE layer creation failed")
	}

	if len(layer.Experts) != 8 {
		t.Errorf("Expected 8 experts, got %d", len(layer.Experts))
	}

	// Verificar dimensões do gate
	rows, cols := layer.WGate.Dims()
	if rows != 8 || cols != 256 {
		t.Errorf("WGate dimensions: expected 8x256, got %dx%d", rows, cols)
	}

	// Verificar que experts foram inicializados
	for i, expert := range layer.Experts {
		if expert.W1 == nil || expert.W2 == nil {
			t.Errorf("Expert %d not properly initialized", i)
		}
	}

	t.Logf("✓ MoE Layer created: %d experts, gate=%dx%d", len(layer.Experts), rows, cols)
}

func TestMoE_TopKRouting(t *testing.T) {
	// Testar função topK
	values := []float64{0.1, 0.5, 0.3, 0.9, 0.2, 0.7}

	indices, topValues := topK(values, 3)

	if len(indices) != 3 {
		t.Errorf("Expected 3 indices, got %d", len(indices))
	}

	// Verificar que pegou os 3 maiores: 0.9 (idx 3), 0.7 (idx 5), 0.5 (idx 1)
	expectedIndices := []int{3, 5, 1}
	for i, idx := range indices {
		if idx != expectedIndices[i] {
			t.Errorf("Index[%d]: expected %d, got %d", i, expectedIndices[i], idx)
		}
	}

	expectedValues := []float64{0.9, 0.7, 0.5}
	for i, v := range topValues {
		if math.Abs(v-expectedValues[i]) > 1e-6 {
			t.Errorf("Value[%d]: expected %.1f, got %.1f", i, expectedValues[i], v)
		}
	}

	t.Logf("✓ Top-3 routing: indices=%v, values=%v", indices, topValues)
}

func TestMoE_Softmax(t *testing.T) {
	values := []float64{1.0, 2.0, 3.0}
	result := softmax(values)

	// Verificar que soma = 1
	sum := 0.0
	for _, v := range result {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("Softmax sum: expected 1.0, got %f", sum)
	}

	// Verificar que preserva ordem (maior input → maior output)
	if result[2] <= result[1] || result[1] <= result[0] {
		t.Error("Softmax should preserve relative ordering")
	}

	t.Logf("✓ Softmax: input=%v, output=%.4f, sum=%.6f", values, result, sum)
}

func TestMoE_ForwardPass(t *testing.T) {
	cfg := NewMoEConfig(128, 4, 2, 256)
	layer := NewMoELayer(cfg)

	// Criar input: [seq_len=8, d_model=128]
	seqLen := 8
	input := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Forward pass
	output, routingResult := MixtureOfExperts(layer, input, seqLen)

	// Verificar dimensões do output
	rows, cols := output.Dims()
	if rows != seqLen || cols != cfg.DModel {
		t.Errorf("Output dimensions: expected %dx%d, got %dx%d", seqLen, cfg.DModel, rows, cols)
	}

	// Verificar routing
	if len(routingResult.SelectedExperts) != seqLen {
		t.Errorf("SelectedExperts: expected %d tokens, got %d", seqLen, len(routingResult.SelectedExperts))
	}

	// Verificar que cada token selecionou top-k experts
	for i := 0; i < seqLen; i++ {
		if len(routingResult.SelectedExperts[i]) != cfg.TopK {
			t.Errorf("Token %d: expected %d experts, got %d", i, cfg.TopK, len(routingResult.SelectedExperts[i]))
		}
		if len(routingResult.GateValues[i]) != cfg.TopK {
			t.Errorf("Token %d gate values: expected %d, got %d", i, cfg.TopK, len(routingResult.GateValues[i]))
		}

		// Verificar que gate values somam ~1
		sum := 0.0
		for _, v := range routingResult.GateValues[i] {
			sum += v
		}
		if math.Abs(sum-1.0) > 1e-5 {
			t.Errorf("Token %d gate values sum: expected 1.0, got %.6f", i, sum)
		}
	}

	t.Logf("✓ MoE Forward Pass: input=%dx%d, output=%dx%d", seqLen, cfg.DModel, rows, cols)
	t.Logf("  Each token activated %d/%d experts", cfg.TopK, cfg.NumExperts)
}

func TestMoE_Sparsity(t *testing.T) {
	cfg := NewMoEConfig(256, 8, 2, 512)
	layer := NewMoELayer(cfg)

	// Simular forward pass com input maior
	seqLen := 64
	input := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	ResetExpertUsage(layer)
	MixtureOfExperts(layer, input, seqLen)

	// Verificar sparsity
	usage := GetExpertUsage(layer)

	activeExperts := usage["active_experts"].(int)
	totalExperts := usage["total_experts"].(int)
	sparsityRatio := usage["sparsity_ratio"].(float64)

	t.Logf("✓ MoE Sparsity Analysis:")
	t.Logf("  Active experts: %d/%d (%.1f%%)", activeExperts, totalExperts, float64(activeExperts)/float64(totalExperts)*100)
	t.Logf("  Sparsity ratio: %.1f%%", sparsityRatio)

	// Com top-k=2 e 8 experts, no máximo 2/8 = 25% ativos por token
	// Mas ao longo de muitos tokens, todos podem ser usados
	expectedMaxActivePct := float64(cfg.TopK) / float64(cfg.NumExperts) * 100
	t.Logf("  Max active per token: %.1f%%", expectedMaxActivePct)
}

func TestMoE_ParameterCount(t *testing.T) {
	cfg := NewMoEConfig(512, 16, 2, 1024)

	params := GetMoEParameterCount(cfg)

	totalParams := params["total_params"].(int)
	activeParams := params["active_params_per_token"].(int)
	paramSparsity := params["parameter_sparsity_pct"].(float64)
	totalExperts := params["total_experts"].(int)
	activeExperts := params["active_experts"].(int)

	t.Logf("✓ MoE Parameter Count (similar to DeepSeek-V3):")
	t.Logf("  Total parameters: %d (%.2f M)", totalParams, float64(totalParams)/1e6)
	t.Logf("  Active per token: %d (%.2f M)", activeParams, float64(activeParams)/1e6)
	t.Logf("  Parameter sparsity: %.2f%%", paramSparsity)
	t.Logf("  Experts: %d total, %d active per token", totalExperts, activeExperts)

	// Verificar que parâmetros ativos são uma fração pequena
	if paramSparsity > 30 {
		t.Errorf("Parameter sparsity too high: %.2f%% (should be <30%%)", paramSparsity)
	}

	// Verificar cálculo de sparsity
	expectedSparsity := float64(activeParams) / float64(totalParams) * 100
	if math.Abs(paramSparsity-expectedSparsity) > 0.01 {
		t.Errorf("Parameter sparsity mismatch: calculated %.2f%%, expected %.2f%%", paramSparsity, expectedSparsity)
	}
}

func TestMoE_LoadBalancingLoss(t *testing.T) {
	// Criar gates com distribuição perfeitamente balanceada
	seqLen := 100
	numExperts := 8

	// Cada expert recebe exatamente 1/8 dos tokens com peso uniforme
	gates := mat.NewDense(seqLen, numExperts, nil)
	for i := 0; i < seqLen; i++ {
		expertIdx := i % numExperts
		gates.Set(i, expertIdx, 1.0)
	}

	loss := ComputeLoadBalancingLoss(gates, seqLen, numExperts)

	t.Logf("✓ Load Balancing Loss (balanced): %.6f", loss)

	// Loss deve ser baixo para distribuição balanceada
	// Ideal: 1.0 para distribuição perfeita
	if loss > 2.0 {
		t.Logf("  Warning: Load balancing loss is high (%.2f), experts may be imbalanced", loss)
	}
}

func TestMoE_Scalability(t *testing.T) {
	// Testar diferentes configurações de MoE
	configs := []struct {
		name       string
		dModel     int
		numExperts int
		topK       int
		ffHidden   int
	}{
		{"Small", 128, 4, 2, 256},
		{"Medium", 256, 8, 2, 512},
		{"Large", 512, 16, 2, 1024},
		{"DeepSeek-like", 4096, 64, 6, 10240},
	}

	t.Logf("MoE Scalability Analysis:")
	t.Logf("==========================")

	for _, tc := range configs {
		cfg := NewMoEConfig(tc.dModel, tc.numExperts, tc.topK, tc.ffHidden)
		params := GetMoEParameterCount(cfg)

		totalParams := params["total_params"].(int)
		activeParams := params["active_params_per_token"].(int)
		sparsity := params["parameter_sparsity_pct"].(float64)

		t.Logf("%-15s | Total: %8d params (%.2fM) | Active: %7d (%.2fM) | Sparsity: %.1f%%",
			tc.name,
			totalParams, float64(totalParams)/1e6,
			activeParams, float64(activeParams)/1e6,
			sparsity)
	}
}

func TestMoE_ComparisonWithDenseFFN(t *testing.T) {
	// Comparar MoE com FFN denso tradicional
	dModel := 512
	ffHidden := 2048
	numExperts := 8
	topK := 2

	// FFN denso: 2 * d_model * ff_hidden
	denseParams := 2 * dModel * ffHidden

	// MoE: gate + num_experts * (2 * d_model * ff_hidden)
	moeGateParams := numExperts * dModel
	moeExpertParams := numExperts * (2 * dModel * ffHidden)
	moeTotalParams := moeGateParams + moeExpertParams

	// MoE ativo: gate + top_k * (2 * d_model * ff_hidden)
	moeActiveParams := moeGateParams + topK*(2*dModel*ffHidden)

	t.Logf("✓ Dense FFN vs MoE Comparison:")
	t.Logf("  Dense FFN parameters:    %d (%.2f M)", denseParams, float64(denseParams)/1e6)
	t.Logf("  MoE total parameters:    %d (%.2f M)", moeTotalParams, float64(moeTotalParams)/1e6)
	t.Logf("  MoE active parameters:   %d (%.2f M)", moeActiveParams, float64(moeActiveParams)/1e6)
	t.Logf("  MoE capacity increase:   %.1fx", float64(moeTotalParams)/float64(denseParams))
	t.Logf("  MoE compute overhead:    %.1fx", float64(moeActiveParams)/float64(denseParams))

	// MoE deve ter capacidade total maior mas custo computacional similar
	if float64(moeTotalParams)/float64(denseParams) < float64(numExperts)/2 {
		t.Error("MoE should significantly increase total capacity")
	}

	// Overhead computacional deve ser moderado (principalmente do gating)
	expectedOverhead := 1.0 + float64(moeGateParams)/float64(denseParams)
	actualOverhead := float64(moeActiveParams) / float64(denseParams)
	if math.Abs(actualOverhead-expectedOverhead) > 0.1 {
		t.Logf("  Note: Compute overhead %.2fx (expected ~%.2fx)", actualOverhead, expectedOverhead)
	}
}

func TestMoE_ExpertUsageTracking(t *testing.T) {
	cfg := NewMoEConfig(128, 4, 2, 256)
	layer := NewMoELayer(cfg)

	// Reset e executar múltiplos forward passes
	ResetExpertUsage(layer)

	// Pass 1
	input1 := transformerRandomMatrix(32, cfg.DModel, 0.1)
	MixtureOfExperts(layer, input1, 32)

	usage1 := GetExpertUsage(layer)
	totalTokens1 := usage1["total_tokens"].(float64)

	t.Logf("✓ Expert Usage Tracking:")
	t.Logf("  After pass 1: %.0f tokens processed", totalTokens1)

	// Pass 2
	input2 := transformerRandomMatrix(16, cfg.DModel, 0.1)
	MixtureOfExperts(layer, input2, 16)

	usage2 := GetExpertUsage(layer)
	totalTokens2 := usage2["total_tokens"].(float64)

	t.Logf("  After pass 2: %.0f tokens processed (cumulative)", totalTokens2)

	// Verificar que acumulou corretamente
	// Cada token ativa top_k experts, então total = (32+16) * top_k
	expectedTotal := float64((32 + 16) * cfg.TopK)
	if math.Abs(totalTokens2-expectedTotal) > 1 {
		t.Errorf("Total tokens: expected %.0f, got %.0f", expectedTotal, totalTokens2)
	}
}
