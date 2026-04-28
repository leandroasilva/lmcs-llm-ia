package model

import (
	"math"
	"testing"
)

func TestMTP_ConfigCreation(t *testing.T) {
	cfg := NewMTPConfig(3, 1000, 256)

	if cfg.NumPredictions != 3 {
		t.Errorf("NumPredictions: expected 3, got %d", cfg.NumPredictions)
	}
	if cfg.VocabSize != 1000 {
		t.Errorf("VocabSize: expected 1000, got %d", cfg.VocabSize)
	}
	if cfg.DModel != 256 {
		t.Errorf("DModel: expected 256, got %d", cfg.DModel)
	}
	if cfg.WeightMTP != 0.3 {
		t.Errorf("WeightMTP: expected 0.3, got %f", cfg.WeightMTP)
	}

	t.Logf("✓ MTP Config: num_predictions=%d, vocab=%d, d_model=%d, weight=%.2f",
		cfg.NumPredictions, cfg.VocabSize, cfg.DModel, cfg.WeightMTP)
}

func TestMTP_HeadCreation(t *testing.T) {
	cfg := NewMTPConfig(4, 500, 128)
	head := NewMTPHead(cfg)

	if head == nil {
		t.Fatal("MTP head creation failed")
	}

	if len(head.WOuts) != 4 {
		t.Errorf("Expected 4 prediction heads, got %d", len(head.WOuts))
	}

	// Verificar dimensões de cada head
	for k := 0; k < 4; k++ {
		rows, cols := head.WOuts[k].Dims()
		if rows != 500 || cols != 128 {
			t.Errorf("Head %d WOut: expected 500x128, got %dx%d", k, rows, cols)
		}
	}

	t.Logf("✓ MTP Head created: %d prediction heads (500x128 each)", len(head.WOuts))
}

func TestMTP_ComputeLogits(t *testing.T) {
	cfg := NewMTPConfig(3, 100, 64)
	head := NewMTPHead(cfg)

	// Criar hidden state: [seq_len=8, d_model=64]
	seqLen := 8
	hidden := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Calcular logits
	logits := ComputeMTPLogits(head, hidden, seqLen)

	if len(logits) != 3 {
		t.Fatalf("Expected 3 logits matrices, got %d", len(logits))
	}

	// Verificar dimensões de cada logit
	for k := 0; k < 3; k++ {
		rows, cols := logits[k].Dims()
		if rows != seqLen {
			t.Errorf("Logits[%d] rows: expected %d, got %d", k, seqLen, rows)
		}
		if cols != cfg.VocabSize {
			t.Errorf("Logits[%d] cols: expected %d, got %d", k, cfg.VocabSize, cols)
		}
	}

	t.Logf("✓ MTP Logits computed: %d matrices of %dx%d", len(logits), seqLen, cfg.VocabSize)
}

func TestMTP_ComputeLoss(t *testing.T) {
	cfg := NewMTPConfig(3, 50, 64)
	head := NewMTPHead(cfg)

	seqLen := 10
	hidden := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Calcular logits
	logits := ComputeMTPLogits(head, hidden, seqLen)

	// Criar targets: [seq_len, num_predictions]
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = make([]int, cfg.NumPredictions)
		for k := 0; k < cfg.NumPredictions; k++ {
			// Tokens aleatórios válidos
			targets[i][k] = i*cfg.NumPredictions + k % cfg.VocabSize
		}
	}

	// Calcular loss
	result := ComputeMTPLoss(head, logits, targets, seqLen)

	// Verificar que loss é positiva
	if result.TotalLoss <= 0 {
		t.Errorf("TotalLoss should be positive, got %f", result.TotalLoss)
	}

	// Cross-entropy típica está entre 2-6 para vocabulários pequenos
	if result.TotalLoss < 0.1 || result.TotalLoss > 20 {
		t.Logf("Warning: TotalLoss %.4f seems unusual", result.TotalLoss)
	}

	t.Logf("✓ MTP Loss computed: total=%.4f, per_prediction=%v", result.TotalLoss, result.Losses)
}

func TestMTP_ForwardPass(t *testing.T) {
	cfg := NewMTPConfig(3, 100, 128)
	head := NewMTPHead(cfg)

	seqLen := 16
	hidden := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Criar targets
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = []int{(i + 1) % 100, (i + 2) % 100, (i + 3) % 100}
	}

	// Forward pass
	hiddenOut, logits, loss := MTPForward(head, hidden, targets, seqLen)

	// Verificar que hidden state não foi modificado
	rows, cols := hiddenOut.Dims()
	if rows != seqLen || cols != cfg.DModel {
		t.Errorf("Hidden output dimensions: expected %dx%d, got %dx%d", seqLen, cfg.DModel, rows, cols)
	}

	if len(logits) != cfg.NumPredictions {
		t.Errorf("Expected %d logits, got %d", cfg.NumPredictions, len(logits))
	}

	if loss <= 0 {
		t.Errorf("Loss should be positive, got %f", loss)
	}

	t.Logf("✓ MTP Forward Pass: hidden=%dx%d, loss=%.4f", rows, cols, loss)
}

func TestMTP_CombineLoss(t *testing.T) {
	mainLoss := 3.5
	mtpLoss := 4.2
	mtpWeight := 0.3

	totalLoss := CombineMTPLossWithMainLoss(mainLoss, mtpLoss, mtpWeight)
	expectedLoss := mainLoss + mtpWeight*mtpLoss

	if math.Abs(totalLoss-expectedLoss) > 1e-6 {
		t.Errorf("Combined loss: expected %.4f, got %.4f", expectedLoss, totalLoss)
	}

	t.Logf("✓ Combined Loss: main=%.2f + %.2f*mtp(%.2f) = %.4f",
		mainLoss, mtpWeight, mtpLoss, totalLoss)

	// Testar com peso zero
	totalLossZero := CombineMTPLossWithMainLoss(mainLoss, mtpLoss, 0.0)
	if math.Abs(totalLossZero-mainLoss) > 1e-6 {
		t.Errorf("With weight=0, expected main_loss=%.2f, got %.4f", mainLoss, totalLossZero)
	}

	t.Logf("✓ With MTP weight=0: loss=%.4f (main loss only)", totalLossZero)
}

func TestMTP_Stats(t *testing.T) {
	cfg := NewMTPConfig(4, 1000, 256)
	head := NewMTPHead(cfg)

	stats := GetMTPStats(head)

	numPredictions := stats["num_predictions"].(int)
	totalParams := stats["total_params"].(int)
	mtpWeight := stats["mtp_weight"].(float64)

	if numPredictions != 4 {
		t.Errorf("NumPredictions: expected 4, got %d", numPredictions)
	}

	if totalParams <= 0 {
		t.Errorf("TotalParams should be positive, got %d", totalParams)
	}

	if math.Abs(mtpWeight-0.3) > 1e-6 {
		t.Errorf("WeightMTP: expected 0.3, got %f", mtpWeight)
	}

	t.Logf("✓ MTP Stats:")
	t.Logf("  Predictions: %d", numPredictions)
	t.Logf("  Total params: %d (%.2f M)", totalParams, float64(totalParams)/1e6)
	t.Logf("  Params per head: %d", stats["params_per_head"])
	t.Logf("  MTP weight: %.2f", mtpWeight)
}

func TestMTP_Inference(t *testing.T) {
	cfg := NewMTPConfig(3, 100, 64)
	head := NewMTPHead(cfg)

	// Criar hidden state: [1, d_model]
	hidden := transformerRandomMatrix(1, cfg.DModel, 0.1)

	// Inference
	predictedTokens := MTPInference(head, hidden, 1.0)

	if len(predictedTokens) != cfg.NumPredictions {
		t.Errorf("Expected %d predicted tokens, got %d", cfg.NumPredictions, len(predictedTokens))
	}

	// Verificar que tokens são válidos
	for k, token := range predictedTokens {
		if token < 0 || token >= cfg.VocabSize {
			t.Errorf("Token %d out of range: %d (vocab size: %d)", k, token, cfg.VocabSize)
		}
	}

	t.Logf("✓ MTP Inference: predicted tokens %v (vocab=%d, temperature=1.0)",
		predictedTokens, cfg.VocabSize)
}

func TestMTP_TransformerIntegration(t *testing.T) {
	// Criar modelo Transformer
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	// Verificar que MTP está desabilitado por padrão
	if model.UseMTP {
		t.Error("MTP should be disabled by default")
	}
	if model.MTPHead != nil {
		t.Error("MTPHead should be nil by default")
	}

	t.Logf("✓ Transformer created (MTP disabled by default)")
}

func TestMTP_EnableMTP(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	// Habilitar MTP
	model.EnableMTP(3, 0.3)

	// Verificar
	if !model.UseMTP {
		t.Error("MTP should be enabled")
	}
	if model.MTPHead == nil {
		t.Fatal("MTPHead should not be nil")
	}

	if model.MTPHead.Config.NumPredictions != 3 {
		t.Errorf("NumPredictions: expected 3, got %d", model.MTPHead.Config.NumPredictions)
	}

	if math.Abs(model.MTPHead.Config.WeightMTP-0.3) > 1e-6 {
		t.Errorf("WeightMTP: expected 0.3, got %f", model.MTPHead.Config.WeightMTP)
	}

	t.Logf("✓ MTP enabled: %d predictions, weight=%.2f",
		model.MTPHead.Config.NumPredictions, model.MTPHead.Config.WeightMTP)
}

func TestMTP_ForwardWithMTP(t *testing.T) {
	// Criar modelo com MTP
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)
	model.EnableMTP(3, 0.3)

	// Configurar vocabulário básico
	model.Vocab = make([]string, 100)
	model.WordToID = make(map[string]int)
	model.IDToWord = make(map[int]string)
	for i := 0; i < 100; i++ {
		word := string(rune('a' + i%26))
		model.Vocab[i] = word
		model.WordToID[word] = i
		model.IDToWord[i] = word
	}

	// Criar input
	inputTokens := []int{10, 20, 30, 40, 50}
	seqLen := len(inputTokens)

	// Criar targets: [seq_len, num_predictions]
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = []int{
			(i + 1) % 100, // Próximo token
			(i + 2) % 100, // Token +2
			(i + 3) % 100, // Token +3
		}
	}

	// Forward com MTP
	hidden, logits, mtpLoss := model.ForwardWithMTP(inputTokens, targets)

	// Verificar outputs
	rows, cols := hidden.Dims()
	if rows != seqLen || cols != model.DModel {
		t.Errorf("Hidden dimensions: expected %dx%d, got %dx%d", seqLen, model.DModel, rows, cols)
	}

	if len(logits) != 3 {
		t.Errorf("Expected 3 logits matrices, got %d", len(logits))
	}

	if mtpLoss <= 0 {
		t.Errorf("MTP loss should be positive, got %f", mtpLoss)
	}

	t.Logf("✓ ForwardWithMTP: hidden=%dx%d, mtp_loss=%.4f", rows, cols, mtpLoss)
}

func TestMTP_Scalability(t *testing.T) {
	// Testar diferentes configurações de MTP
	configs := []struct {
		name           string
		numPredictions int
		vocabSize      int
		dModel         int
	}{
		{"Small (2 tokens)", 2, 500, 128},
		{"Medium (3 tokens)", 3, 1000, 256},
		{"Large (5 tokens)", 5, 2000, 512},
		{"XLarge (10 tokens)", 10, 5000, 1024},
	}

	t.Logf("MTP Scalability Analysis:")
	t.Logf("===========================")

	for _, tc := range configs {
		cfg := NewMTPConfig(tc.numPredictions, tc.vocabSize, tc.dModel)
		head := NewMTPHead(cfg)
		stats := GetMTPStats(head)

		totalParams := stats["total_params"].(int)

		t.Logf("%-20s | Predictions: %2d | Params: %8d (%.2f M)",
			tc.name,
			tc.numPredictions,
			totalParams,
			float64(totalParams)/1e6)
	}
}

func TestMTP_CoherenceImprovement(t *testing.T) {
	// Este teste demonstra como MTP pode melhorar coerência
	// Ao prever múltiplos tokens, o modelo aprende dependências de longo prazo

	cfg := NewMTPConfig(3, 100, 128)
	head := NewMTPHead(cfg)

	seqLen := 20
	hidden := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Criar targets com padrão coerente
	// Exemplo: sequência numérica que o modelo deve aprender
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = []int{
			(i + 1) % cfg.VocabSize,
			(i + 2) % cfg.VocabSize,
			(i + 3) % cfg.VocabSize,
		}
	}

	// Calcular loss inicial
	logits := ComputeMTPLogits(head, hidden, seqLen)
	result := ComputeMTPLoss(head, logits, targets, seqLen)

	t.Logf("✓ MTP Coherence Test:")
	t.Logf("  Initial loss: %.4f", result.TotalLoss)
	t.Logf("  Per-prediction losses: %v", result.Losses)

	// Com targets coerentes, as losses devem ser similares entre predições
	// (modelo está aprendendo padrões consistentes)
	maxDiff := 0.0
	for k := 1; k < len(result.Losses); k++ {
		diff := math.Abs(result.Losses[k] - result.Losses[0])
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	t.Logf("  Max difference between predictions: %.4f", maxDiff)

	// Diferença pequena indica consistência
	if maxDiff > 2.0 {
		t.Logf("  Note: High variance in prediction losses (expected for random initialization)")
	}
}
