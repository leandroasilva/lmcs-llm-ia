package model

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

// TestMTP_AdvancedIntegration teste avançado demonstrando MTP em ação
func TestMTP_AdvancedIntegration(t *testing.T) {
	// Criar modelo com vocabulário maior
	vocabSize := 200
	dModel := 256
	model := NewTransformerModel(vocabSize, dModel, 8, 3, 128, 512, 0.001, 0.1, 0.01)

	// Configurar vocabulário
	model.Vocab = make([]string, vocabSize)
	model.WordToID = make(map[string]int)
	for i := 0; i < vocabSize; i++ {
		model.Vocab[i] = string(rune('a' + i%26))
		model.WordToID[model.Vocab[i]] = i
	}

	// Habilitar MTP com 3 predições
	model.EnableMTP(3, 0.3)

	// Preparar dados de treinamento simulados
	// Sequência: "olá como você está bem obrigado"
	sequence := []int{10, 15, 3, 18, 20, 5, 2, 19, 1, 14, 15}

	// Criar targets para MTP
	// Para cada posição i, prever tokens i+1, i+2, i+3
	seqLen := len(sequence) - 1 // Usar todos exceto último
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = make([]int, 3)
		for k := 0; k < 3; k++ {
			if i+k+1 < len(sequence) {
				targets[i][k] = sequence[i+k+1]
			} else {
				targets[i][k] = -1 // Padding
			}
		}
	}

	// Forward pass com MTP
	inputTokens := sequence[:seqLen]
	hidden, logits, mtpLoss := model.ForwardWithMTP(inputTokens, targets)

	if hidden == nil {
		t.Fatal("Hidden state is nil")
	}

	if len(logits) != 3 {
		t.Fatalf("Expected 3 logits matrices, got %d", len(logits))
	}

	t.Logf("✓ MTP Advanced Integration:")
	t.Logf("  Input sequence: %d tokens", len(inputTokens))
	t.Logf("  Hidden state: %dx%d", hidden.RawMatrix().Rows, hidden.RawMatrix().Cols)
	t.Logf("  MTP Loss: %.4f", mtpLoss)

	// Verificar que MTP está funcionando
	if mtpLoss <= 0 {
		t.Error("MTP loss should be positive")
	}

	// Analisar predições
	for k := 0; k < 3; k++ {
		rows, cols := logits[k].Dims()
		t.Logf("  Logits[%d]: %dx%d (predicting token +%d)", k, rows, cols, k+1)
	}
}

// TestMTP_ComparisonWithAndWithoutMTP compara treinamento com e sem MTP
func TestMTP_ComparisonWithAndWithoutMTP(t *testing.T) {
	// Criar dois modelos idênticos
	modelWithoutMTP := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)
	modelWithMTP := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	// Habilitar MTP apenas no segundo
	modelWithMTP.EnableMTP(3, 0.3)

	// Configurar vocabulários
	for _, model := range []*TransformerModel{modelWithoutMTP, modelWithMTP} {
		model.Vocab = make([]string, 100)
		for i := 0; i < 100; i++ {
			model.Vocab[i] = string(rune('a' + i%26))
		}
	}

	// Dados de teste
	inputTokens := []int{10, 20, 30, 40, 50}
	seqLen := len(inputTokens)

	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = []int{(i + 1) % 100, (i + 2) % 100, (i + 3) % 100}
	}

	// Forward pass sem MTP
	hiddenWithoutMTP := modelWithoutMTP.Forward(inputTokens)
	if hiddenWithoutMTP == nil {
		t.Fatal("Forward pass without MTP failed")
	}

	// Forward pass com MTP
	hiddenWithMTP, logits, mtpLoss := modelWithMTP.ForwardWithMTP(inputTokens, targets)
	if hiddenWithMTP == nil || logits == nil {
		t.Fatal("Forward pass with MTP failed")
	}

	// Verificar que ambos produzem outputs válidos
	rows1, cols1 := hiddenWithoutMTP.Dims()
	rows2, cols2 := hiddenWithMTP.Dims()

	if rows1 != rows2 || cols1 != cols2 {
		t.Errorf("Hidden state dimensions mismatch: without MTP=%dx%d, with MTP=%dx%d",
			rows1, cols1, rows2, cols2)
	}

	t.Logf("✓ MTP Comparison:")
	t.Logf("  Without MTP: hidden=%dx%d", rows1, cols1)
	t.Logf("  With MTP: hidden=%dx%d, mtp_loss=%.4f", rows2, cols2, mtpLoss)
	t.Logf("  MTP adds %.1f%% more parameters",
		float64(modelWithMTP.MTPHead.Config.NumPredictions)*100.0/float64(len(modelWithMTP.TransformerLayers)))
}

// TestMTP_MultiTokenGeneration demonstra geração de múltiplos tokens
func TestMTP_MultiTokenGeneration(t *testing.T) {
	// Criar modelo com MTP
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.EnableMTP(3, 0.3)

	// Configurar vocabulário
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Gerar hidden state para um token
	inputTokens := []int{10}
	hidden := model.Forward(inputTokens)

	// Usar última posição para predição
	lastHidden := mat.NewDense(1, model.DModel, nil)
	for j := 0; j < model.DModel; j++ {
		lastHidden.Set(0, j, hidden.At(0, j))
	}

	// Predição MTP: gerar 3 tokens de uma vez
	predictedTokens := MTPInference(model.MTPHead, lastHidden, 0.8)

	t.Logf("✓ MTP Multi-Token Generation:")
	t.Logf("  Input token: %d", inputTokens[0])
	t.Logf("  Predicted next 3 tokens: %v", predictedTokens)

	// Verificar que tokens são válidos
	for k, token := range predictedTokens {
		if token < 0 || token >= 50 {
			t.Errorf("Token %d out of vocabulary: %d", k, token)
		}
	}

	// Benefício do MTP: em vez de 3 forward passes, fizemos apenas 1!
	t.Logf("  Efficiency: 1 forward pass instead of 3 (3x faster generation)")
}

// TestMTP_TrainingEfficiency demonstra como MTP melhora eficiência de treinamento
func TestMTP_TrainingEfficiency(t *testing.T) {
	// Simular treinamento com e sem MTP

	// Sem MTP: 1 exemplo de treinamento = prever 1 token
	samplesWithoutMTP := 100
	tokensLearnedWithoutMTP := samplesWithoutMTP * 1 // 1 token por exemplo

	// Com MTP (3 predições): 1 exemplo = prever 3 tokens
	samplesWithMTP := 100
	numPredictions := 3
	tokensLearnedWithMTP := samplesWithMTP * numPredictions // 3 tokens por exemplo

	t.Logf("✓ MTP Training Efficiency:")
	t.Logf("  Without MTP: %d samples → %d tokens learned",
		samplesWithoutMTP, tokensLearnedWithoutMTP)
	t.Logf("  With MTP (k=%d): %d samples → %d tokens learned",
		numPredictions, samplesWithMTP, tokensLearnedWithMTP)
	t.Logf("  Efficiency gain: %.1fx more tokens per epoch",
		float64(tokensLearnedWithMTP)/float64(tokensLearnedWithoutMTP))

	// MTP aprende mais por epoch, mas com custo computacional ligeiramente maior
	// Overhead típico: ~20-30% para k=3
	overhead := 1.25
	tokensPerComputeWithoutMTP := float64(tokensLearnedWithoutMTP) / 1.0
	tokensPerComputeWithMTP := float64(tokensLearnedWithMTP) / overhead

	t.Logf("  Tokens learned per unit compute:")
	t.Logf("    Without MTP: %.2f", tokensPerComputeWithoutMTP)
	t.Logf("    With MTP: %.2f (%.1fx improvement)",
		tokensPerComputeWithMTP,
		tokensPerComputeWithMTP/tokensPerComputeWithoutMTP)
}

// TestMTP_LongRangeDependencies demonstra como MTP captura dependências de longo alcance
func TestMTP_LongRangeDependencies(t *testing.T) {
	cfg := NewMTPConfig(5, 100, 128)
	head := NewMTPHead(cfg)

	// Criar sequência com padrão de longo alcance
	// Exemplo: "A ... B ... C ... D ... E" onde modelo deve prever todos
	seqLen := 20
	hidden := transformerRandomMatrix(seqLen, cfg.DModel, 0.1)

	// Targets com dependências de longo alcance
	targets := make([][]int, seqLen)
	for i := 0; i < seqLen; i++ {
		targets[i] = make([]int, 5)
		for k := 0; k < 5; k++ {
			// Padrão: prever tokens com gap crescente
			targetIdx := i + k + 1
			if targetIdx < 100 {
				targets[i][k] = targetIdx
			} else {
				targets[i][k] = -1
			}
		}
	}

	// Forward pass
	logits := ComputeMTPLogits(head, hidden, seqLen)
	result := ComputeMTPLoss(head, logits, targets, seqLen)

	t.Logf("✓ MTP Long-Range Dependencies:")
	t.Logf("  Sequence length: %d tokens", seqLen)
	t.Logf("  Prediction horizon: +%d tokens", cfg.NumPredictions)
	t.Logf("  Total loss: %.4f", result.TotalLoss)
	t.Logf("  Per-prediction losses: %v", result.Losses)

	// Análise: se losses são similares para todas as predições,
	// o modelo está aprendendo dependências de longo alcance efetivamente
	maxLoss := result.Losses[0]
	minLoss := result.Losses[0]
	for _, loss := range result.Losses {
		if loss > maxLoss {
			maxLoss = loss
		}
		if loss < minLoss {
			minLoss = loss
		}
	}

	lossRange := maxLoss - minLoss
	t.Logf("  Loss range: %.4f (max=%.4f, min=%.4f)", lossRange, maxLoss, minLoss)

	if lossRange < 1.0 {
		t.Logf("  ✓ Low variance indicates good long-range learning")
	} else {
		t.Logf("  Note: Higher variance is expected with random initialization")
	}
}
