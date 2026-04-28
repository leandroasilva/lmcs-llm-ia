package model

import (
	"testing"
)

func TestMoE_TransformerIntegration(t *testing.T) {
	// Criar modelo Transformer pequeno
	vocabSize := 100
	dModel := 128
	nHeads := 4
	nLayers := 2
	maxSeqLen := 64
	ffHidden := 256
	learningRate := 0.001
	dropoutRate := 0.1
	weightDecay := 0.01

	model := NewTransformerModel(
		vocabSize, dModel, nHeads, nLayers, maxSeqLen,
		ffHidden, learningRate, dropoutRate, weightDecay,
	)

	// Verificar que MoE está desabilitado por padrão
	for i, layer := range model.TransformerLayers {
		if layer.UseMoE {
			t.Errorf("Layer %d: MoE should be disabled by default", i)
		}
		if layer.MoELayer != nil {
			t.Errorf("Layer %d: MoELayer should be nil by default", i)
		}
	}

	t.Logf("✓ Transformer created with %d layers (MoE disabled by default)", nLayers)
}

func TestMoE_EnableMoEForSingleLayer(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(100, 128, 4, 3, 64, 256, 0.001, 0.1, 0.01)

	// Habilitar MoE apenas na camada 1
	moeConfig := NewMoEConfig(128, 4, 2, 256)
	model.EnableMoEForLayer(1, moeConfig)

	// Verificar
	if model.TransformerLayers[0].UseMoE {
		t.Error("Layer 0: MoE should still be disabled")
	}
	if !model.TransformerLayers[1].UseMoE {
		t.Error("Layer 1: MoE should be enabled")
	}
	if model.TransformerLayers[1].MoELayer == nil {
		t.Fatal("Layer 1: MoELayer should not be nil")
	}
	if len(model.TransformerLayers[1].MoELayer.Experts) != 4 {
		t.Errorf("Layer 1: Expected 4 experts, got %d", len(model.TransformerLayers[1].MoELayer.Experts))
	}
	if model.TransformerLayers[2].UseMoE {
		t.Error("Layer 2: MoE should still be disabled")
	}

	t.Logf("✓ MoE enabled for layer 1 only (4 experts, top-k=2)")
}

func TestMoE_EnableMoEForAllLayers(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(100, 128, 4, 3, 64, 256, 0.001, 0.1, 0.01)

	// Habilitar MoE em todas as camadas
	model.EnableMoEForAllLayers(8, 2)

	// Verificar que todas as camadas têm MoE
	for i, layer := range model.TransformerLayers {
		if !layer.UseMoE {
			t.Errorf("Layer %d: MoE should be enabled", i)
		}
		if layer.MoELayer == nil {
			t.Errorf("Layer %d: MoELayer should not be nil", i)
		}
		if len(layer.MoELayer.Experts) != 8 {
			t.Errorf("Layer %d: Expected 8 experts, got %d", i, len(layer.MoELayer.Experts))
		}
	}

	t.Logf("✓ MoE enabled for all %d layers (8 experts each, top-k=2)", len(model.TransformerLayers))
}

func TestMoE_TransformerForwardWithMoE(t *testing.T) {
	// Criar modelo com BPE tokenizer
	model := NewTransformerModel(100, 128, 4, 2, 64, 256, 0.001, 0.1, 0.01)

	// Habilitar MoE
	model.EnableMoEForAllLayers(4, 2)

	// Configurar tokenizer básico
	model.Vocab = make([]string, 100)
	model.WordToID = make(map[string]int)
	model.IDToWord = make(map[int]string)
	for i := 0; i < 100; i++ {
		word := string(rune('a' + i%26))
		model.Vocab[i] = word
		model.WordToID[word] = i
		model.IDToWord[i] = word
	}
	model.SpecialTokens = map[string]int{
		"<PAD>": 0,
		"<UNK>": 1,
	}

	// Criar input de teste
	input := []int{10, 20, 30, 40, 50}

	// Forward pass
	output := model.Forward(input)

	// Verificar output
	rows, cols := output.Dims()
	if rows != len(input) {
		t.Errorf("Output rows: expected %d, got %d", len(input), rows)
	}

	// Output do Forward é [seq_len, d_model], não vocab_size
	if cols != model.DModel {
		t.Errorf("Output cols: expected %d (d_model), got %d", model.DModel, cols)
	}

	t.Logf("✓ Transformer forward pass with MoE: input=%d tokens, output=%dx%d", len(input), rows, cols)
}

func TestMoE_HybridArchitecture(t *testing.T) {
	// Criar modelo com 4 camadas
	model := NewTransformerModel(100, 256, 8, 4, 128, 512, 0.001, 0.1, 0.01)

	// Configurar arquitetura híbrida:
	// Camadas 0, 1: FFN denso (mais rápido)
	// Camadas 2, 3: MoE (mais capacidade)

	moeConfig := NewMoEConfig(256, 8, 2, 512)
	model.EnableMoEForLayer(2, moeConfig)
	model.EnableMoEForLayer(3, NewMoEConfig(256, 8, 2, 512))

	// Verificar configuração híbrida
	denseCount := 0
	moeCount := 0

	for i, layer := range model.TransformerLayers {
		if layer.UseMoE {
			moeCount++
			t.Logf("Layer %d: MoE (8 experts)", i)
		} else {
			denseCount++
			t.Logf("Layer %d: Dense FFN", i)
		}
	}

	if denseCount != 2 || moeCount != 2 {
		t.Errorf("Hybrid architecture: expected 2 dense + 2 MoE, got %d dense + %d MoE", denseCount, moeCount)
	}

	t.Logf("✓ Hybrid architecture: %d dense layers + %d MoE layers", denseCount, moeCount)

	// Calcular parâmetros totais
	totalParams := 0
	for _, layer := range model.TransformerLayers {
		if layer.UseMoE {
			params := GetMoEParameterCount(layer.MoELayer.Config)
			totalParams += params["total_params"].(int)
		} else {
			// FFN denso: 2 * d_model * ff_hidden
			params := 2 * model.DModel * model.FFHidden
			totalParams += params
		}
	}

	t.Logf("✓ Total parameters (all MoE layers): %d (%.2f M)", totalParams, float64(totalParams)/1e6)
}
