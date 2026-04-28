package model

import (
	"testing"
)

func TestMLA_MemoryReduction(t *testing.T) {
	tests := []struct {
		name              string
		dModel            int
		latentDim         int
		expectedReduction float64
	}{
		{
			"d_model=512, latent=128 (87.5% reduction)",
			512,
			128,
			87.5,
		},
		{
			"d_model=512, latent=64 (93.75% reduction)",
			512,
			64,
			93.75,
		},
		{
			"d_model=256, latent=64 (87.5% reduction)",
			256,
			64,
			87.5,
		},
		{
			"d_model=1024, latent=256 (87.5% reduction)",
			1024,
			256,
			87.5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetMemoryReduction(tt.dModel, tt.latentDim)

			originalMemory := result["original_kv_memory"].(int)
			mlaMemory := result["mla_latent_memory"].(int)
			reduction := result["reduction_percent"].(float64)

			// Verificar cálculos
			expectedOriginal := 2 * tt.dModel
			if originalMemory != expectedOriginal {
				t.Errorf("Original memory: expected %d, got %d", expectedOriginal, originalMemory)
			}

			if mlaMemory != tt.latentDim {
				t.Errorf("MLA memory: expected %d, got %d", tt.latentDim, mlaMemory)
			}

			if reduction != tt.expectedReduction {
				t.Errorf("Reduction: expected %.1f%%, got %.1f%%", tt.expectedReduction, reduction)
			}

			t.Logf("✓ Memory reduction: %.1f%% (%d → %d)", reduction, originalMemory, mlaMemory)
		})
	}
}

func TestMLA_NewMLAConfig(t *testing.T) {
	tests := []struct {
		name           string
		dModel         int
		nHeads         int
		expectedLatent int
	}{
		{"d_model=512, 8 heads", 512, 8, 128},
		{"d_model=256, 4 heads", 256, 4, 64},
		{"d_model=128, 4 heads", 128, 4, 64}, // Minimum 64
		{"d_model=1024, 16 heads", 1024, 16, 256},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := NewMLAConfig(tt.dModel, tt.nHeads)

			if cfg.DModel != tt.dModel {
				t.Errorf("DModel: expected %d, got %d", tt.dModel, cfg.DModel)
			}

			if cfg.NHeads != tt.nHeads {
				t.Errorf("NHeads: expected %d, got %d", tt.nHeads, cfg.NHeads)
			}

			if cfg.LatentDim != tt.expectedLatent {
				t.Errorf("LatentDim: expected %d, got %d", tt.expectedLatent, cfg.LatentDim)
			}

			expectedHeadDim := tt.dModel / tt.nHeads
			if cfg.HeadDim != expectedHeadDim {
				t.Errorf("HeadDim: expected %d, got %d", expectedHeadDim, cfg.HeadDim)
			}

			t.Logf("✓ Config: d_model=%d, heads=%d, latent=%d, head_dim=%d",
				cfg.DModel, cfg.NHeads, cfg.LatentDim, cfg.HeadDim)
		})
	}
}

func TestMLA_LayerCreation(t *testing.T) {
	cfg := NewMLAConfig(256, 4)

	layer := NewMLALayer(cfg)

	if layer == nil {
		t.Fatal("Layer creation failed")
	}

	// Verificar dimensões
	if layer.LatentDim != cfg.LatentDim {
		t.Errorf("LatentDim: expected %d, got %d", cfg.LatentDim, layer.LatentDim)
	}

	// Verificar que pesos foram inicializados
	if layer.WQ == nil {
		t.Error("WQ not initialized")
	}
	if layer.WKVDown == nil {
		t.Error("WKVDown not initialized")
	}
	if layer.WKVUp == nil {
		t.Error("WKVUp not initialized")
	}
	if layer.WO == nil {
		t.Error("WO not initialized")
	}

	// Verificar shapes
	qRows, qCols := layer.WQ.Dims()
	if qRows != cfg.DModel || qCols != cfg.DModel {
		t.Errorf("WQ shape: expected [%d, %d], got [%d, %d]", cfg.DModel, cfg.DModel, qRows, qCols)
	}

	kvDownRows, kvDownCols := layer.WKVDown.Dims()
	if kvDownRows != cfg.LatentDim || kvDownCols != cfg.DModel {
		t.Errorf("WKVDown shape: expected [%d, %d], got [%d, %d]", cfg.LatentDim, cfg.DModel, kvDownRows, kvDownCols)
	}

	kvUpRows, kvUpCols := layer.WKVUp.Dims()
	if kvUpRows != 2*cfg.DModel || kvUpCols != cfg.LatentDim {
		t.Errorf("WKVUp shape: expected [%d, %d], got [%d, %d]", 2*cfg.DModel, cfg.LatentDim, kvUpRows, kvUpCols)
	}

	t.Logf("✓ Layer created successfully")
	t.Logf("  WQ: %dx%d", qRows, qCols)
	t.Logf("  WKVDown: %dx%d", kvDownRows, kvDownCols)
	t.Logf("  WKVUp: %dx%d", kvUpRows, kvUpCols)
	t.Logf("  Latent dim: %d", layer.LatentDim)
}

func TestMLA_ComparisonWithStandard(t *testing.T) {
	// Comparar memória entre atenção standard e MLA
	dModel := 512
	nHeads := 8

	// Standard attention: K + V = 2 × d_model
	standardKV := 2 * dModel

	// MLA: latent_dim apenas
	cfg := NewMLAConfig(dModel, nHeads)
	mlaKV := cfg.LatentDim

	reduction := float64(standardKV-mlaKV) / float64(standardKV) * 100

	t.Logf("Standard Attention KV Cache: %d", standardKV)
	t.Logf("MLA Latent Cache: %d", mlaKV)
	t.Logf("Memory Reduction: %.1f%%", reduction)
	t.Logf("Speedup Factor: %.2fx", float64(standardKV)/float64(mlaKV))

	// Verificar que MLA usa menos memória
	if mlaKV >= standardKV {
		t.Errorf("MLA should use less memory: %d >= %d", mlaKV, standardKV)
	}

	// Verificar redução significativa (>50%)
	if reduction < 50.0 {
		t.Errorf("MLA should reduce memory by at least 50%%, got %.1f%%", reduction)
	}
}

func TestMLA_Scalability(t *testing.T) {
	// Testar como MLA escala com diferentes tamanhos de modelo
	modelSizes := []int{128, 256, 512, 1024, 2048}

	t.Log("MLA Scalability Analysis:")
	t.Log("=========================")

	for _, dModel := range modelSizes {
		cfg := NewMLAConfig(dModel, 8)
		reduction := GetMemoryReduction(dModel, cfg.LatentDim)

		originalMemory := reduction["original_kv_memory"].(int)
		mlaMemory := reduction["mla_latent_memory"].(int)
		percentReduction := reduction["reduction_percent"].(float64)

		t.Logf("d_model=%4d | Standard: %5d | MLA: %4d | Reduction: %.1f%% | Speedup: %.2fx",
			dModel, originalMemory, mlaMemory, percentReduction,
			float64(originalMemory)/float64(mlaMemory))
	}
}

func TestMLA_InferenceMemory(t *testing.T) {
	// Calcular memória total para inferência com sequência longa
	seqLen := 256
	dModel := 512
	nHeads := 8

	// Standard attention
	// Cache: seq_len × seq_len × (K + V)
	standardCache := seqLen * seqLen * 2 * dModel

	// MLA
	// Cache: seq_len × seq_len × latent (muito menor!)
	cfg := NewMLAConfig(dModel, nHeads)
	mlaCache := seqLen * seqLen * cfg.LatentDim

	reduction := float64(standardCache-mlaCache) / float64(standardCache) * 100

	t.Logf("Inference Memory Comparison (seq_len=%d, d_model=%d):", seqLen, dModel)
	t.Logf("Standard KV Cache: %d floats (%.2f MB)", standardCache, float64(standardCache)*8/1024/1024)
	t.Logf("MLA Latent Cache:  %d floats (%.2f MB)", mlaCache, float64(mlaCache)*8/1024/1024)
	t.Logf("Memory Saved: %d floats (%.2f MB)", standardCache-mlaCache, float64(standardCache-mlaCache)*8/1024/1024)
	t.Logf("Reduction: %.1f%%", reduction)
	t.Logf("Speedup: %.2fx", float64(standardCache)/float64(mlaCache))

	// Verificar redução significativa
	if reduction < 50.0 {
		t.Errorf("Expected >50%% reduction, got %.1f%%", reduction)
	}
}

func TestMLA_ConfigValidation(t *testing.T) {
	// Testar que configuração mínima é respeitada
	cfg := NewMLAConfig(64, 4)

	// Latent dim deve ser pelo menos 64
	if cfg.LatentDim < 64 {
		t.Errorf("LatentDim should be at least 64, got %d", cfg.LatentDim)
	}

	t.Logf("✓ Minimum latent dimension enforced: %d", cfg.LatentDim)
}
