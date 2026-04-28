package model

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// MLALayer representa uma camada com Multi-head Latent Attention
// MLA comprime K e V em um vetor latente de dimensão reduzida
type MLALayer struct {
	// Q projection (mantém dimensão completa)
	WQ *mat.Dense // [d_model, d_model]

	// Low-rank KV projection (compressão)
	WKVDown *mat.Dense // [latent_dim, d_model] - Comprime para latent_dim
	WKVUp   *mat.Dense // [2*d_model, latent_dim] - Expande de volta para K e V

	// Output projection
	WO *mat.Dense // [d_model, d_model]

	// Dimensão latente (tipicamente d_model/4 ou d_model/8)
	LatentDim int

	// Feed-forward (igual ao Transformer normal)
	W1 *mat.Dense // [ff_hidden, d_model]
	B1 *mat.Dense // [ff_hidden, 1]
	W2 *mat.Dense // [d_model, ff_hidden]
	B2 *mat.Dense // [d_model, 1]

	// Layer normalization
	LN1Weight, LN1Bias *mat.Dense
	LN2Weight, LN2Bias *mat.Dense

	// Gradients
	GradWQ, GradWKVDown, GradWKVUp *mat.Dense
	GradWO                         *mat.Dense
	GradW1, GradB1                 *mat.Dense
	GradW2, GradB2                 *mat.Dense
	GradLN1Weight, GradLN1Bias     *mat.Dense
	GradLN2Weight, GradLN2Bias     *mat.Dense
}

// MLAConfig configurações para MLA
type MLAConfig struct {
	DModel    int // Dimensão do modelo
	NHeads    int // Número de heads
	LatentDim int // Dimensão latente (compressão KV)
	FFHidden  int // Hidden size do feed-forward
	HeadDim   int // Dimensão de cada head
}

// NewMLAConfig cria configuração padrão para MLA
func NewMLAConfig(dModel, nHeads int) *MLAConfig {
	// Latent dimension: tipicamente d_model/4 para 75% de redução
	latentDim := dModel / 4
	if latentDim < 64 {
		latentDim = 64 // Mínimo de 64
	}

	return &MLAConfig{
		DModel:    dModel,
		NHeads:    nHeads,
		LatentDim: latentDim,
		FFHidden:  dModel * 2,
		HeadDim:   dModel / nHeads,
	}
}

// NewMLALayer cria uma nova camada MLA
func NewMLALayer(cfg *MLAConfig) *MLALayer {
	layer := &MLALayer{
		LatentDim: cfg.LatentDim,
	}

	// Inicializar pesos com Xavier/Glorot
	scaleQ := math.Sqrt(2.0 / float64(cfg.DModel+cfg.DModel))
	layer.WQ = transformerRandomMatrix(cfg.DModel, cfg.DModel, scaleQ)

	// Low-rank KV: compressão
	scaleDown := math.Sqrt(2.0 / float64(cfg.DModel+cfg.LatentDim))
	layer.WKVDown = transformerRandomMatrix(cfg.LatentDim, cfg.DModel, scaleDown)

	// Low-rank KV: expansão (para 2*d_model porque K e V concatenados)
	scaleUp := math.Sqrt(2.0 / float64(cfg.LatentDim+2*cfg.DModel))
	layer.WKVUp = transformerRandomMatrix(2*cfg.DModel, cfg.LatentDim, scaleUp)

	// Output projection
	scaleO := math.Sqrt(2.0 / float64(cfg.DModel+cfg.DModel))
	layer.WO = transformerRandomMatrix(cfg.DModel, cfg.DModel, scaleO)

	// Feed-forward
	scale1 := math.Sqrt(2.0 / float64(cfg.DModel+cfg.FFHidden))
	layer.W1 = transformerRandomMatrix(cfg.FFHidden, cfg.DModel, scale1)
	layer.B1 = mat.NewDense(cfg.FFHidden, 1, make([]float64, cfg.FFHidden))

	scale2 := math.Sqrt(2.0 / float64(cfg.FFHidden+cfg.DModel))
	layer.W2 = transformerRandomMatrix(cfg.DModel, cfg.FFHidden, scale2)
	layer.B2 = mat.NewDense(cfg.DModel, 1, make([]float64, cfg.DModel))

	// Layer normalization
	layer.LN1Weight = mat.NewDense(cfg.DModel, 1, onesVector(cfg.DModel))
	layer.LN1Bias = mat.NewDense(cfg.DModel, 1, make([]float64, cfg.DModel))
	layer.LN2Weight = mat.NewDense(cfg.DModel, 1, onesVector(cfg.DModel))
	layer.LN2Bias = mat.NewDense(cfg.DModel, 1, make([]float64, cfg.DModel))

	return layer
}

// multiHeadLatentAttention implementa MLA com compressão KV
// Reduz memória de O(seq_len² × d_model) para O(seq_len² × latent_dim)
func multiHeadLatentAttention(layer *MLALayer, X *mat.Dense, seqLen, dModel, nHeads int) *mat.Dense {
	headDim := dModel / nHeads
	latentDim := layer.LatentDim

	// Q projection (dimensão completa)
	Q := mat.NewDense(seqLen, dModel, nil)
	Q.Mul(X, layer.WQ)

	// KV compression: X → latent
	KVLatent := mat.NewDense(seqLen, latentDim, nil)
	KVLatent.Mul(X, layer.WKVDown.T())

	// KV decompression: latent → K, V concatenados
	KV := mat.NewDense(seqLen, 2*dModel, nil)
	KV.Mul(KVLatent, layer.WKVUp.T())

	// Separar K e V
	K := mat.NewDense(seqLen, dModel, nil)
	V := mat.NewDense(seqLen, dModel, nil)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			K.Set(i, j, KV.At(i, j))
			V.Set(i, j, KV.At(i, j+dModel))
		}
	}

	// Calcular atenção para cada head em paralelo
	type headResult struct {
		index  int
		output *mat.Dense
	}

	results := make(chan headResult, nHeads)
	var wg sync.WaitGroup

	// Lançar goroutines para cada head
	for h := 0; h < nHeads; h++ {
		wg.Add(1)
		go func(headIdx int) {
			defer wg.Done()

			// Extrair fatias do head
			headStart := headIdx * headDim

			// Extrair colunas para Q, K, V deste head
			QHead := mat.NewDense(seqLen, headDim, nil)
			KHead := mat.NewDense(seqLen, headDim, nil)
			VHead := mat.NewDense(seqLen, headDim, nil)

			for i := 0; i < seqLen; i++ {
				for j := 0; j < headDim; j++ {
					QHead.Set(i, j, Q.At(i, headStart+j))
					KHead.Set(i, j, K.At(i, headStart+j))
					VHead.Set(i, j, V.At(i, headStart+j))
				}
			}

			// Scaled dot-product attention
			KTHead := KHead.T()
			scores := mat.NewDense(seqLen, seqLen, nil)
			scores.Mul(QHead, KTHead)

			// Scale
			scale := 1.0 / math.Sqrt(float64(headDim))
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					scores.Set(i, j, scores.At(i, j)*scale)
				}
			}

			// Softmax por linha
			for i := 0; i < seqLen; i++ {
				row := make([]float64, seqLen)
				for j := 0; j < seqLen; j++ {
					row[j] = scores.At(i, j)
				}
				row = transformerSoftmax(row)
				for j := 0; j < seqLen; j++ {
					scores.Set(i, j, row[j])
				}
			}

			// Weighted sum: scores * V
			headOutput := mat.NewDense(seqLen, headDim, nil)
			headOutput.Mul(scores, VHead)

			results <- headResult{headIdx, headOutput}
		}(h)
	}

	// Fechar channel quando todas goroutines terminarem
	go func() {
		wg.Wait()
		close(results)
	}()

	// Concatenar outputs dos heads
	headOutputs := make([]*mat.Dense, nHeads)
	for result := range results {
		headOutputs[result.index] = result.output
	}

	// Concatenar todos os heads
	output := mat.NewDense(seqLen, dModel, nil)
	for h := 0; h < nHeads; h++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < headDim; j++ {
				output.Set(i, h*headDim+j, headOutputs[h].At(i, j))
			}
		}
	}

	// Output projection
	finalOutput := mat.NewDense(seqLen, dModel, nil)
	finalOutput.Mul(output, layer.WO.T())

	return finalOutput
}

// mlaForward faz forward pass completo da camada MLA
func mlaForward(layer *MLALayer, X *mat.Dense, seqLen, dModel, nHeads int) *mat.Dense {
	// ===== SUB-LAYER 1: Multi-Head Latent Attention =====
	XResidual := mat.NewDense(seqLen, dModel, nil)
	XResidual.CloneFrom(X)

	// MLA attention
	X = multiHeadLatentAttention(layer, X, seqLen, dModel, nHeads)

	// Residual Connection + Layer Norm
	X = addResidualAndNorm(X, XResidual, seqLen, dModel)

	// ===== SUB-LAYER 2: Feed-Forward =====
	XResidual2 := mat.NewDense(seqLen, dModel, nil)
	XResidual2.CloneFrom(X)

	// FF: W2(ReLU(W1(X) + B1)) + B2
	temp := mat.NewDense(seqLen, layer.W1.RawMatrix().Rows, nil)
	temp.Mul(X, layer.W1.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < layer.B1.RawMatrix().Rows; j++ {
			val := temp.At(i, j) + layer.B1.At(j, 0)
			temp.Set(i, j, val)
		}
	}

	// ReLU activation
	for i := 0; i < seqLen; i++ {
		for j := 0; j < temp.RawMatrix().Cols; j++ {
			val := temp.At(i, j)
			if val < 0 {
				temp.Set(i, j, 0)
			}
		}
	}

	// W2 projection
	X = mat.NewDense(seqLen, dModel, nil)
	X.Mul(temp, layer.W2.T())

	// Add bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < layer.B2.RawMatrix().Rows; j++ {
			val := X.At(i, j) + layer.B2.At(j, 0)
			X.Set(i, j, val)
		}
	}

	// Residual Connection + Layer Norm
	X = addResidualAndNorm(X, XResidual2, seqLen, dModel)

	return X
}

// GetMemoryReduction calcula a redução de memória do MLA
func GetMemoryReduction(dModel, latentDim int) map[string]interface{} {
	// Memória original: K + V = 2 × d_model
	originalMemory := 2 * dModel

	// Memória MLA: latent_dim + 2 × d_model (mas latent é muito menor)
	// Na prática, o cache é só o latent: latent_dim
	mlaMemory := latentDim

	reduction := float64(originalMemory-mlaMemory) / float64(originalMemory) * 100

	return map[string]interface{}{
		"original_kv_memory": originalMemory,
		"mla_latent_memory":  mlaMemory,
		"memory_saved":       originalMemory - mlaMemory,
		"reduction_percent":  reduction,
	}
}
