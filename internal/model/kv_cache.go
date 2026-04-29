package model

import (
	"math"
	"sync"

	"gonum.org/v1/gonum/mat"
)

// KVCache armazena Key-Value cache para geração incremental
// Evita recomputação de K, V para tokens anteriores
type KVCache struct {
	// Cache por camada: [nLayers]
	Layers []LayerKVCache

	// Posição atual na sequência
	Position int

	// Tamanho máximo da sequência
	MaxSeqLen int

	// Configurações
	NHeads  int
	HeadDim int
}

// LayerKVCache armazena K, V para uma camada específica
type LayerKVCache struct {
	// Keys: [seq_len, n_heads, head_dim]
	Keys [][][]float64

	// Values: [seq_len, n_heads, head_dim]
	Values [][][]float64

	// Número de posições preenchidas
	Filled int
}

// NewKVCache cria um novo KV cache
func NewKVCache(nLayers, nHeads, headDim, maxSeqLen int) *KVCache {
	cache := &KVCache{
		Layers:    make([]LayerKVCache, nLayers),
		Position:  0,
		MaxSeqLen: maxSeqLen,
		NHeads:    nHeads,
		HeadDim:   headDim,
	}

	// Inicializar cache para cada camada
	for i := 0; i < nLayers; i++ {
		cache.Layers[i] = LayerKVCache{
			Keys:   make([][][]float64, maxSeqLen),
			Values: make([][][]float64, maxSeqLen),
			Filled: 0,
		}

		// Pré-alocar slices para cada posição
		for pos := 0; pos < maxSeqLen; pos++ {
			cache.Layers[i].Keys[pos] = make([][]float64, nHeads)
			cache.Layers[i].Values[pos] = make([][]float64, nHeads)

			for h := 0; h < nHeads; h++ {
				cache.Layers[i].Keys[pos][h] = make([]float64, headDim)
				cache.Layers[i].Values[pos][h] = make([]float64, headDim)
			}
		}
	}

	return cache
}

// Reset reseta o cache para nova geração
func (c *KVCache) Reset() {
	c.Position = 0
	for i := range c.Layers {
		c.Layers[i].Filled = 0
	}
}

// Store armazena K, V para uma posição específica
func (c *KVCache) Store(layerIdx int, keys, values *mat.Dense, position int) {
	if layerIdx >= len(c.Layers) || position >= c.MaxSeqLen {
		return
	}

	layer := &c.Layers[layerIdx]

	// Verificar se a posição é válida
	if position < 0 || position >= c.MaxSeqLen {
		return
	}

	// Extrair K, V da matriz e armazenar
	for h := 0; h < c.NHeads; h++ {
		headStart := h * c.HeadDim
		for j := 0; j < c.HeadDim; j++ {
			// Verificar bounds
			if position < len(layer.Keys) && h < len(layer.Keys[position]) {
				layer.Keys[position][h][j] = keys.At(0, headStart+j)
				layer.Values[position][h][j] = values.At(0, headStart+j)
			}
		}
	}

	// Atualizar contador
	if position >= layer.Filled {
		layer.Filled = position + 1
	}
}

// GetCached retorna todos os K, V armazenados até a posição atual
func (c *KVCache) GetCached(layerIdx int) (*mat.Dense, *mat.Dense, int) {
	if layerIdx >= len(c.Layers) {
		return nil, nil, 0
	}

	layer := &c.Layers[layerIdx]
	seqLen := layer.Filled

	// Criar matrizes para K, V concatenados
	K := mat.NewDense(seqLen, c.NHeads*c.HeadDim, nil)
	V := mat.NewDense(seqLen, c.NHeads*c.HeadDim, nil)

	// Preencher com dados do cache
	for pos := 0; pos < seqLen && pos < layer.Filled; pos++ {
		for h := 0; h < c.NHeads; h++ {
			headStart := h * c.HeadDim
			for j := 0; j < c.HeadDim; j++ {
				K.Set(pos, headStart+j, layer.Keys[pos][h][j])
				V.Set(pos, headStart+j, layer.Values[pos][h][j])
			}
		}
	}

	return K, V, seqLen
}

// IncrementalForward realiza forward pass incremental com KV cache
// Apenas processa o último token, reusa K, V dos tokens anteriores
func (m *TransformerModel) IncrementalForward(tokenID int, cache *KVCache, position int) []float64 {
	// Obter embedding do token atual
	tokenEmb := make([]float64, m.DModel)
	for j := 0; j < m.DModel; j++ {
		tokenEmb[j] = m.TokenEmbedding.At(tokenID, j) + m.PositionEmbedding.At(position, j)
	}

	// X contém apenas o token atual [1, d_model]
	X := mat.NewDense(1, m.DModel, tokenEmb)

	// Passar por cada camada transformer
	for i := 0; i < m.NLayers; i++ {
		X = transformerLayerForwardIncremental(&m.TransformerLayers[i], X, cache, i, position, m.DModel, m.NHeads)
	}

	// Layer normalization final
	X = applyLayerNorm(X, 1, m.DModel)

	// Retornar hidden state do último token
	result := make([]float64, m.DModel)
	for j := 0; j < m.DModel; j++ {
		result[j] = X.At(0, j)
	}

	return result
}

// transformerLayerForwardIncremental forward pass incremental de uma camada
func transformerLayerForwardIncremental(layer *TransformerLayer, X *mat.Dense, cache *KVCache, layerIdx, position, dModel, nHeads int) *mat.Dense {
	// ===== SUB-LAYER 1: Multi-Head Self-Attention com KV Cache =====
	XAttn := multiHeadAttentionIncremental(layer, X, cache, layerIdx, position, dModel, nHeads)

	// Residual connection + layer norm
	X = addResidualAndNorm(XAttn, X, 1, dModel)

	// ===== SUB-LAYER 2: Feed-Forward =====
	XFF := feedForwardIncremental(layer, X, dModel)

	// Residual connection + layer norm
	X = addResidualAndNorm(XFF, X, 1, dModel)

	return X
}

// multiHeadAttentionIncremental attention com KV cache
func multiHeadAttentionIncremental(layer *TransformerLayer, X *mat.Dense, cache *KVCache, layerIdx, position, dModel, nHeads int) *mat.Dense {
	headDim := dModel / nHeads

	// Calcular Q, K, V apenas para o token atual [1, d_model]
	Q := mat.NewDense(1, dModel, nil)
	K := mat.NewDense(1, dModel, nil)
	V := mat.NewDense(1, dModel, nil)

	Q.Mul(X, layer.WQ)
	K.Mul(X, layer.WK)
	V.Mul(X, layer.WV)

	// Armazenar K, V no cache
	if cache != nil {
		cache.Store(layerIdx, K, V, position)
	}

	// Obter K, V do cache (todos os tokens anteriores + atual)
	var KCached, VCached *mat.Dense
	var cachedSeqLen int

	if cache != nil && cache.Layers[layerIdx].Filled > 0 {
		KCached, VCached, cachedSeqLen = cache.GetCached(layerIdx)
	} else {
		// Fallback: sem cache ou primeiro token, usar apenas token atual
		KCached = K
		VCached = V
		cachedSeqLen = 1
	}

	// Calcular atenção para cada head
	output := mat.NewDense(1, dModel, nil)

	for h := 0; h < nHeads; h++ {
		headStart := h * headDim

		// Extrair Q para este head [1, head_dim]
		QHead := mat.NewDense(1, headDim, nil)
		for j := 0; j < headDim; j++ {
			QHead.Set(0, j, Q.At(0, headStart+j))
		}

		// Extrair K, V para este head [cached_seq_len, head_dim]
		KHead := mat.NewDense(cachedSeqLen, headDim, nil)
		VHead := mat.NewDense(cachedSeqLen, headDim, nil)

		for pos := 0; pos < cachedSeqLen; pos++ {
			for j := 0; j < headDim; j++ {
				KHead.Set(pos, j, KCached.At(pos, headStart+j))
				VHead.Set(pos, j, VCached.At(pos, headStart+j))
			}
		}

		// Scaled dot-product attention
		// Q [1, head_dim] * K^T [head_dim, cached_seq_len] = scores [1, cached_seq_len]
		KTHead := KHead.T()
		scores := mat.NewDense(1, cachedSeqLen, nil)
		scores.Mul(QHead, KTHead)

		// Scale
		scale := 1.0 / math.Sqrt(float64(headDim))
		for j := 0; j < cachedSeqLen; j++ {
			scores.Set(0, j, scores.At(0, j)*scale)
		}

		// Softmax
		row := make([]float64, cachedSeqLen)
		for j := 0; j < cachedSeqLen; j++ {
			row[j] = scores.At(0, j)
		}
		row = transformerSoftmax(row)

		// Weighted sum: scores [1, cached_seq_len] * V [cached_seq_len, head_dim] = output [1, head_dim]
		headOutput := mat.NewDense(1, headDim, nil)
		for j := 0; j < headDim; j++ {
			sum := 0.0
			for pos := 0; pos < cachedSeqLen; pos++ {
				sum += row[pos] * VHead.At(pos, j)
			}
			headOutput.Set(0, j, sum)
		}

		// Colocar no output
		for j := 0; j < headDim; j++ {
			output.Set(0, headStart+j, headOutput.At(0, j))
		}
	}

	// Output projection
	result := mat.NewDense(1, dModel, nil)
	result.Mul(output, layer.WO)

	return result
}

// feedForwardIncremental feed-forward incremental
func feedForwardIncremental(layer *TransformerLayer, X *mat.Dense, dModel int) *mat.Dense {
	// X * W1^T + B1 [1, ff_hidden]
	hidden := mat.NewDense(1, layer.W1.RawMatrix().Rows, nil)
	hidden.Mul(X, layer.W1.T())

	// Add bias
	for j := 0; j < hidden.RawMatrix().Cols; j++ {
		hidden.Set(0, j, hidden.At(0, j)+layer.B1.At(j, 0))
	}

	// ReLU activation
	for j := 0; j < hidden.RawMatrix().Cols; j++ {
		val := hidden.At(0, j)
		if val < 0 {
			hidden.Set(0, j, 0)
		}
	}

	// Hidden * W2^T + B2 [1, d_model]
	output := mat.NewDense(1, dModel, nil)
	output.Mul(hidden, layer.W2.T())

	// Add bias
	for j := 0; j < dModel; j++ {
		output.Set(0, j, output.At(0, j)+layer.B2.At(j, 0))
	}

	return output
}

// GenerateWithKVCache gera texto usando KV cache para speedup
func (m *TransformerModel) GenerateWithKVCache(prompt string, maxTokens int, temperature float64, topK int) string {
	// Tokenizar prompt
	promptTokens := m.Tokenize(prompt)

	// Criar KV cache
	headDim := m.DModel / m.NHeads
	cache := NewKVCache(m.NLayers, m.NHeads, headDim, m.MaxSeqLen)

	// Gerar tokens
	generatedTokens := make([]int, 0, maxTokens)
	currentTokens := make([]int, len(promptTokens))
	copy(currentTokens, promptTokens)

	for i := 0; i < maxTokens; i++ {
		var lastHidden []float64

		if i == 0 {
			// Primeiro token: forward pass completo do prompt e popular cache
			// Processar token por token para popular o cache
			for pos := 0; pos < len(currentTokens); pos++ {
				tokenID := currentTokens[pos]
				lastHidden = m.IncrementalForward(tokenID, cache, pos)
			}
		} else {
			// Tokens subsequentes: forward incremental
			lastToken := currentTokens[len(currentTokens)-1]
			position := len(currentTokens) - 1
			lastHidden = m.IncrementalForward(lastToken, cache, position)
		}

		// Calcular logits
		logits := make([]float64, m.VocabSize)
		for v := 0; v < m.VocabSize; v++ {
			logits[v] = m.BOut.At(v, 0)
			for j := 0; j < m.DModel; j++ {
				logits[v] += m.WOut.At(v, j) * lastHidden[j]
			}
		}

		// Aplicar temperatura
		if temperature > 0 && temperature != 1.0 {
			for j := range logits {
				logits[j] /= temperature
			}
		}

		// Softmax
		probs := Softmax(logits)

		// Sample token
		nextToken := SampleTopK(probs, topK)

		// Verificar token de fim
		if nextToken == m.SpecialTokens["<EOS>"] {
			break
		}

		generatedTokens = append(generatedTokens, nextToken)
		currentTokens = append(currentTokens, nextToken)

		// Limitar tamanho da sequência
		if len(currentTokens) > m.MaxSeqLen {
			currentTokens = currentTokens[1:]
		}
	}

	// Converter para texto
	return m.Detokenize(generatedTokens)
}

// Pool de KV Cache para reutilização (performance)
var kvCachePool = sync.Pool{
	New: func() interface{} {
		return &KVCache{}
	},
}

// GetKVCache obtém KV cache do pool
func GetKVCache(nLayers, nHeads, headDim, maxSeqLen int) *KVCache {
	cache := kvCachePool.Get().(*KVCache)

	// Reinitializar se necessário
	if len(cache.Layers) != nLayers || cache.MaxSeqLen != maxSeqLen {
		cache = NewKVCache(nLayers, nHeads, headDim, maxSeqLen)
	} else {
		cache.Reset()
	}

	return cache
}

// PutKVCache devolve KV cache ao pool
func PutKVCache(cache *KVCache) {
	cache.Reset()
	kvCachePool.Put(cache)
}
