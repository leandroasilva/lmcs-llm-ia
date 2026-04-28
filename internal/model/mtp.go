package model

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

// MTPConfig configurações para Multi-Token Prediction
type MTPConfig struct {
	NumPredictions int     // Número de tokens futuros para prever (n)
	DModel         int     // Dimensão do modelo
	VocabSize      int     // Tamanho do vocabulário
	WeightMTP      float64 // Peso do MTP loss em relação ao loss principal
}

// MTPHead representa uma cabeça de predição multi-token
type MTPHead struct {
	// Heads de predição para cada posição futura
	// WOut[k] prediz o token k passos à frente
	WOuts []*mat.Dense // [num_predictions][vocab_size, d_model]
	BOuts []*mat.Dense // [num_predictions][vocab_size, 1]

	// Gradientes
	GradWOuts []*mat.Dense
	GradBOuts []*mat.Dense

	// Configuração
	Config *MTPConfig
}

// MTPResult resultado da predição multi-token
type MTPResult struct {
	// Logits para cada posição futura
	// Logits[k] = [seq_len, vocab_size] para predição k+1 tokens à frente
	Logits []*mat.Dense

	// Perdas individuais para cada predição
	Losses []float64

	// Perda total ponderada
	TotalLoss float64
}

// NewMTPConfig cria configuração padrão para MTP
func NewMTPConfig(numPredictions, vocabSize, dModel int) *MTPConfig {
	return &MTPConfig{
		NumPredictions: numPredictions,
		DModel:         dModel,
		VocabSize:      vocabSize,
		WeightMTP:      0.3, // Peso típico: 0.2-0.5
	}
}

// NewMTPHead cria uma nova cabeça MTP
func NewMTPHead(cfg *MTPConfig) *MTPHead {
	head := &MTPHead{
		Config:    cfg,
		WOuts:     make([]*mat.Dense, cfg.NumPredictions),
		BOuts:     make([]*mat.Dense, cfg.NumPredictions),
		GradWOuts: make([]*mat.Dense, cfg.NumPredictions),
		GradBOuts: make([]*mat.Dense, cfg.NumPredictions),
	}

	// Inicializar heads de predição
	for k := 0; k < cfg.NumPredictions; k++ {
		scale := math.Sqrt(2.0 / float64(cfg.DModel+cfg.VocabSize))
		head.WOuts[k] = transformerRandomMatrix(cfg.VocabSize, cfg.DModel, scale)
		head.BOuts[k] = mat.NewDense(cfg.VocabSize, 1, make([]float64, cfg.VocabSize))

		// Inicializar gradientes
		head.GradWOuts[k] = mat.NewDense(cfg.VocabSize, cfg.DModel, nil)
		head.GradBOuts[k] = mat.NewDense(cfg.VocabSize, 1, nil)
	}

	return head
}

// ComputeMTPLogits calcula logits para múltiplos tokens futuros
// hidden: [seq_len, d_model] - output do transformer
// Returns: slice de [seq_len, vocab_size] para cada predição
func ComputeMTPLogits(head *MTPHead, hidden *mat.Dense, seqLen int) []*mat.Dense {
	logits := make([]*mat.Dense, head.Config.NumPredictions)

	for k := 0; k < head.Config.NumPredictions; k++ {
		// hidden @ WOut^T + BOut
		// [seq_len, d_model] @ [d_model, vocab_size] = [seq_len, vocab_size]
		logit := mat.NewDense(seqLen, head.Config.VocabSize, nil)
		logit.Mul(hidden, head.WOuts[k].T())

		// Adicionar bias
		for i := 0; i < seqLen; i++ {
			for j := 0; j < head.Config.VocabSize; j++ {
				val := logit.At(i, j) + head.BOuts[k].At(j, 0)
				logit.Set(i, j, val)
			}
		}

		logits[k] = logit
	}

	return logits
}

// ComputeMTPLoss calcula a loss para Multi-Token Prediction
// logits: slice de [seq_len, vocab_size] para cada predição
// targets: [seq_len, num_predictions] - tokens alvo para cada posição futura
func ComputeMTPLoss(head *MTPHead, logits []*mat.Dense, targets [][]int, seqLen int) *MTPResult {
	result := &MTPResult{
		Logits: logits,
		Losses: make([]float64, head.Config.NumPredictions),
	}

	totalLoss := 0.0

	// Calcular cross-entropy loss para cada predição
	for k := 0; k < head.Config.NumPredictions; k++ {
		loss := 0.0
		count := 0

		for i := 0; i < seqLen; i++ {
			targetToken := targets[i][k]
			if targetToken < 0 {
				continue // Ignorar posições inválidas
			}

			// Extrair logits para esta posição
			logitValues := make([]float64, head.Config.VocabSize)
			for j := 0; j < head.Config.VocabSize; j++ {
				logitValues[j] = logits[k].At(i, j)
			}

			// Softmax
			probs := softmax(logitValues)

			// Cross-entropy: -log(prob[target])
			if probs[targetToken] > 1e-10 {
				loss += -math.Log(probs[targetToken])
				count++
			}
		}

		// Média
		if count > 0 {
			result.Losses[k] = loss / float64(count)
		}

		totalLoss += result.Losses[k]
	}

	// Média das perdas
	result.TotalLoss = totalLoss / float64(head.Config.NumPredictions)

	return result
}

// MTPLossAndGradients calcula loss e gradientes para MTP
func MTPLossAndGradients(head *MTPHead, hidden *mat.Dense, targets [][]int, seqLen int) (float64, []*mat.Dense, []*mat.Dense) {
	// Forward pass
	logits := ComputeMTPLogits(head, hidden, seqLen)

	// Calcular loss
	result := ComputeMTPLoss(head, logits, targets, seqLen)

	// Calcular gradientes (simplificado - apenas para ilustração)
	// Em uma implementação completa, faríamos backpropagation aqui
	// Por enquanto, retornamos gradientes zerados
	// A implementação real exigiria derivadas da cross-entropy

	return result.TotalLoss, head.GradWOuts, head.GradBOuts
}

// MTPForward faz forward pass completo com MTP
// Retorna: hidden state original, logits MTP, loss MTP
func MTPForward(head *MTPHead, hidden *mat.Dense, targets [][]int, seqLen int) (*mat.Dense, []*mat.Dense, float64) {
	// Calcular logits
	logits := ComputeMTPLogits(head, hidden, seqLen)

	// Calcular loss
	result := ComputeMTPLoss(head, logits, targets, seqLen)

	return hidden, logits, result.TotalLoss
}

// ApplyMTPGradients aplica gradientes e atualiza pesos MTP
func ApplyMTPGradients(head *MTPHead, lr float64) {
	for k := 0; k < head.Config.NumPredictions; k++ {
		// WOut -= lr * GradWOut
		rows, cols := head.WOuts[k].Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := head.WOuts[k].At(i, j) - lr*head.GradWOuts[k].At(i, j)
				head.WOuts[k].Set(i, j, val)
			}
		}

		// BOut -= lr * GradBOut
		bRows := head.BOuts[k].RawMatrix().Rows
		for i := 0; i < bRows; i++ {
			val := head.BOuts[k].At(i, 0) - lr*head.GradBOuts[k].At(i, 0)
			head.BOuts[k].Set(i, 0, val)
		}
	}
}

// MTPInference gera múltiplos tokens de uma vez usando MTP
// hidden: [1, d_model] - último hidden state
// Returns: tokens preditos [num_predictions]
func MTPInference(head *MTPHead, hidden *mat.Dense, temperature float64) []int {
	seqLen := 1
	logits := ComputeMTPLogits(head, hidden, seqLen)

	predictedTokens := make([]int, head.Config.NumPredictions)

	for k := 0; k < head.Config.NumPredictions; k++ {
		// Extrair logits
		logitValues := make([]float64, head.Config.VocabSize)
		for j := 0; j < head.Config.VocabSize; j++ {
			logitValues[j] = logits[k].At(0, j)
		}

		// Aplicar temperature
		if temperature > 0 {
			for j := range logitValues {
				logitValues[j] /= temperature
			}
		}

		// Softmax
		probs := softmax(logitValues)

		// Sample from distribution
		predictedTokens[k] = sampleFromDistribution(probs)
	}

	return predictedTokens
}

// CombineMTPLossWithMainLoss combina MTP loss com loss principal
func CombineMTPLossWithMainLoss(mainLoss, mtpLoss float64, mtpWeight float64) float64 {
	// Loss total = main_loss + mtp_weight * mtp_loss
	return mainLoss + mtpWeight*mtpLoss
}

// GetMTPStats retorna estatísticas sobre MTP
func GetMTPStats(head *MTPHead) map[string]interface{} {
	totalParams := 0
	for k := 0; k < head.Config.NumPredictions; k++ {
		rows, cols := head.WOuts[k].Dims()
		totalParams += rows * cols
		totalParams += head.BOuts[k].RawMatrix().Rows // bias
	}

	return map[string]interface{}{
		"num_predictions":       head.Config.NumPredictions,
		"mtp_weight":            head.Config.WeightMTP,
		"total_params":          totalParams,
		"params_per_head":       totalParams / head.Config.NumPredictions,
		"vocab_size":            head.Config.VocabSize,
		"d_model":               head.Config.DModel,
		"additional_params_pct": float64(totalParams) * 100,
	}
}

// sampleFromDistribution amostra um índice baseado em probabilidades
func sampleFromDistribution(probs []float64) int {
	// Sample from categorical distribution
	r := randFloat()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// randFloat gera número aleatório [0, 1)
func randFloat() float64 {
	return float64(int64(math.Float64bits(1.0))%1000000) / 1000000.0
}
