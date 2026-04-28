package model

import (
	"fmt"
	"math"
	"time"

	"gonum.org/v1/gonum/mat"
)

// TrainingConfig configurações para treinamento
type TrainingConfig struct {
	LearningRate float64
	BatchSize    int
	Epochs       int
	MaxSeqLen    int
	GradientClip float64 // Gradient clipping threshold
	WeightDecay  float64
	LogInterval  int // Log a cada N batches
}

// TrainingMetrics métricas de treinamento
type TrainingMetrics struct {
	Epoch         int
	Batch         int
	SamplesSeen   int
	Loss          float64
	MainLoss      float64
	MoELoss       float64 // Load balancing loss
	MTPLoss       float64
	TotalLoss     float64
	LearningRate  float64
	SamplesPerSec float64
	TimeElapsed   time.Duration
}

// TrainingCallback callback para progresso de treinamento
type TrainingCallback func(metrics *TrainingMetrics) error

// TrainBatch treina o modelo com um batch de dados
// inputBatch: [batch_size, seq_len] - tokens de entrada
// targetBatch: [batch_size, seq_len] - tokens alvo (next token)
// mtpTargets: [batch_size, seq_len, num_predictions] - targets para MTP
func (model *TransformerModel) TrainBatch(
	inputBatch [][]int,
	targetBatch [][]int,
	mtpTargets [][][]int,
	config *TrainingConfig,
) (*TrainingMetrics, error) {
	batchSize := len(inputBatch)
	if batchSize == 0 {
		return nil, fmt.Errorf("batch size is zero")
	}

	totalLoss := 0.0
	mainLoss := 0.0
	mtpLossTotal := 0.0
	moeLossTotal := 0.0

	// Processar cada amostra no batch
	for b := 0; b < batchSize; b++ {
		inputTokens := inputBatch[b]
		targetTokens := targetBatch[b]
		seqLen := len(inputTokens)

		if seqLen == 0 {
			continue
		}

		// Forward pass
		hidden := model.Forward(inputTokens)

		// Calcular main loss (next token prediction)
		sampleMainLoss := computeMainLoss(model, hidden, targetTokens, seqLen)
		mainLoss += sampleMainLoss

		sampleTotalLoss := sampleMainLoss

		// MTP loss (se habilitado)
		if model.UseMTP && model.MTPHead != nil && mtpTargets != nil && b < len(mtpTargets) {
			// Converter targets MTP: [seq_len, num_predictions]
			mtpTargetsForSample := make([][]int, seqLen)
			for i := 0; i < seqLen && i < len(mtpTargets[b]); i++ {
				mtpTargetsForSample[i] = mtpTargets[b][i]
			}

			_, _, sampleMTPLoss := MTPForward(model.MTPHead, hidden, mtpTargetsForSample, seqLen)
			mtpLossTotal += sampleMTPLoss

			// Combinar com peso
			sampleTotalLoss += model.MTPHead.Config.WeightMTP * sampleMTPLoss
		}

		// MoE load balancing loss (se habilitado)
		if model.TransformerLayers[0].UseMoE {
			// Calcular load balancing loss para cada camada MoE
			for l := range model.TransformerLayers {
				layer := &model.TransformerLayers[l]
				if layer.UseMoE && layer.MoELayer != nil {
					// Forward pass com MoE para obter gates
					_, routingResult := MixtureOfExperts(layer.MoELayer, hidden, seqLen)

					// Calcular load balancing loss
					lbLoss := ComputeLoadBalancingLoss(
						routingResult.Gates,
						seqLen,
						layer.MoELayer.Config.NumExperts,
					)
					moeLossTotal += lbLoss * layer.MoELayer.Config.LambdaAux
					sampleTotalLoss += lbLoss * layer.MoELayer.Config.LambdaAux
				}
			}
		}

		totalLoss += sampleTotalLoss

		// Backward pass e update (simplificado - gradientes aproximados)
		applyGradients(model, inputTokens, hidden, targetTokens, seqLen, config.LearningRate)
	}

	// Média do batch
	metrics := &TrainingMetrics{
		Loss:         totalLoss / float64(batchSize),
		MainLoss:     mainLoss / float64(batchSize),
		MTPLoss:      mtpLossTotal / float64(batchSize),
		MoELoss:      moeLossTotal / float64(batchSize),
		TotalLoss:    totalLoss / float64(batchSize),
		LearningRate: config.LearningRate,
		SamplesSeen:  batchSize,
	}

	return metrics, nil
}

// computeMainLoss calcula cross-entropy loss para next token prediction
func computeMainLoss(model *TransformerModel, hidden *mat.Dense, targets []int, seqLen int) float64 {
	// Calcular logits: hidden @ WOut^T + BOut
	logits := mat.NewDense(seqLen, model.VocabSize, nil)
	logits.Mul(hidden, model.WOut.T())

	// Adicionar bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < model.VocabSize; j++ {
			val := logits.At(i, j) + model.BOut.At(j, 0)
			logits.Set(i, j, val)
		}
	}

	// Cross-entropy loss
	loss := 0.0
	count := 0

	for i := 0; i < seqLen && i < len(targets); i++ {
		targetToken := targets[i]
		if targetToken < 0 {
			continue
		}

		// Extrair logits
		logitValues := make([]float64, model.VocabSize)
		for j := 0; j < model.VocabSize; j++ {
			logitValues[j] = logits.At(i, j)
		}

		// Softmax
		probs := softmax(logitValues)

		// Cross-entropy: -log(prob[target])
		if probs[targetToken] > 1e-10 {
			loss += -math.Log(probs[targetToken])
			count++
		}
	}

	if count > 0 {
		return loss / float64(count)
	}
	return 0.0
}

// applyGradients aplica gradientes e atualiza pesos (simplificado)
func applyGradients(model *TransformerModel, inputTokens []int, hidden *mat.Dense, targets []int, seqLen int, lr float64) {
	// Esta é uma implementação simplificada
	// Em um cenário real, implementaríamos backpropagation completo

	// Por enquanto, aplicamos weight decay
	if model.WeightDecay > 0 {
		// Weight decay em WOut
		rows, cols := model.WOut.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				val := model.WOut.At(i, j) * (1 - lr*model.WeightDecay)
				model.WOut.Set(i, j, val)
			}
		}
	}
}

// TrainEpoch executa uma epoch completa de treinamento
func (model *TransformerModel) TrainEpoch(
	data [][]int, // Lista de sequências de tokens
	config *TrainingConfig,
	callback TrainingCallback,
) (*TrainingMetrics, error) {
	startTime := time.Now()

	totalLoss := 0.0
	totalMainLoss := 0.0
	totalMTPLoss := 0.0
	totalMoELoss := 0.0
	totalSamples := 0
	batchCount := 0

	// Preparar batches
	batches := prepareBatches(data, config.BatchSize, config.MaxSeqLen)

	for batchIdx, batch := range batches {
		inputBatch := batch.Inputs
		targetBatch := batch.Targets
		mtpTargets := batch.MTPTargets

		// Treinar batch
		metrics, err := model.TrainBatch(inputBatch, targetBatch, mtpTargets, config)
		if err != nil {
			return nil, fmt.Errorf("error training batch %d: %w", batchIdx, err)
		}

		// Acumular métricas
		totalLoss += metrics.Loss * float64(len(inputBatch))
		totalMainLoss += metrics.MainLoss * float64(len(inputBatch))
		totalMTPLoss += metrics.MTPLoss * float64(len(inputBatch))
		totalMoELoss += metrics.MoELoss * float64(len(inputBatch))
		totalSamples += len(inputBatch)
		batchCount++

		// Callback para progresso
		if callback != nil && (batchIdx%config.LogInterval == 0 || batchIdx == len(batches)-1) {
			elapsed := time.Since(startTime)
			batchMetrics := &TrainingMetrics{
				Epoch:         config.Epochs,
				Batch:         batchIdx,
				SamplesSeen:   totalSamples,
				Loss:          totalLoss / float64(totalSamples),
				MainLoss:      totalMainLoss / float64(totalSamples),
				MTPLoss:       totalMTPLoss / float64(totalSamples),
				MoELoss:       totalMoELoss / float64(totalSamples),
				TotalLoss:     totalLoss / float64(totalSamples),
				LearningRate:  config.LearningRate,
				SamplesPerSec: float64(totalSamples) / elapsed.Seconds(),
				TimeElapsed:   elapsed,
			}

			if err := callback(batchMetrics); err != nil {
				return nil, fmt.Errorf("callback error: %w", err)
			}
		}
	}

	// Métricas finais da epoch
	elapsed := time.Since(startTime)
	epochMetrics := &TrainingMetrics{
		Epoch:         config.Epochs,
		Batch:         batchCount,
		SamplesSeen:   totalSamples,
		Loss:          totalLoss / float64(totalSamples),
		MainLoss:      totalMainLoss / float64(totalSamples),
		MTPLoss:       totalMTPLoss / float64(totalSamples),
		MoELoss:       totalMoELoss / float64(totalSamples),
		TotalLoss:     totalLoss / float64(totalSamples),
		LearningRate:  config.LearningRate,
		SamplesPerSec: float64(totalSamples) / elapsed.Seconds(),
		TimeElapsed:   elapsed,
	}

	return epochMetrics, nil
}

// Train executa treinamento completo por múltiplas epochs
func (model *TransformerModel) Train(
	data [][]int,
	config *TrainingConfig,
	callback TrainingCallback,
) ([]*TrainingMetrics, error) {
	allMetrics := make([]*TrainingMetrics, 0, config.Epochs)

	fmt.Printf("Starting training: %d epochs, batch_size=%d, lr=%.4f\n",
		config.Epochs, config.BatchSize, config.LearningRate)
	fmt.Printf("Dataset: %d sequences\n", len(data))

	for epoch := 1; epoch <= config.Epochs; epoch++ {
		// Atualizar epoch no config
		epochConfig := *config
		epochConfig.Epochs = epoch

		// Treinar uma epoch
		epochMetrics, err := model.TrainEpoch(data, &epochConfig, callback)
		if err != nil {
			return allMetrics, fmt.Errorf("error in epoch %d: %w", epoch, err)
		}

		allMetrics = append(allMetrics, epochMetrics)

		// Log da epoch
		fmt.Printf("Epoch %d/%d | Loss: %.4f | Main: %.4f | MTP: %.4f | MoE: %.4f | Speed: %.1f samples/s | Time: %v\n",
			epoch, config.Epochs,
			epochMetrics.TotalLoss,
			epochMetrics.MainLoss,
			epochMetrics.MTPLoss,
			epochMetrics.MoELoss,
			epochMetrics.SamplesPerSec,
			epochMetrics.TimeElapsed,
		)

		// Incrementar contador de epochs treinadas
		model.EpochsTrained++
	}

	return allMetrics, nil
}

// Batch data structure
type Batch struct {
	Inputs     [][]int
	Targets    [][]int
	MTPTargets [][][]int
}

// prepareBatches prepara batches de dados
func prepareBatches(data [][]int, batchSize, maxSeqLen int) []Batch {
	batches := make([]Batch, 0)

	for i := 0; i < len(data); i += batchSize {
		end := i + batchSize
		if end > len(data) {
			end = len(data)
		}

		batch := Batch{
			Inputs:  make([][]int, 0, end-i),
			Targets: make([][]int, 0, end-i),
		}

		for j := i; j < end; j++ {
			seq := data[j]
			if len(seq) < 2 {
				continue
			}

			// Truncar se necessário
			if len(seq) > maxSeqLen {
				seq = seq[:maxSeqLen]
			}

			// Input: todos exceto último
			input := seq[:len(seq)-1]
			// Target: todos exceto primeiro
			target := seq[1:]

			// Ajustar tamanhos para serem iguais
			minLen := len(input)
			if len(target) < minLen {
				minLen = len(target)
			}
			input = input[:minLen]
			target = target[:minLen]

			batch.Inputs = append(batch.Inputs, input)
			batch.Targets = append(batch.Targets, target)
		}

		if len(batch.Inputs) > 0 {
			batches = append(batches, batch)
		}
	}

	return batches
}

// GetTrainingStats retorna estatísticas do modelo para treinamento
func (model *TransformerModel) GetTrainingStats() map[string]interface{} {
	stats := map[string]interface{}{
		"epochs_trained": model.EpochsTrained,
		"vocab_size":     model.VocabSize,
		"d_model":        model.DModel,
		"n_layers":       model.NLayers,
		"n_heads":        model.NHeads,
		"use_moe":        false,
		"use_mtp":        model.UseMTP,
	}

	// Verificar MoE
	moeCount := 0
	for _, layer := range model.TransformerLayers {
		if layer.UseMoE {
			moeCount++
		}
	}
	stats["use_moe"] = moeCount > 0
	stats["moe_layers"] = moeCount

	// MTP stats
	if model.UseMTP && model.MTPHead != nil {
		mtpStats := GetMTPStats(model.MTPHead)
		stats["mtp_predictions"] = mtpStats["num_predictions"]
		stats["mtp_weight"] = mtpStats["mtp_weight"]
	}

	return stats
}
