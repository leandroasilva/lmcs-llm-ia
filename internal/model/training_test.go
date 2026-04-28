package model

import (
	"testing"
)

func TestTraining_TrainBatch(t *testing.T) {
	// Criar modelo pequeno
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)

	// Configurar vocabulário
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar batch
	inputBatch := [][]int{
		{10, 11, 12, 13, 14},
		{20, 21, 22, 23, 24},
	}

	targetBatch := [][]int{
		{11, 12, 13, 14, 15},
		{21, 22, 23, 24, 25},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    2,
	}

	// Treinar batch
	metrics, err := model.TrainBatch(inputBatch, targetBatch, nil, config)
	if err != nil {
		t.Fatalf("TrainBatch failed: %v", err)
	}

	if metrics == nil {
		t.Fatal("Metrics is nil")
	}

	if metrics.Loss <= 0 {
		t.Errorf("Loss should be positive, got %f", metrics.Loss)
	}

	t.Logf("✓ TrainBatch: loss=%.4f, main_loss=%.4f, samples=%d",
		metrics.Loss, metrics.MainLoss, metrics.SamplesSeen)
}

func TestTraining_TrainBatchWithMTP(t *testing.T) {
	// Criar modelo com MTP
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.EnableMTP(3, 0.3)

	// Configurar vocabulário
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar batch
	inputBatch := [][]int{
		{10, 11, 12, 13, 14},
	}

	targetBatch := [][]int{
		{11, 12, 13, 14, 15},
	}

	// MTP targets: [batch_size=1, seq_len=5, num_predictions=3]
	mtpTargets := [][][]int{
		{
			{11, 12, 13}, // Position 0: predict +1, +2, +3
			{12, 13, 14}, // Position 1
			{13, 14, 15}, // Position 2
			{14, 15, 16}, // Position 3
			{15, 16, 17}, // Position 4
		},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    1,
	}

	// Treinar batch com MTP
	metrics, err := model.TrainBatch(inputBatch, targetBatch, mtpTargets, config)
	if err != nil {
		t.Fatalf("TrainBatch with MTP failed: %v", err)
	}

	if metrics.MTPLoss <= 0 {
		t.Errorf("MTP loss should be positive, got %f", metrics.MTPLoss)
	}

	t.Logf("✓ TrainBatch with MTP: total_loss=%.4f, main_loss=%.4f, mtp_loss=%.4f",
		metrics.TotalLoss, metrics.MainLoss, metrics.MTPLoss)
}

func TestTraining_TrainBatchWithMoE(t *testing.T) {
	// Criar modelo com MoE
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.EnableMoEForAllLayers(4, 2)

	// Configurar vocabulário
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar batch
	inputBatch := [][]int{
		{10, 11, 12, 13, 14},
	}

	targetBatch := [][]int{
		{11, 12, 13, 14, 15},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    1,
	}

	// Treinar batch com MoE
	metrics, err := model.TrainBatch(inputBatch, targetBatch, nil, config)
	if err != nil {
		t.Fatalf("TrainBatch with MoE failed: %v", err)
	}

	t.Logf("✓ TrainBatch with MoE: total_loss=%.4f, main_loss=%.4f, moe_loss=%.4f",
		metrics.TotalLoss, metrics.MainLoss, metrics.MoELoss)
}

func TestTraining_TrainBatchWithMoEAndMTP(t *testing.T) {
	// Criar modelo com MoE e MTP
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.EnableMoEForAllLayers(4, 2)
	model.EnableMTP(2, 0.3)

	// Configurar vocabulário
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar batch
	inputBatch := [][]int{
		{10, 11, 12, 13, 14},
	}

	targetBatch := [][]int{
		{11, 12, 13, 14, 15},
	}

	mtpTargets := [][][]int{
		{
			{11, 12},
			{12, 13},
			{13, 14},
			{14, 15},
			{15, 16},
		},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    1,
	}

	// Treinar batch com MoE + MTP
	metrics, err := model.TrainBatch(inputBatch, targetBatch, mtpTargets, config)
	if err != nil {
		t.Fatalf("TrainBatch with MoE+MTP failed: %v", err)
	}

	t.Logf("✓ TrainBatch with MoE+MTP:")
	t.Logf("  Total loss: %.4f", metrics.TotalLoss)
	t.Logf("  Main loss: %.4f", metrics.MainLoss)
	t.Logf("  MTP loss: %.4f (weight=%.2f)", metrics.MTPLoss, model.MTPHead.Config.WeightMTP)
	t.Logf("  MoE loss: %.4f (load balancing)", metrics.MoELoss)

	// Verificar que todas as losses são positivas
	if metrics.MainLoss <= 0 {
		t.Error("Main loss should be positive")
	}
	if metrics.MTPLoss <= 0 {
		t.Error("MTP loss should be positive")
	}
}

func TestTraining_PrepareBatches(t *testing.T) {
	// Preparar dados
	data := [][]int{
		{1, 2, 3, 4, 5, 6},
		{10, 11, 12, 13, 14},
		{20, 21, 22},
		{30, 31, 32, 33},
		{40, 41, 42, 43, 44, 45, 46, 47},
	}

	batches := prepareBatches(data, 2, 10)

	if len(batches) < 1 {
		t.Fatal("Expected at least 1 batch")
	}

	totalSamples := 0
	for _, batch := range batches {
		totalSamples += len(batch.Inputs)

		// Verificar que inputs e targets têm o mesmo tamanho
		for i := range batch.Inputs {
			if len(batch.Inputs[i]) != len(batch.Targets[i]) {
				t.Errorf("Batch sample %d: input len %d != target len %d",
					i, len(batch.Inputs[i]), len(batch.Targets[i]))
			}
		}
	}

	if totalSamples != len(data) {
		t.Errorf("Expected %d total samples, got %d", len(data), totalSamples)
	}

	t.Logf("✓ PrepareBatches: %d batches, %d total samples", len(batches), totalSamples)
}

func TestTraining_TrainEpoch(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(50, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 50)
	for i := 0; i < 50; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar dados pequenos
	data := [][]int{
		{1, 2, 3, 4, 5},
		{10, 11, 12, 13, 14, 15},
		{20, 21, 22, 23},
		{30, 31, 32, 33, 34, 35, 36},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    2,
		MaxSeqLen:    32,
		LogInterval:  1,
	}

	// Treinar uma epoch
	callbackCalled := 0
	callback := func(metrics *TrainingMetrics) error {
		callbackCalled++
		t.Logf("  Callback: batch=%d, samples=%d, loss=%.4f",
			metrics.Batch, metrics.SamplesSeen, metrics.Loss)
		return nil
	}

	metrics, err := model.TrainEpoch(data, config, callback)
	if err != nil {
		t.Fatalf("TrainEpoch failed: %v", err)
	}

	if callbackCalled == 0 {
		t.Error("Callback was never called")
	}

	t.Logf("✓ TrainEpoch: loss=%.4f, samples=%d, callbacks=%d",
		metrics.Loss, metrics.SamplesSeen, callbackCalled)
}

func TestTraining_FullTrainingLoop(t *testing.T) {
	// Criar modelo
	model := NewTransformerModel(30, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.Vocab = make([]string, 30)
	for i := 0; i < 30; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar dados
	data := [][]int{
		{1, 2, 3, 4, 5},
		{5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14},
		{15, 16, 17, 18, 19},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    2,
		Epochs:       3,
		MaxSeqLen:    32,
		LogInterval:  1,
	}

	// Treinar
	allMetrics, err := model.Train(data, config, nil)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	if len(allMetrics) != 3 {
		t.Errorf("Expected 3 epochs of metrics, got %d", len(allMetrics))
	}

	// Verificar que modelo foi treinado
	if model.EpochsTrained != 3 {
		t.Errorf("Model epochs trained: expected 3, got %d", model.EpochsTrained)
	}

	t.Logf("✓ Full Training Loop:")
	for i, m := range allMetrics {
		t.Logf("  Epoch %d: loss=%.4f, time=%v", i+1, m.Loss, m.TimeElapsed)
	}
}

func TestTraining_TrainingWithMoEAndMTP(t *testing.T) {
	// Criar modelo com MoE e MTP
	model := NewTransformerModel(30, 64, 4, 2, 32, 128, 0.001, 0.1, 0.01)
	model.EnableMoEForAllLayers(4, 2)
	model.EnableMTP(2, 0.3)

	// Configurar vocabulário
	model.Vocab = make([]string, 30)
	for i := 0; i < 30; i++ {
		model.Vocab[i] = string(rune('a' + i))
	}

	// Preparar dados
	data := [][]int{
		{1, 2, 3, 4, 5},
		{5, 6, 7, 8, 9},
		{10, 11, 12, 13, 14},
	}

	config := &TrainingConfig{
		LearningRate: 0.001,
		BatchSize:    2,
		Epochs:       2,
		MaxSeqLen:    32,
		LogInterval:  1,
	}

	// Treinar com MoE + MTP
	allMetrics, err := model.Train(data, config, func(metrics *TrainingMetrics) error {
		t.Logf("  Progress: epoch=%d, batch=%d, loss=%.4f (main=%.4f, mtp=%.4f, moe=%.4f)",
			metrics.Epoch, metrics.Batch, metrics.TotalLoss,
			metrics.MainLoss, metrics.MTPLoss, metrics.MoELoss)
		return nil
	})
	if err != nil {
		t.Fatalf("Training with MoE+MTP failed: %v", err)
	}

	t.Logf("✓ Training with MoE+MTP: %d epochs completed", len(allMetrics))
	t.Logf("  Final loss: %.4f", allMetrics[len(allMetrics)-1].TotalLoss)
}

func TestTraining_GetTrainingStats(t *testing.T) {
	// Criar modelo sem MoE/MTP
	model1 := NewTransformerModel(100, 128, 8, 4, 64, 256, 0.001, 0.1, 0.01)
	stats1 := model1.GetTrainingStats()

	t.Logf("✓ Training Stats (baseline):")
	t.Logf("  Epochs trained: %v", stats1["epochs_trained"])
	t.Logf("  Use MoE: %v", stats1["use_moe"])
	t.Logf("  Use MTP: %v", stats1["use_mtp"])

	// Criar modelo com MoE e MTP
	model2 := NewTransformerModel(100, 128, 8, 4, 64, 256, 0.001, 0.1, 0.01)
	model2.EnableMoEForAllLayers(8, 2)
	model2.EnableMTP(3, 0.3)
	stats2 := model2.GetTrainingStats()

	t.Logf("✓ Training Stats (MoE+MTP):")
	t.Logf("  Epochs trained: %v", stats2["epochs_trained"])
	t.Logf("  Use MoE: %v (%d layers)", stats2["use_moe"], stats2["moe_layers"])
	t.Logf("  Use MTP: %v (%d predictions)", stats2["use_mtp"], stats2["mtp_predictions"])
	t.Logf("  MTP weight: %v", stats2["mtp_weight"])

	// Verificar estatísticas
	if stats1["use_moe"] != false {
		t.Error("Model1 should not use MoE")
	}
	if stats2["use_moe"] != true {
		t.Error("Model2 should use MoE")
	}
	if stats2["moe_layers"] != 4 {
		t.Errorf("Model2 should have 4 MoE layers, got %v", stats2["moe_layers"])
	}
}
