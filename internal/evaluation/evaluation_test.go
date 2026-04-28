package evaluation

import (
	"fmt"
	"testing"
)

func TestCrossValidator_Split(t *testing.T) {
	// Criar dados de teste ÚNICOS (sem duplicatas)
	data := make([]string, 100)
	for i := 0; i < 100; i++ {
		data[i] = fmt.Sprintf("sample_%03d", i)
	}

	cv := NewCrossValidator(data, 5)

	// Testar cada fold
	for fold := 0; fold < 5; fold++ {
		trainData, valData, err := cv.Split(fold)
		if err != nil {
			t.Fatalf("Fold %d: Split failed: %v", fold, err)
		}

		// Verificar tamanhos
		expectedValSize := 20 // 100 / 5
		expectedTrainSize := 80

		if len(valData) != expectedValSize {
			t.Errorf("Fold %d: val size %d, expected %d", fold, len(valData), expectedValSize)
		}
		if len(trainData) != expectedTrainSize {
			t.Errorf("Fold %d: train size %d, expected %d", fold, len(trainData), expectedTrainSize)
		}

		// Verificar que não há overlap
		overlap := 0
		trainSet := make(map[string]bool)
		for _, s := range trainData {
			trainSet[s] = true
		}
		for _, s := range valData {
			if trainSet[s] {
				overlap++
			}
		}

		if overlap > 0 {
			t.Errorf("Fold %d: %d samples overlap between train and val", fold, overlap)
		}
	}

	t.Logf("Cross-validation split: 5 folds, train=80, val=20 each")
}

func TestCrossValidator_InvalidFold(t *testing.T) {
	data := []string{"a", "b", "c", "d", "e"}
	cv := NewCrossValidator(data, 5)

	// Testar fold inválido
	_, _, err := cv.Split(-1)
	if err == nil {
		t.Error("Expected error for fold -1")
	}

	_, _, err = cv.Split(5)
	if err == nil {
		t.Error("Expected error for fold 5")
	}
}

func TestCalculatePerplexity(t *testing.T) {
	tests := []struct {
		loss     float64
		expected float64
	}{
		{0.0, 1.0},
		{1.0, 2.718281828},
		{2.0, 7.389056099},
		{3.0, 20.085536923},
	}

	for _, tt := range tests {
		result := CalculatePerplexity(tt.loss)
		diff := result - tt.expected
		if diff < 0 {
			diff = -diff
		}
		if diff > 0.001 {
			t.Errorf("Perplexity(%f) = %f, expected %f", tt.loss, result, tt.expected)
		}
	}

	t.Logf("Perplexity calculations verified")
}

func TestTokenLevelAccuracy(t *testing.T) {
	tests := []struct {
		name      string
		predicted []int
		expected  []int
		acc       float64
	}{
		{
			"perfect match",
			[]int{1, 2, 3, 4, 5},
			[]int{1, 2, 3, 4, 5},
			1.0,
		},
		{
			"no match",
			[]int{1, 2, 3},
			[]int{4, 5, 6},
			0.0,
		},
		{
			"partial match",
			[]int{1, 2, 3, 4},
			[]int{1, 5, 3, 6},
			0.5,
		},
		{
			"different lengths",
			[]int{1, 2, 3},
			[]int{1, 2},
			1.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acc := TokenLevelAccuracy(tt.predicted, tt.expected)
			diff := acc - tt.acc
			if diff < 0 {
				diff = -diff
			}
			if diff > 0.01 {
				t.Errorf("Accuracy: got %f, expected %f", acc, tt.acc)
			}
		})
	}
}

func TestSequenceLevelAccuracy(t *testing.T) {
	tests := []struct {
		name      string
		predicted []int
		expected  []int
		acc       float64
	}{
		{
			"exact match",
			[]int{1, 2, 3},
			[]int{1, 2, 3},
			1.0,
		},
		{
			"different",
			[]int{1, 2, 3},
			[]int{1, 2, 4},
			0.0,
		},
		{
			"different lengths",
			[]int{1, 2, 3},
			[]int{1, 2},
			0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acc := SequenceLevelAccuracy(tt.predicted, tt.expected)
			if acc != tt.acc {
				t.Errorf("Sequence accuracy: got %f, expected %f", acc, tt.acc)
			}
		})
	}
}

func TestCalculateBLEUScore(t *testing.T) {
	tests := []struct {
		name      string
		generated string
		refs      []string
		minScore  float64
		maxScore  float64
	}{
		{
			"perfect match",
			"the cat sat on the mat",
			[]string{"the cat sat on the mat"},
			0.9,
			1.0,
		},
		{
			"partial match",
			"the cat sat",
			[]string{"the cat sat on the mat"},
			0.3,
			0.8,
		},
		{
			"no match",
			"dog barked loudly",
			[]string{"the cat sat on the mat"},
			0.0,
			0.1,
		},
		{
			"empty",
			"",
			[]string{"the cat sat"},
			0.0,
			0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bleu := CalculateBLEUScore(tt.generated, tt.refs)
			if bleu < tt.minScore || bleu > tt.maxScore {
				t.Errorf("BLEU score %f not in range [%f, %f]", bleu, tt.minScore, tt.maxScore)
			}
			t.Logf("BLEU: '%s' vs %v = %f", tt.generated, tt.refs, bleu)
		})
	}
}

func TestValidateNoDataLeakage(t *testing.T) {
	trainData := []string{"sample1", "sample2", "sample3"}
	testData := []string{"sample4", "sample5"}

	// Sem leakage
	err := ValidateNoDataLeakage(trainData, testData)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Com leakage
	testDataLeak := []string{"sample3", "sample4"}
	err = ValidateNoDataLeakage(trainData, testDataLeak)
	if err == nil {
		t.Error("Expected error for data leakage")
	} else {
		t.Logf("Correctly detected data leakage: %v", err)
	}
}

func TestGetAverageResult(t *testing.T) {
	results := []EvaluationResult{
		{Perplexity: 10.0, AverageLoss: 2.3, Accuracy: 0.85},
		{Perplexity: 12.0, AverageLoss: 2.5, Accuracy: 0.80},
		{Perplexity: 11.0, AverageLoss: 2.4, Accuracy: 0.82},
	}

	avg := GetAverageResult(results)

	// Verificar médias
	expectedPerplexity := 11.0
	expectedLoss := 2.4
	expectedAccuracy := 0.82333

	diff := avg.Perplexity - expectedPerplexity
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.1 {
		t.Errorf("Avg perplexity: %f, expected %f", avg.Perplexity, expectedPerplexity)
	}

	diff = avg.AverageLoss - expectedLoss
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.01 {
		t.Errorf("Avg loss: %f, expected %f", avg.AverageLoss, expectedLoss)
	}

	diff = avg.Accuracy - expectedAccuracy
	if diff < 0 {
		diff = -diff
	}
	if diff > 0.01 {
		t.Errorf("Avg accuracy: %f, expected %f", avg.Accuracy, expectedAccuracy)
	}

	t.Logf("Average result: perplexity=%.2f, loss=%.4f, accuracy=%.2f",
		avg.Perplexity, avg.AverageLoss, avg.Accuracy)
}

func TestCrossValidator_RunCrossValidation(t *testing.T) {
	// Criar dados sintéticos
	data := make([]string, 50)
	for i := 0; i < 50; i++ {
		data[i] = string(rune('A' + i%26))
	}

	cv := NewCrossValidator(data, 5)

	// Mock evaluation function
	evaluateFn := func(trainData, valData []string) EvaluationResult {
		return EvaluationResult{
			Perplexity:   10.0 + float64(len(valData)),
			AverageLoss:  2.3,
			Accuracy:     0.85,
			TotalSamples: len(valData),
		}
	}

	results := cv.RunCrossValidation(evaluateFn)

	if len(results) != 5 {
		t.Errorf("Expected 5 results, got %d", len(results))
	}

	// Verificar que todos os folds foram executados
	for i, r := range results {
		if r.TotalSamples == 0 {
			t.Errorf("Fold %d: no samples evaluated", i)
		}
	}

	t.Logf("Cross-validation completed: %d folds", len(results))
}
