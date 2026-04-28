package evaluation

import (
	"fmt"
	"math"
	"strings"
)

// EvaluationResult contém métricas de avaliação do modelo
type EvaluationResult struct {
	Perplexity         float64 `json:"perplexity"`
	AverageLoss        float64 `json:"average_loss"`
	Accuracy           float64 `json:"accuracy"`
	TotalSamples       int     `json:"total_samples"`
	CorrectPredictions int     `json:"correct_predictions"`
}

// CrossValidator implementa validação cruzada k-fold
type CrossValidator struct {
	K           int      // Número de folds
	Data        []string // Dataset completo
	TrainFolds  []string // Dados de treino
	ValFolds    []string // Dados de validação
	CurrentFold int      // Fold atual
}

// NewCrossValidator cria um novo validador cruzado
func NewCrossValidator(data []string, k int) *CrossValidator {
	if k <= 1 {
		k = 5 // Default: 5-fold
	}
	if k > len(data) {
		k = len(data)
	}

	return &CrossValidator{
		K:    k,
		Data: data,
	}
}

// Split divide os dados em k folds
func (cv *CrossValidator) Split(fold int) ([]string, []string, error) {
	if fold < 0 || fold >= cv.K {
		return nil, nil, fmt.Errorf("fold deve estar entre 0 e %d", cv.K-1)
	}

	n := len(cv.Data)
	foldSize := n / cv.K

	// Calcular índices
	valStart := fold * foldSize
	valEnd := valStart + foldSize

	if fold == cv.K-1 {
		valEnd = n // Último fold pega o restante
	}

	// Separar validation set
	valSet := cv.Data[valStart:valEnd]

	// Combinar todos os outros folds para treino
	trainSet := make([]string, 0, n-len(valSet))
	trainSet = append(trainSet, cv.Data[:valStart]...)
	trainSet = append(trainSet, cv.Data[valEnd:]...)

	cv.TrainFolds = trainSet
	cv.ValFolds = valSet
	cv.CurrentFold = fold

	return trainSet, valSet, nil
}

// RunCrossValidation executa validação cruzada completa
func (cv *CrossValidator) RunCrossValidation(evaluateFn func(trainData, valData []string) EvaluationResult) []EvaluationResult {
	results := make([]EvaluationResult, cv.K)

	for fold := 0; fold < cv.K; fold++ {
		trainData, valData, err := cv.Split(fold)
		if err != nil {
			continue
		}

		fmt.Printf("Fold %d/%d: train=%d samples, val=%d samples\n",
			fold+1, cv.K, len(trainData), len(valData))

		// Avaliar neste fold
		results[fold] = evaluateFn(trainData, valData)
	}

	return results
}

// GetAverageResult calcula média dos resultados de todos os folds
func GetAverageResult(results []EvaluationResult) EvaluationResult {
	if len(results) == 0 {
		return EvaluationResult{}
	}

	avg := EvaluationResult{}
	for _, r := range results {
		avg.Perplexity += r.Perplexity
		avg.AverageLoss += r.AverageLoss
		avg.Accuracy += r.Accuracy
		avg.TotalSamples += r.TotalSamples
		avg.CorrectPredictions += r.CorrectPredictions
	}

	n := float64(len(results))
	avg.Perplexity /= n
	avg.AverageLoss /= n
	avg.Accuracy /= n

	return avg
}

// CalculatePerplexity calcula perplexidade a partir da loss
// Perplexity = exp(average_cross_entropy_loss)
func CalculatePerplexity(loss float64) float64 {
	return math.Exp(loss)
}

// CalculateAccuracy calcula acurácia de predições
func CalculateAccuracy(correct, total int) float64 {
	if total == 0 {
		return 0.0
	}
	return float64(correct) / float64(total)
}

// TokenLevelAccuracy calcula acurácia em nível de token
func TokenLevelAccuracy(predicted []int, expected []int) float64 {
	if len(predicted) == 0 || len(expected) == 0 {
		return 0.0
	}

	minLen := len(predicted)
	if len(expected) < minLen {
		minLen = len(expected)
	}

	correct := 0
	for i := 0; i < minLen; i++ {
		if predicted[i] == expected[i] {
			correct++
		}
	}

	return float64(correct) / float64(minLen)
}

// SequenceLevelAccuracy calcula acurácia em nível de sequência (exact match)
func SequenceLevelAccuracy(predicted []int, expected []int) float64 {
	if len(predicted) != len(expected) {
		return 0.0
	}

	for i := 0; i < len(predicted); i++ {
		if predicted[i] != expected[i] {
			return 0.0
		}
	}

	return 1.0
}

// CalculateBLEUScore calcula BLEU score simplificado (unigram precision)
func CalculateBLEUScore(generated string, references []string) float64 {
	if len(generated) == 0 || len(references) == 0 {
		return 0.0
	}

	genTokens := strings.Fields(strings.ToLower(generated))
	if len(genTokens) == 0 {
		return 0.0
	}

	// Calcular unigram precision
	matchingTokens := 0
	totalTokens := len(genTokens)

	for _, token := range genTokens {
		// Verificar se token aparece em alguma referência
		for _, ref := range references {
			refTokens := strings.Fields(strings.ToLower(ref))
			for _, refToken := range refTokens {
				if token == refToken {
					matchingTokens++
					goto nextToken
				}
			}
		}
	nextToken:
	}

	precision := float64(matchingTokens) / float64(totalTokens)

	// Brevity penalty
	refLengths := make([]int, len(references))
	for i, ref := range references {
		refLengths[i] = len(strings.Fields(ref))
	}

	// Encontrar referência mais próxima em comprimento
	bestRefLen := refLengths[0]
	minDiff := int(math.Abs(float64(len(genTokens)) - float64(bestRefLen)))
	for _, refLen := range refLengths {
		diff := int(math.Abs(float64(len(genTokens)) - float64(refLen)))
		if diff < minDiff {
			minDiff = diff
			bestRefLen = refLen
		}
	}

	brevityPenalty := 1.0
	if len(genTokens) < bestRefLen {
		brevityPenalty = math.Exp(1 - float64(bestRefLen)/float64(len(genTokens)))
	}

	bleu := brevityPenalty * precision
	return bleu
}

// EvaluateModel avalia modelo em dataset de teste (separado do treino!)
func EvaluateModel(model interface {
	Tokenize(text string) []int
	Detokenize(tokens []int) string
	Forward(tokens []int) interface{}
}, testDataset []string) EvaluationResult {
	result := EvaluationResult{
		TotalSamples: len(testDataset),
	}

	if len(testDataset) == 0 {
		return result
	}

	totalLoss := 0.0
	correctTokens := 0
	totalTokens := 0

	for _, sample := range testDataset {
		// Tokenizar
		tokens := model.Tokenize(sample)
		if len(tokens) <= 1 {
			continue
		}

		// Forward pass (simplificado - em implementação real, calcular cross-entropy loss)
		// output := model.Forward(tokens)

		// Para avaliação real, comparar predicted vs expected
		// Aqui estamos apenas contando tokens
		totalTokens += len(tokens)
	}

	result.AverageLoss = totalLoss / float64(len(testDataset))
	result.Perplexity = CalculatePerplexity(result.AverageLoss)
	result.Accuracy = CalculateAccuracy(correctTokens, totalTokens)

	return result
}

// PrintReport imprime relatório de avaliação
func PrintReport(result EvaluationResult, foldResults []EvaluationResult) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("  RELATÓRIO DE AVALIAÇÃO DO MODELO")
	fmt.Println(strings.Repeat("=", 60))

	if len(foldResults) > 0 {
		fmt.Println("\nResultados por Fold:")
		for i, r := range foldResults {
			fmt.Printf("  Fold %d: Perplexity=%.2f, Loss=%.4f, Accuracy=%.2f%%\n",
				i+1, r.Perplexity, r.AverageLoss, r.Accuracy*100)
		}
	}

	fmt.Println("\nMédia (Cross-Validation):")
	fmt.Printf("  Perplexity:     %.2f\n", result.Perplexity)
	fmt.Printf("  Average Loss:   %.4f\n", result.AverageLoss)
	fmt.Printf("  Accuracy:       %.2f%%\n", result.Accuracy*100)
	fmt.Printf("  Total Samples:  %d\n", result.TotalSamples)

	fmt.Println(strings.Repeat("=", 60))
}

// ValidateNoDataLeakage verifica que não há overlap entre treino e teste
func ValidateNoDataLeakage(trainData, testData []string) error {
	trainSet := make(map[string]bool)
	for _, sample := range trainData {
		trainSet[sample] = true
	}

	leaked := 0
	for _, sample := range testData {
		if trainSet[sample] {
			leaked++
		}
	}

	if leaked > 0 {
		return fmt.Errorf("DATA LEAKAGE DETECTED: %d samples出现在 training and test sets", leaked)
	}

	return nil
}
