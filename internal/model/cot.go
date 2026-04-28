package model

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// CoTTokens tokens especiais para Chain of Thought
const (
	CoTThinkOpenTag  = "<|think_start|>"
	CoTThinkCloseTag = "<|think_end|>"
	CoTAnswerTag     = "<|answer|>"
)

// CoTSample uma amostra de treinamento Chain of Thought
type CoTSample struct {
	Question       string // Pergunta original
	ChainOfThought string // Raciocínio passo-a-passo
	Answer         string // Resposta final
	FullText       string // Texto completo formatado
}

// CoTConfig configurações para Chain of Thought SFT
type CoTConfig struct {
	LearningRate float64
	BatchSize    int
	Epochs       int
	MaxSeqLen    int
	WeightDecay  float64
	ThinkWeight  float64 // Peso para loss do pensamento (tipicamente 1.0)
	AnswerWeight float64 // Peso para loss da resposta (tipicamente 1.0)
}

// CoTTrainer trainer para Chain of Thought SFT
type CoTTrainer struct {
	Model  *TransformerModel
	Config *CoTConfig
}

// NewCoTConfig cria configuração padrão para CoT SFT
func NewCoTConfig() *CoTConfig {
	return &CoTConfig{
		LearningRate: 0.001,
		BatchSize:    8,
		Epochs:       10,
		MaxSeqLen:    512,
		WeightDecay:  0.01,
		ThinkWeight:  1.0,
		AnswerWeight: 1.0,
	}
}

// NewCoTTrainer cria um novo trainer CoT
func NewCoTTrainer(model *TransformerModel, config *CoTConfig) *CoTTrainer {
	return &CoTTrainer{
		Model:  model,
		Config: config,
	}
}

// FormatCoTSample formata uma amostra com tags CoT
func FormatCoTSample(question, chainOfThought, answer string) string {
	return question + "\n" +
		CoTThinkOpenTag + "\n" +
		chainOfThought + "\n" +
		CoTThinkCloseTag + "\n" +
		CoTAnswerTag + "\n" +
		answer
}

// ParseCoTResponse parseia uma resposta CoT gerada
func ParseCoTResponse(response string) (chainOfThought string, answer string) {
	// Extrair chain of thought
	thinkStartIdx := strings.Index(response, CoTThinkOpenTag)
	thinkEndIdx := strings.Index(response, CoTThinkCloseTag)
	answerIdx := strings.Index(response, CoTAnswerTag)

	if thinkStartIdx != -1 && thinkEndIdx != -1 {
		chainOfThought = response[thinkStartIdx+len(CoTThinkOpenTag) : thinkEndIdx]
		chainOfThought = strings.TrimSpace(chainOfThought)
	}

	if answerIdx != -1 {
		answer = response[answerIdx+len(CoTAnswerTag):]
		answer = strings.TrimSpace(answer)
	} else if thinkEndIdx != -1 {
		// Se não tem tag de resposta, pegar tudo após think_end
		answer = response[thinkEndIdx+len(CoTThinkCloseTag):]
		answer = strings.TrimSpace(answer)
	}

	return chainOfThought, answer
}

// CreateCoTDataset cria dataset de treinamento CoT
func CreateCoTDataset(samples []CoTSample) []string {
	dataset := make([]string, 0, len(samples))

	for _, sample := range samples {
		fullText := FormatCoTSample(sample.Question, sample.ChainOfThought, sample.Answer)
		dataset = append(dataset, fullText)
	}

	return dataset
}

// ComputeCoTLoss calcula loss separada para thinking e answer
func (trainer *CoTTrainer) ComputeCoTLoss(
	hidden *mat.Dense,
	targets []int,
	tokenTypes []int, // 0=question, 1=thinking, 2=answer
	seqLen int,
) (float64, float64, float64) {
	// Calcular logits
	logits := mat.NewDense(seqLen, trainer.Model.VocabSize, nil)
	logits.Mul(hidden, trainer.Model.WOut.T())

	// Adicionar bias
	for i := 0; i < seqLen; i++ {
		for j := 0; j < trainer.Model.VocabSize; j++ {
			val := logits.At(i, j) + trainer.Model.BOut.At(j, 0)
			logits.Set(i, j, val)
		}
	}

	thinkLoss := 0.0
	answerLoss := 0.0
	thinkCount := 0
	answerCount := 0

	// Calcular loss por tipo de token
	for i := 0; i < seqLen && i < len(targets); i++ {
		targetToken := targets[i]
		tokenType := tokenTypes[i]

		// Extrair logits
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(i, j)
		}

		// Softmax
		probs := softmax(logitValues)

		// Cross-entropy loss
		if probs[targetToken] > 1e-10 {
			loss := -math.Log(probs[targetToken])

			if tokenType == 1 {
				// Thinking tokens
				thinkLoss += loss
				thinkCount++
			} else if tokenType == 2 {
				// Answer tokens
				answerLoss += loss
				answerCount++
			}
		}
	}

	// Média
	if thinkCount > 0 {
		thinkLoss /= float64(thinkCount)
	}
	if answerCount > 0 {
		answerLoss /= float64(answerCount)
	}

	// Loss total ponderada
	totalLoss := trainer.Config.ThinkWeight*thinkLoss + trainer.Config.AnswerWeight*answerLoss

	return totalLoss, thinkLoss, answerLoss
}

// TokenizeCoTText tokeniza texto CoT e cria máscaras de tipo
func (trainer *CoTTrainer) TokenizeCoTText(text string) ([]int, []int) {
	// Simplificação: em produção, usaria o tokenizer real
	// Aqui retornamos tokens dummy com tipos

	lines := strings.Split(text, "\n")
	tokens := make([]int, 0)
	tokenTypes := make([]int, 0) // 0=question, 1=thinking, 2=answer

	currentType := 0 // Começa com pergunta

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Detectar tags
		if line == CoTThinkOpenTag {
			currentType = 1
			continue
		} else if line == CoTThinkCloseTag {
			currentType = 0
			continue
		} else if line == CoTAnswerTag {
			currentType = 2
			continue
		}

		// Tokenizar linha (simplificado)
		words := strings.Fields(line)
		for _, word := range words {
			// Em produção: token = tokenizer.Encode(word)
			// Aqui: usar hash simples como placeholder
			token := hashString(word) % trainer.Model.VocabSize
			tokens = append(tokens, token)
			tokenTypes = append(tokenTypes, currentType)
		}
	}

	return tokens, tokenTypes
}

// TrainCoTEpoch executa uma epoch de treinamento CoT
func (trainer *CoTTrainer) TrainCoTEpoch(
	dataset []string,
) (float64, float64, float64, error) {
	totalLoss := 0.0
	totalThinkLoss := 0.0
	totalAnswerLoss := 0.0
	count := 0

	for _, text := range dataset {
		// Tokenizar
		tokens, tokenTypes := trainer.TokenizeCoTText(text)

		if len(tokens) < 2 {
			continue
		}

		// Truncar se necessário
		if len(tokens) > trainer.Config.MaxSeqLen {
			tokens = tokens[:trainer.Config.MaxSeqLen]
			tokenTypes = tokenTypes[:trainer.Config.MaxSeqLen]
		}

		// Input e target
		inputTokens := tokens[:len(tokens)-1]
		targetTokens := tokens[1:]
		targetTypes := tokenTypes[1:]

		// Forward pass
		hidden := trainer.Model.Forward(inputTokens)
		seqLen := len(inputTokens)

		// Calcular loss
		loss, thinkLoss, answerLoss := trainer.ComputeCoTLoss(
			hidden, targetTokens, targetTypes, seqLen,
		)

		totalLoss += loss
		totalThinkLoss += thinkLoss
		totalAnswerLoss += answerLoss
		count++

		// Aplicar gradientes (simplificado)
		trainer.applyCoTGradients(inputTokens, hidden, targetTokens, seqLen)
	}

	if count > 0 {
		return totalLoss / float64(count),
			totalThinkLoss / float64(count),
			totalAnswerLoss / float64(count),
			nil
	}

	return 0, 0, 0, nil
}

// TrainCoT executa treinamento CoT completo
func (trainer *CoTTrainer) TrainCoT(
	dataset []string,
) ([]map[string]float64, error) {
	allMetrics := make([]map[string]float64, 0, trainer.Config.Epochs)

	fmt.Printf("Starting CoT SFT training: %d epochs, %d samples\n",
		trainer.Config.Epochs, len(dataset))
	fmt.Printf("Config: lr=%.4f, batch_size=%d, think_weight=%.1f, answer_weight=%.1f\n",
		trainer.Config.LearningRate,
		trainer.Config.BatchSize,
		trainer.Config.ThinkWeight,
		trainer.Config.AnswerWeight)

	for epoch := 1; epoch <= trainer.Config.Epochs; epoch++ {
		epochLoss, thinkLoss, answerLoss, err := trainer.TrainCoTEpoch(dataset)
		if err != nil {
			return nil, fmt.Errorf("epoch %d failed: %v", epoch, err)
		}

		metrics := map[string]float64{
			"epoch":       float64(epoch),
			"total_loss":  epochLoss,
			"think_loss":  thinkLoss,
			"answer_loss": answerLoss,
		}

		allMetrics = append(allMetrics, metrics)

		fmt.Printf("Epoch %d/%d | Total Loss: %.4f | Think Loss: %.4f | Answer Loss: %.4f\n",
			epoch, trainer.Config.Epochs,
			epochLoss, thinkLoss, answerLoss)
	}

	return allMetrics, nil
}

// GenerateCoTResponse gera resposta com Chain of Thought
func (trainer *CoTTrainer) GenerateCoTResponse(
	question string,
	maxThinkLen int,
	maxAnswerLen int,
	temperature float64,
) (chainOfThought string, answer string) {
	// 1. Gerar chain of thought
	thinkPrompt := fmt.Sprintf("%s\n%s", question, CoTThinkOpenTag)
	thinkTokens := trainer.generateTokens(thinkPrompt, maxThinkLen, temperature)

	// Adicionar tag de fechamento
	chainOfThought = trainer.tokensToText(thinkTokens)

	// 2. Gerar resposta
	answerPrompt := fmt.Sprintf("%s\n%s\n%s",
		question,
		CoTThinkOpenTag+chainOfThought+CoTThinkCloseTag,
		CoTAnswerTag,
	)
	answerTokens := trainer.generateTokens(answerPrompt, maxAnswerLen, temperature)
	answer = trainer.tokensToText(answerTokens)

	return chainOfThought, answer
}

// generateTokens gera tokens (simplificado)
func (trainer *CoTTrainer) generateTokens(
	prompt string,
	maxLen int,
	temperature float64,
) []int {
	// Tokenizar prompt
	tokens, _ := trainer.TokenizeCoTText(prompt)

	// Gerar tokens
	for i := 0; i < maxLen; i++ {
		if len(tokens) == 0 {
			break
		}

		// Forward pass
		hidden := trainer.Model.Forward(tokens)
		seqLen := len(tokens)

		// Última posição
		lastHidden := mat.NewDense(1, trainer.Model.DModel, nil)
		for j := 0; j < trainer.Model.DModel; j++ {
			lastHidden.Set(0, j, hidden.At(seqLen-1, j))
		}

		// Calcular logits
		logits := mat.NewDense(1, trainer.Model.VocabSize, nil)
		logits.Mul(lastHidden, trainer.Model.WOut.T())

		// Aplicar temperature
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(0, j) / temperature
		}

		// Softmax e sample
		probs := softmax(logitValues)
		nextToken := sampleFromDistribution(probs)

		tokens = append(tokens, nextToken)
	}

	// Retornar apenas tokens gerados (não o prompt)
	return tokens[len(tokens)-maxLen:]
}

// tokensToText converte tokens para texto (simplificado)
func (trainer *CoTTrainer) tokensToText(tokens []int) string {
	// Em produção: usar tokenizer.Decode()
	// Aqui: simulação simples
	words := make([]string, 0)
	for _, token := range tokens {
		if token < len(trainer.Model.Vocab) {
			words = append(words, trainer.Model.Vocab[token])
		}
	}
	return strings.Join(words, " ")
}

// applyCoTGradients aplica gradientes (simplificado)
func (trainer *CoTTrainer) applyCoTGradients(
	inputTokens []int,
	hidden *mat.Dense,
	targetTokens []int,
	seqLen int,
) {
	lr := trainer.Config.LearningRate

	for i := 0; i < seqLen && i < len(targetTokens); i++ {
		targetToken := targetTokens[i]

		for j := 0; j < trainer.Model.DModel; j++ {
			grad := lr * hidden.At(i, j) * 0.01
			newVal := trainer.Model.WOut.At(targetToken, j) + grad
			trainer.Model.WOut.Set(targetToken, j, newVal)
		}
	}
}

// hashString cria hash simples para string
func hashString(s string) int {
	hash := 0
	for _, c := range s {
		hash = hash*31 + int(c)
	}
	return int(math.Abs(float64(hash)))
}

// GetCoTStats retorna estatísticas do CoT
func GetCoTStats(config *CoTConfig) map[string]interface{} {
	return map[string]interface{}{
		"learning_rate": config.LearningRate,
		"batch_size":    config.BatchSize,
		"epochs":        config.Epochs,
		"max_seq_len":   config.MaxSeqLen,
		"think_weight":  config.ThinkWeight,
		"answer_weight": config.AnswerWeight,
		"think_tag":     CoTThinkOpenTag,
		"end_tag":       CoTThinkCloseTag,
		"answer_tag":    CoTAnswerTag,
	}
}
