package model

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// GRPOConfig configurações para Group Relative Policy Optimization
type GRPOConfig struct {
	GroupSize         int     // Número de respostas por prompt (G)
	Epsilon           float64 // Clipping parameter (typically 0.2)
	BetaKL            float64 // KL penalty coefficient
	LearningRate      float64 // Learning rate
	EntropyCoeff      float64 // Entropy regularization coefficient
	MaxGradNorm       float64 // Gradient clipping threshold
}

// GRPOSample uma amostra de treinamento GRPO
type GRPOSample struct {
	Prompt      []int   // Prompt inicial
	Response    []int   // Response gerado
	LogProbs    []float64 // Log probs da policy antiga
	Reward      float64   // Recompensa total
	Advantage   float64   // Vantagem normalizada (computed)
}

// GRPOTrainer trainer para GRPO
type GRPOTrainer struct {
	Model  *TransformerModel
	Config *GRPOConfig
}

// NewGRPOConfig cria configuração padrão para GRPO
func NewGRPOConfig(groupSize int, epsilon, betaKL, lr float64) *GRPOConfig {
	return &GRPOConfig{
		GroupSize:    groupSize,
		Epsilon:      epsilon,
		BetaKL:       0.01, // KL penalty típico
		LearningRate: lr,
		EntropyCoeff: 0.001, // Entropy regularization
		MaxGradNorm:  1.0,
	}
}

// NewGRPOTrainer cria um novo trainer GRPO
func NewGRPOTrainer(model *TransformerModel, config *GRPOConfig) *GRPOTrainer {
	return &GRPOTrainer{
		Model:  model,
		Config: config,
	}
}

// ComputeLogProbs calcula log probs para uma sequência
func (trainer *GRPOTrainer) ComputeLogProbs(tokens []int) []float64 {
	if len(tokens) < 2 {
		return []float64{}
	}

	// Input e target
	inputTokens := tokens[:len(tokens)-1]
	targetTokens := tokens[1:]

	// Forward pass
	hidden := trainer.Model.Forward(inputTokens)
	seqLen := len(inputTokens)

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

	// Calcular log probs
	logProbs := make([]float64, len(targetTokens))
	for i := 0; i < len(targetTokens) && i < seqLen; i++ {
		targetToken := targetTokens[i]

		// Extrair logits
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(i, j)
		}

		// Softmax
		probs := softmax(logitValues)

		// Log prob
		if probs[targetToken] > 1e-10 {
			logProbs[i] = math.Log(probs[targetToken])
		} else {
			logProbs[i] = -23.0 // log(1e-10)
		}
	}

	return logProbs
}

// ComputeGroupAdvantages calcula vantagens relativas ao grupo
// Normaliza recompensas dentro do grupo para cada prompt
func ComputeGroupAdvantages(samples []*GRPOSample) {
	// Agrupar por prompt
	promptGroups := make(map[string][]*GRPOSample)
	
	for _, sample := range samples {
		key := fmt.Sprintf("%v", sample.Prompt)
		promptGroups[key] = append(promptGroups[key], sample)
	}

	// Para cada grupo, calcular vantagens relativas
	for _, group := range promptGroups {
		if len(group) == 0 {
			continue
		}

		// Calcular média e desvio padrão das recompensas
		meanReward := 0.0
		for _, s := range group {
			meanReward += s.Reward
		}
		meanReward /= float64(len(group))

		variance := 0.0
		for _, s := range group {
			diff := s.Reward - meanReward
			variance += diff * diff
		}
		variance /= float64(len(group))
		stdDev := math.Sqrt(variance)

		// Normalizar: advantage = (reward - mean) / std
		if stdDev > 1e-8 {
			for _, s := range group {
				s.Advantage = (s.Reward - meanReward) / stdDev
			}
		} else {
			// Se std é muito pequeno, usar diferença simples
			for _, s := range group {
				s.Advantage = s.Reward - meanReward
			}
		}
	}
}

// ComputeGRPOLoss calcula a loss do GRPO para um grupo de samples
func (trainer *GRPOTrainer) ComputeGRPOLoss(samples []*GRPOSample) float64 {
	totalLoss := 0.0
	count := 0

	for _, sample := range samples {
		if len(sample.Response) < 2 {
			continue
		}

		// Calcular log probs da policy atual
		currentLogProbs := trainer.ComputeLogProbs(sample.Response)
		
		// Calcular ratio: exp(log_prob_current - log_prob_old)
		for i := 0; i < len(currentLogProbs) && i < len(sample.LogProbs); i++ {
			ratio := math.Exp(currentLogProbs[i] - sample.LogProbs[i])
			
			// Clipped surrogate loss
			clippedRatio := math.Max(
				1.0-trainer.Config.Epsilon,
				math.Min(1.0+trainer.Config.Epsilon, ratio),
			)

			// PPO-style loss com vantagem GRPO
			surrogate1 := ratio * sample.Advantage
			surrogate2 := clippedRatio * sample.Advantage
			
			// Loss (minimize)
			loss := -math.Min(surrogate1, surrogate2)
			totalLoss += loss
			count++
		}
	}

	if count > 0 {
		return totalLoss / float64(count)
	}
	return 0.0
}

// ComputeKLPenalty calcula KL divergence entre policy atual e reference
func (trainer *GRPOTrainer) ComputeKLPenalty(samples []*GRPOSample) float64 {
	totalKL := 0.0
	count := 0

	for _, sample := range samples {
		if len(sample.Response) < 2 {
			continue
		}

		// Forward pass para obter probs atuais
		inputTokens := sample.Response[:len(sample.Response)-1]
		hidden := trainer.Model.Forward(inputTokens)
		seqLen := len(inputTokens)

		// Calcular probs
		logits := mat.NewDense(seqLen, trainer.Model.VocabSize, nil)
		logits.Mul(hidden, trainer.Model.WOut.T())

		for i := 0; i < seqLen && i < len(sample.Response)-1; i++ {
			targetToken := sample.Response[i+1]

			// Extrair logits
			logitValues := make([]float64, trainer.Model.VocabSize)
			for j := 0; j < trainer.Model.VocabSize; j++ {
				logitValues[j] = logits.At(i, j)
			}

			// Softmax
			probs := softmax(logitValues)

			// KL divergence (aproximada)
			// KL(P||Q) = sum(P * log(P/Q))
			// Aqui usamos uma aproximação simples
			if probs[targetToken] > 1e-10 {
				// Penalizar prob muito baixa ou muito alta
				kl := math.Log(1.0 / probs[targetToken])
				totalKL += kl
				count++
			}
		}
	}

	if count > 0 {
		return totalKL / float64(count)
	}
	return 0.0
}

// ComputeEntropy calcula entropia da policy (para regularização)
func (trainer *GRPOTrainer) ComputeEntropy(samples []*GRPOSample) float64 {
	totalEntropy := 0.0
	count := 0

	for _, sample := range samples {
		if len(sample.Response) < 2 {
			continue
		}

		inputTokens := sample.Response[:len(sample.Response)-1]
		hidden := trainer.Model.Forward(inputTokens)
		seqLen := len(inputTokens)

		logits := mat.NewDense(seqLen, trainer.Model.VocabSize, nil)
		logits.Mul(hidden, trainer.Model.WOut.T())

		for i := 0; i < seqLen; i++ {
			logitValues := make([]float64, trainer.Model.VocabSize)
			for j := 0; j < trainer.Model.VocabSize; j++ {
				logitValues[j] = logits.At(i, j)
			}

			probs := softmax(logitValues)

			// Entropia: -sum(p * log(p))
			entropy := 0.0
			for _, p := range probs {
				if p > 1e-10 {
					entropy -= p * math.Log(p)
				}
			}

			totalEntropy += entropy
			count++
		}
	}

	if count > 0 {
		return totalEntropy / float64(count)
	}
	return 0.0
}

// GRPOStep executa um step de otimização GRPO
func (trainer *GRPOTrainer) GRPOStep(samples []*GRPOSample) map[string]float64 {
	// 1. Calcular vantagens relativas ao grupo
	ComputeGroupAdvantages(samples)

	// 2. Calcular losses
	policyLoss := trainer.ComputeGRPOLoss(samples)
	klPenalty := trainer.ComputeKLPenalty(samples)
	entropy := trainer.ComputeEntropy(samples)

	// 3. Loss total
	totalLoss := policyLoss + trainer.Config.BetaKL*klPenalty - trainer.Config.EntropyCoeff*entropy

	// 4. Aplicar gradientes (simplificado)
	trainer.applyGRPOGradients(samples, trainer.Config.LearningRate)

	return map[string]float64{
		"total_loss":  totalLoss,
		"policy_loss": policyLoss,
		"kl_penalty":  klPenalty,
		"entropy":     entropy,
	}
}

// applyGRPOGradients aplica gradientes (simplificado - apenas weight updates)
func (trainer *GRPOTrainer) applyGRPOGradients(samples []*GRPOSample, lr float64) {
	// Implementação simplificada de gradientes
	// Em uma implementação completa, faríamos backpropagation completo

	// Aqui aplicamos updates baseados nas vantagens
	for _, sample := range samples {
		if len(sample.Response) < 2 || sample.Advantage == 0 {
			continue
		}

		// Forward pass
		inputTokens := sample.Response[:len(sample.Response)-1]
		hidden := trainer.Model.Forward(inputTokens)
		seqLen := len(inputTokens)

		// Calcular gradientes aproximados
		for i := 0; i < seqLen && i < len(sample.Response)-1; i++ {
			targetToken := sample.Response[i+1]
			
			// Update WOut: aumentar prob do token se advantage > 0
			if sample.Advantage > 0 {
				for j := 0; j < trainer.Model.DModel; j++ {
					grad := lr * sample.Advantage * hidden.At(i, j) * 0.01
					newVal := trainer.Model.WOut.At(targetToken, j) + grad
					trainer.Model.WOut.Set(targetToken, j, newVal)
				}
			}
		}
	}
}

// GenerateResponses gera múltiplas respostas para um prompt
func (trainer *GRPOTrainer) GenerateResponses(prompt []int, numResponses int, maxLen int, temperature float64) [][]int {
	responses := make([][]int, 0, numResponses)

	for g := 0; g < numResponses; g++ {
		response := trainer.generateSingleResponse(prompt, maxLen, temperature)
		responses = append(responses, response)
	}

	return responses
}

// generateSingleResponse gera uma única resposta
func (trainer *GRPOTrainer) generateSingleResponse(prompt []int, maxLen int, temperature float64) []int {
	tokens := make([]int, len(prompt))
	copy(tokens, prompt)

	for i := 0; i < maxLen; i++ {
		// Forward pass
		hidden := trainer.Model.Forward(tokens)
		seqLen := len(tokens)

		// Pegar última posição
		lastHidden := mat.NewDense(1, trainer.Model.DModel, nil)
		for j := 0; j < trainer.Model.DModel; j++ {
			lastHidden.Set(0, j, hidden.At(seqLen-1, j))
		}

		// Calcular logits
		logits := mat.NewDense(1, trainer.Model.VocabSize, nil)
		logits.Mul(lastHidden, trainer.Model.WOut.T())

		// Extrair e aplicar temperature
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(0, j) / temperature
		}

		// Softmax e sample
		probs := softmax(logitValues)
		nextToken := sampleFromDistribution(probs)

		tokens = append(tokens, nextToken)
	}

	return tokens
}

// RewardFunction tipo para funções de recompensa
type RewardFunction func(response []int, prompt []int) float64

// DefaultRewardFunction exemplo simples de reward function
func DefaultRewardFunction(response []int, prompt []int) float64 {
	// Recompensa baseada em comprimento e diversidade
	lengthReward := float64(len(response)) * 0.01

	// Diversidade (número de tokens únicos)
	uniqueTokens := make(map[int]bool)
	for _, token := range response {
		uniqueTokens[token] = true
	}
	diversityReward := float64(len(uniqueTokens)) * 0.05

	return lengthReward + diversityReward
}

// TrainGRPO executa treinamento GRPO completo
func (trainer *GRPOTrainer) TrainGRPO(
	prompts [][]int,
	rewardFunc RewardFunction,
	numIterations int,
) ([]map[string]float64, error) {
	allMetrics := make([]map[string]float64, 0, numIterations)

	fmt.Printf("Starting GRPO training: %d iterations, group_size=%d\n",
		numIterations, trainer.Config.GroupSize)
	fmt.Printf("Prompts: %d\n", len(prompts))

	for iter := 0; iter < numIterations; iter++ {
		// 1. Gerar respostas para cada prompt
		allSamples := make([]*GRPOSample, 0, len(prompts)*trainer.Config.GroupSize)

		for _, prompt := range prompts {
			// Gerar grupo de respostas
			responses := trainer.GenerateResponses(
				prompt,
				trainer.Config.GroupSize,
				20, // max_len
				1.0, // temperature
			)

			// Criar samples
			for _, response := range responses {
				// Calcular log probs da policy atual (será usada como "old" na próxima iteração)
				logProbs := trainer.ComputeLogProbs(response)

				// Calcular recompensa
				reward := rewardFunc(response, prompt)

				sample := &GRPOSample{
					Prompt:   prompt,
					Response: response,
					LogProbs: logProbs,
					Reward:   reward,
				}

				allSamples = append(allSamples, sample)
			}
		}

		// 2. GRPO step
		metrics := trainer.GRPOStep(allSamples)
		metrics["iteration"] = float64(iter)

		// 3. Calcular estatísticas do grupo
		rewards := make([]float64, len(allSamples))
		for i, s := range allSamples {
			rewards[i] = s.Reward
		}
		metrics["mean_reward"] = mean(rewards)
		metrics["std_reward"] = stdDev(rewards)

		allMetrics = append(allMetrics, metrics)

		// 4. Log
		fmt.Printf("Iteration %d/%d | Loss: %.4f | Policy: %.4f | KL: %.4f | Entropy: %.4f | Reward: %.2f ± %.2f\n",
			iter+1, numIterations,
			metrics["total_loss"],
			metrics["policy_loss"],
			metrics["kl_penalty"],
			metrics["entropy"],
			metrics["mean_reward"],
			metrics["std_reward"],
		)
	}

	return allMetrics, nil
}

// Funções auxiliares
func mean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func stdDev(values []float64) float64 {
	m := mean(values)
	variance := 0.0
	for _, v := range values {
		diff := v - m
		variance += diff * diff
	}
	variance /= float64(len(values))
	return math.Sqrt(variance)
}

// GetGRPOStats retorna estatísticas do GRPO
func GetGRPOStats(config *GRPOConfig) map[string]interface{} {
	return map[string]interface{}{
		"group_size":    config.GroupSize,
		"epsilon":       config.Epsilon,
		"beta_kl":       config.BetaKL,
		"learning_rate": config.LearningRate,
		"entropy_coeff": config.EntropyCoeff,
		"max_grad_norm": config.MaxGradNorm,
	}
}
