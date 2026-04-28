package model

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

// DAPOConfig configurações para Decoupled Clip and Dynamic Sampling Policy Optimization
type DAPOConfig struct {
	// Decoupled clipping parameters
	EpsilonLow   float64 // Lower clipping bound (typically 0.1-0.2)
	EpsilonHigh  float64 // Upper clipping bound (typically 0.2-0.3)
	
	// Dynamic sampling parameters
	MinGroupSize int     // Minimum samples per group
	MaxGroupSize int     // Maximum samples per group
	TargetKL     float64 // Target KL divergence for adaptive sampling
	
	// Length-adaptive parameters
	MaxLengthPenalty float64 // Penalty coefficient for long sequences
	LengthNormalization float64 // Normalize rewards by length
	
	// GRPO integration
	GroupSize    int
	BetaKL       float64
	LearningRate float64
	EntropyCoeff float64
	MaxGradNorm  float64
}

// DAPOSample uma amostra de treinamento DAPO
type DAPOSample struct {
	Prompt       []int
	Response     []int
	LogProbsOld  []float64
	Reward       float64
	Length       int
	NormalizedReward float64 // Length-normalized reward
	Advantage    float64
	ClipRatio    float64 // Clipped probability ratio
}

// DAPOTrapper trainer para DAPO
type DAPOTrapper struct {
	Model  *TransformerModel
	Config *DAPOConfig
}

// NewDAPOConfig cria configuração padrão para DAPO
func NewDAPOConfig() *DAPOConfig {
	return &DAPOConfig{
		// Decoupled clipping
		EpsilonLow:   0.1,
		EpsilonHigh:  0.3,
		
		// Dynamic sampling
		MinGroupSize: 2,
		MaxGroupSize: 16,
		TargetKL:     0.01,
		
		// Length-adaptive
		MaxLengthPenalty:  0.01,
		LengthNormalization: 0.5,
		
		// GRPO integration
		GroupSize:    8,
		BetaKL:       0.01,
		LearningRate: 0.001,
		EntropyCoeff: 0.001,
		MaxGradNorm:  1.0,
	}
}

// NewDAPOTrapper cria um novo trainer DAPO
func NewDAPOTrapper(model *TransformerModel, config *DAPOConfig) *DAPOTrapper {
	return &DAPOTrapper{
		Model:  model,
		Config: config,
	}
}

// ComputeDecoupledClip aplica decoupled clipping com bounds assimétricos
func (trainer *DAPOTrapper) ComputeDecoupledClip(ratio float64, advantage float64) float64 {
	// Decoupled clipping: diferentes bounds para positive/negative advantages
	if advantage > 0 {
		// Para vantagens positivas: clip superior mais permissivo
		clippedRatio := math.Min(ratio, 1.0+trainer.Config.EpsilonHigh)
		return clippedRatio
	} else {
		// Para vantagens negativas: clip inferior mais restritivo
		clippedRatio := math.Max(ratio, 1.0-trainer.Config.EpsilonLow)
		return clippedRatio
	}
}

// ComputeDynamicGroupSize calcula tamanho ótimo do grupo baseado em KL
func (trainer *DAPOTrapper) ComputeDynamicGroupSize(currentKL float64) int {
	// Se KL está alto, aumentar grupo para melhor estimativa
	// Se KL está baixo, reduzir grupo para eficiência
	
	if currentKL > trainer.Config.TargetKL * 2 {
		// KL muito alto → aumentar grupo
		return minI(trainer.Config.MaxGroupSize, 
			trainer.Config.GroupSize * 2)
	} else if currentKL < trainer.Config.TargetKL * 0.5 {
		// KL muito baixo → reduzir grupo
		return maxI(trainer.Config.MinGroupSize,
			trainer.Config.GroupSize / 2)
	}
	
	// KL normal → usar tamanho padrão
	return trainer.Config.GroupSize
}

// ComputeLengthAdaptiveReward calcula recompensa adaptativa ao comprimento
func (trainer *DAPOTrapper) ComputeLengthAdaptiveReward(
	reward float64,
	length int,
	avgLength float64,
) float64 {
	// 1. Normalizar por comprimento (evitar bias para respostas longas)
	lengthNormReward := reward / math.Pow(float64(length), trainer.Config.LengthNormalization)
	
	// 2. Penalizar sequências excessivamente longas
	lengthPenalty := 0.0
	if float64(length) > avgLength * 2.0 {
		excess := float64(length) - avgLength*2.0
		lengthPenalty = trainer.Config.MaxLengthPenalty * excess
	}
	
	// 3. Reward final
	adjustedReward := lengthNormReward - lengthPenalty
	
	return adjustedReward
}

// ComputeGroupAdvantagesWithLength calcula vantagens com normalização de comprimento
func (trainer *DAPOTrapper) ComputeGroupAdvantagesWithLength(samples []*DAPOSample) {
	if len(samples) == 0 {
		return
	}
	
	// Agrupar por prompt
	promptGroups := make(map[string][]*DAPOSample)
	for _, sample := range samples {
		key := fmt.Sprintf("%v", sample.Prompt)
		promptGroups[key] = append(promptGroups[key], sample)
	}
	
	// Para cada grupo
	for _, group := range promptGroups {
		if len(group) == 0 {
			continue
		}
		
		// Calcular comprimento médio do grupo
		totalLength := 0
		for _, s := range group {
			totalLength += s.Length
		}
		avgLength := float64(totalLength) / float64(len(group))
		
		// Aplicar recompensas adaptativas ao comprimento
		for _, s := range group {
			s.NormalizedReward = trainer.ComputeLengthAdaptiveReward(
				s.Reward,
				s.Length,
				avgLength,
			)
		}
		
		// Calcular estatísticas das recompensas normalizadas
		meanReward := 0.0
		for _, s := range group {
			meanReward += s.NormalizedReward
		}
		meanReward /= float64(len(group))
		
		variance := 0.0
		for _, s := range group {
			diff := s.NormalizedReward - meanReward
			variance += diff * diff
		}
		variance /= float64(len(group))
		stdDev := math.Sqrt(variance)
		
		// Normalizar vantagens
		if stdDev > 1e-8 {
			for _, s := range group {
				s.Advantage = (s.NormalizedReward - meanReward) / stdDev
			}
		} else {
			for _, s := range group {
				s.Advantage = s.NormalizedReward - meanReward
			}
		}
	}
}

// ComputeDAPOLoss calcula loss do DAPO com decoupled clipping
func (trainer *DAPOTrapper) ComputeDAPOLoss(samples []*DAPOSample) float64 {
	totalLoss := 0.0
	count := 0
	
	for _, sample := range samples {
		if len(sample.Response) < 2 {
			continue
		}
		
		// Calcular log probs da policy atual
		currentLogProbs := trainer.ComputeLogProbsDAPO(sample.Response)
		
		// Para cada token
		for i := 0; i < len(currentLogProbs) && i < len(sample.LogProbsOld); i++ {
			// Probability ratio
			ratio := math.Exp(currentLogProbs[i] - sample.LogProbsOld[i])
			
			// Decoupled clipping
			clippedRatio := trainer.ComputeDecoupledClip(ratio, sample.Advantage)
			
			// Calcular surrogate losses
			surrogate1 := ratio * sample.Advantage
			surrogate2 := clippedRatio * sample.Advantage
			
			// Loss (minimize)
			loss := -math.Min(surrogate1, surrogate2)
			totalLoss += loss
			count++
			
			// Store clip ratio para análise
			sample.ClipRatio = clippedRatio
		}
	}
	
	if count > 0 {
		return totalLoss / float64(count)
	}
	return 0.0
}

// ComputeKLApproximation calcula aproximação de KL divergence
func (trainer *DAPOTrapper) ComputeKLApproximation(samples []*DAPOSample) float64 {
	totalKL := 0.0
	count := 0
	
	for _, sample := range samples {
		if len(sample.Response) < 2 {
			continue
		}
		
		currentLogProbs := trainer.ComputeLogProbsDAPO(sample.Response)
		
		for i := 0; i < len(currentLogProbs) && i < len(sample.LogProbsOld); i++ {
			// KL ≈ sum(p * log(p/q))
			// Approximation usando log probs
			kl := math.Exp(currentLogProbs[i]) * (currentLogProbs[i] - sample.LogProbsOld[i])
			totalKL += kl
			count++
		}
	}
	
	if count > 0 {
		return totalKL / float64(count)
	}
	return 0.0
}

// ComputeLogProbsDAPO calcula log probs para DAPO
func (trainer *DAPOTrapper) ComputeLogProbsDAPO(tokens []int) []float64 {
	if len(tokens) < 2 {
		return []float64{}
	}
	
	inputTokens := tokens[:len(tokens)-1]
	targetTokens := tokens[1:]
	
	hidden := trainer.Model.Forward(inputTokens)
	seqLen := len(inputTokens)
	
	logits := mat.NewDense(seqLen, trainer.Model.VocabSize, nil)
	logits.Mul(hidden, trainer.Model.WOut.T())
	
	for i := 0; i < seqLen; i++ {
		for j := 0; j < trainer.Model.VocabSize; j++ {
			val := logits.At(i, j) + trainer.Model.BOut.At(j, 0)
			logits.Set(i, j, val)
		}
	}
	
	logProbs := make([]float64, len(targetTokens))
	for i := 0; i < len(targetTokens) && i < seqLen; i++ {
		targetToken := targetTokens[i]
		
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(i, j)
		}
		
		probs := softmax(logitValues)
		
		if probs[targetToken] > 1e-10 {
			logProbs[i] = math.Log(probs[targetToken])
		} else {
			logProbs[i] = -23.0
		}
	}
	
	return logProbs
}

// DAPOStep executa um step de otimização DAPO
func (trainer *DAPOTrapper) DAPOStep(samples []*DAPOSample) map[string]float64 {
	// 1. Calcular vantagens com normalização de comprimento
	trainer.ComputeGroupAdvantagesWithLength(samples)
	
	// 2. Calcular KL atual
	currentKL := trainer.ComputeKLApproximation(samples)
	
	// 3. Ajustar tamanho do grupo dinamicamente
	dynamicGroupSize := trainer.ComputeDynamicGroupSize(currentKL)
	
	// 4. Calcular losses
	policyLoss := trainer.ComputeDAPOLoss(samples)
	entropy := trainer.ComputeEntropyDAPO(samples)
	
	// 5. Loss total
	totalLoss := policyLoss - trainer.Config.EntropyCoeff * entropy
	
	// 6. Aplicar gradientes
	trainer.applyDAPOGradients(samples, trainer.Config.LearningRate)
	
	return map[string]float64{
		"total_loss":        totalLoss,
		"policy_loss":       policyLoss,
		"kl_divergence":     currentKL,
		"entropy":           entropy,
		"dynamic_group_size": float64(dynamicGroupSize),
	}
}

// ComputeEntropyDAPO calcula entropia da policy
func (trainer *DAPOTrapper) ComputeEntropyDAPO(samples []*DAPOSample) float64 {
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

// applyDAPOGradients aplica gradientes DAPO
func (trainer *DAPOTrapper) applyDAPOGradients(samples []*DAPOSample, lr float64) {
	for _, sample := range samples {
		if len(sample.Response) < 2 || sample.Advantage == 0 {
			continue
		}
		
		inputTokens := sample.Response[:len(sample.Response)-1]
		hidden := trainer.Model.Forward(inputTokens)
		seqLen := len(inputTokens)
		
		for i := 0; i < seqLen && i < len(sample.Response)-1; i++ {
			targetToken := sample.Response[i+1]
			
			// Update baseado em advantage
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

// TrainDAPO executa treinamento DAPO completo
func (trainer *DAPOTrapper) TrainDAPO(
	prompts [][]int,
	rewardFunc func(response []int, prompt []int) float64,
	numIterations int,
) ([]map[string]float64, error) {
	allMetrics := make([]map[string]float64, 0, numIterations)
	
	fmt.Printf("Starting DAPO training: %d iterations\n", numIterations)
	fmt.Printf("Config: epsilon_low=%.2f, epsilon_high=%.2f, group_size=%d\n",
		trainer.Config.EpsilonLow,
		trainer.Config.EpsilonHigh,
		trainer.Config.GroupSize)
	
	for iter := 0; iter < numIterations; iter++ {
		// 1. Gerar respostas com sampling dinâmico
		currentKL := 0.0
		if iter > 0 {
			currentKL = allMetrics[iter-1]["kl_divergence"]
		}
		
		dynamicGroupSize := trainer.ComputeDynamicGroupSize(currentKL)
		
		allSamples := make([]*DAPOSample, 0)
		
		for _, prompt := range prompts {
			// Gerar grupo de respostas
			responses := trainer.generateDAPOResponses(prompt, dynamicGroupSize, 30, 1.0)
			
			for _, response := range responses {
				logProbs := trainer.ComputeLogProbsDAPO(response)
				reward := rewardFunc(response, prompt)
				
				sample := &DAPOSample{
					Prompt:      prompt,
					Response:    response,
					LogProbsOld: logProbs,
					Reward:      reward,
					Length:      len(response),
				}
				
				allSamples = append(allSamples, sample)
			}
		}
		
		// 2. DAPO step
		metrics := trainer.DAPOStep(allSamples)
		metrics["iteration"] = float64(iter)
		
		// 3. Calcular estatísticas
		rewards := make([]float64, len(allSamples))
		lengths := make([]float64, len(allSamples))
		for i, s := range allSamples {
			rewards[i] = s.Reward
			lengths[i] = float64(s.Length)
		}
		
		metrics["mean_reward"] = mean(rewards)
		metrics["std_reward"] = stdDev(rewards)
		metrics["mean_length"] = mean(lengths)
		
		allMetrics = append(allMetrics, metrics)
		
		// 4. Log
		fmt.Printf("Iteration %d/%d | Loss: %.4f | KL: %.4f | Reward: %.2f ± %.2f | Length: %.1f | Group: %d\n",
			iter+1, numIterations,
			metrics["total_loss"],
			metrics["kl_divergence"],
			metrics["mean_reward"],
			metrics["std_reward"],
			metrics["mean_length"],
			int(metrics["dynamic_group_size"]),
		)
	}
	
	return allMetrics, nil
}

// generateDAPOResponses gera respostas para DAPO
func (trainer *DAPOTrapper) generateDAPOResponses(
	prompt []int,
	numResponses int,
	maxLen int,
	temperature float64,
) [][]int {
	responses := make([][]int, 0, numResponses)
	
	for g := 0; g < numResponses; g++ {
		response := trainer.generateSingleDAPOResponse(prompt, maxLen, temperature)
		responses = append(responses, response)
	}
	
	return responses
}

// generateSingleDAPOResponse gera uma única resposta
func (trainer *DAPOTrapper) generateSingleDAPOResponse(
	prompt []int,
	maxLen int,
	temperature float64,
) []int {
	tokens := make([]int, len(prompt))
	copy(tokens, prompt)
	
	for i := 0; i < maxLen; i++ {
		hidden := trainer.Model.Forward(tokens)
		seqLen := len(tokens)
		
		lastHidden := mat.NewDense(1, trainer.Model.DModel, nil)
		for j := 0; j < trainer.Model.DModel; j++ {
			lastHidden.Set(0, j, hidden.At(seqLen-1, j))
		}
		
		logits := mat.NewDense(1, trainer.Model.VocabSize, nil)
		logits.Mul(lastHidden, trainer.Model.WOut.T())
		
		logitValues := make([]float64, trainer.Model.VocabSize)
		for j := 0; j < trainer.Model.VocabSize; j++ {
			logitValues[j] = logits.At(0, j) / temperature
		}
		
		probs := softmax(logitValues)
		nextToken := sampleFromDistribution(probs)
		
		tokens = append(tokens, nextToken)
	}
	
	return tokens
}

// GetDAPOStats retorna estatísticas do DAPO
func GetDAPOStats(config *DAPOConfig) map[string]interface{} {
	return map[string]interface{}{
		"epsilon_low":         config.EpsilonLow,
		"epsilon_high":        config.EpsilonHigh,
		"min_group_size":      config.MinGroupSize,
		"max_group_size":      config.MaxGroupSize,
		"target_kl":           config.TargetKL,
		"max_length_penalty":  config.MaxLengthPenalty,
		"length_normalization": config.LengthNormalization,
		"group_size":          config.GroupSize,
		"beta_kl":             config.BetaKL,
		"learning_rate":       config.LearningRate,
	}
}

// Helper functions
func minI(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxI(a, b int) int {
	if a > b {
		return a
	}
	return b
}
