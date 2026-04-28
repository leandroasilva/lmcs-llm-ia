package model

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// CRVPhase representa as fases do framework CRV
type CRVPhase string

const (
	PhaseCritique CRVPhase = "critique"
	PhaseRethink  CRVPhase = "rethink"
	PhaseVerify   CRVPhase = "verify"
)

// CRVResult resultado de uma fase CRV
type CRVResult struct {
	Phase        CRVPhase
	Content      string
	Score        float64
	Issues       []string
	Improvements []string
}

// CRVConfig configurações do framework CRV
type CRVConfig struct {
	MaxIterations     int
	CritiqueThreshold float64
	RethinkDepth      int
	VerifyStrict      bool
	SelfAlignment     bool // Alinhar com capacidade cognitiva do modelo
}

// CRVFramework implementa Critique-Rethink-Verify
type CRVFramework struct {
	Model   *TransformerModel
	Config  *CRVConfig
	History []CRVResult
}

// NewCRVFramework cria novo framework CRV
func NewCRVFramework(model *TransformerModel, config *CRVConfig) *CRVFramework {
	return &CRVFramework{
		Model:   model,
		Config:  config,
		History: make([]CRVResult, 0),
	}
}

// Execute executa pipeline CRV completo
func (crv *CRVFramework) Execute(problem string, initialAnswer string) (*CRVResult, error) {
	currentAnswer := initialAnswer

	for iteration := 0; iteration < crv.Config.MaxIterations; iteration++ {
		// Phase 1: Critique
		critiqueResult, err := crv.critique(problem, currentAnswer)
		if err != nil {
			return nil, fmt.Errorf("critique failed: %w", err)
		}
		crv.History = append(crv.History, *critiqueResult)

		// Se score acima do threshold, parar
		if critiqueResult.Score >= crv.Config.CritiqueThreshold {
			return critiqueResult, nil
		}

		// Phase 2: Rethink
		rethinkResult, err := crv.rethink(problem, currentAnswer, critiqueResult)
		if err != nil {
			return nil, fmt.Errorf("rethink failed: %w", err)
		}
		crv.History = append(crv.History, *rethinkResult)
		currentAnswer = rethinkResult.Content

		// Phase 3: Verify
		verifyResult, err := crv.verify(problem, currentAnswer)
		if err != nil {
			return nil, fmt.Errorf("verify failed: %w", err)
		}
		crv.History = append(crv.History, *verifyResult)

		if verifyResult.Score >= crv.Config.CritiqueThreshold {
			return verifyResult, nil
		}
	}

	// Retornar melhor resultado
	bestResult := crv.findBestResult()
	return bestResult, nil
}

// critique analisa e critica a resposta
func (crv *CRVFramework) critique(problem string, answer string) (*CRVResult, error) {
	// Em produção: usar modelo para criticar
	// Aqui: heurísticas de qualidade

	issues := make([]string, 0)
	score := 0.5 // Base score

	// Check 1: Comprimento adequado
	if len(answer) < 20 {
		issues = append(issues, "Answer too short, lacks detail")
		score -= 0.2
	}

	// Check 2: Estrutura de raciocínio
	hasReasoning := strings.Contains(strings.ToLower(answer), "because") ||
		strings.Contains(strings.ToLower(answer), "therefore") ||
		strings.Contains(strings.ToLower(answer), "thus")

	if !hasReasoning {
		issues = append(issues, "Missing reasoning structure")
		score -= 0.15
	}

	// Check 3: Consistência com problema
	if len(answer) < len(problem)/2 {
		issues = append(issues, "Answer may not fully address problem")
		score -= 0.1
	}

	// Check 4: Palavras-chave de qualidade
	qualityWords := []string{"step", "analyze", "conclude", "evidence", "logical"}
	improvements := make([]string, 0)
	for _, word := range qualityWords {
		if !strings.Contains(strings.ToLower(answer), word) {
			improvements = append(improvements, fmt.Sprintf("Consider adding '%s' reasoning", word))
		}
	}

	score = math.Max(0.0, math.Min(1.0, score))

	return &CRVResult{
		Phase:        PhaseCritique,
		Content:      answer,
		Score:        score,
		Issues:       issues,
		Improvements: improvements,
	}, nil
}

// rethink repensa a resposta baseado na crítica
func (crv *CRVFramework) rethink(problem string, answer string, critique *CRVResult) (*CRVResult, error) {
	// Em produção: usar modelo para repensar
	// Aqui: aplicar melhorias sugeridas

	improved := answer

	// Adicionar estrutura de raciocínio se missing
	if strings.Contains(strings.Join(critique.Issues, " "), "reasoning structure") {
		improved = "Step 1: Analyze the problem\n" + improved + "\nTherefore, the conclusion follows."
	}

	// Adicionar detalhes se muito curto
	if strings.Contains(strings.Join(critique.Issues, " "), "too short") {
		improved += "\n\nDetailed explanation: This solution considers multiple aspects and provides comprehensive reasoning."
	}

	// Self-alignment: adaptar à capacidade do modelo
	if crv.Config.SelfAlignment {
		improved = crv.alignWithModelCapacity(improved)
	}

	score := critique.Score + 0.2 // Improvement bonus
	score = math.Min(1.0, score)

	return &CRVResult{
		Phase:   PhaseRethink,
		Content: improved,
		Score:   score,
		Issues:  make([]string, 0),
	}, nil
}

// verify verifica a resposta final
func (crv *CRVFramework) verify(problem string, answer string) (*CRVResult, error) {
	issues := make([]string, 0)
	score := 0.7 // Base score for verified answer

	// Check 1: Responde ao problema?
	if len(answer) < len(problem)/3 {
		issues = append(issues, "Answer may not fully address the problem")
		score -= 0.2
	}

	// Check 2: Logicamente consistente?
	hasLogic := strings.Contains(strings.ToLower(answer), "because") ||
		strings.Contains(strings.ToLower(answer), "therefore")

	if !hasLogic {
		issues = append(issues, "Lacks logical flow")
		score -= 0.15
	}

	// Check 3: Sem contradições óbvias
	if strings.Contains(strings.ToLower(answer), "contradiction") {
		issues = append(issues, "Contains contradiction")
		score -= 0.3
	}

	// Strict verification
	if crv.Config.VerifyStrict {
		// Additional checks
		if !strings.Contains(strings.ToLower(answer), "conclusion") {
			issues = append(issues, "Missing clear conclusion")
			score -= 0.1
		}
	}

	score = math.Max(0.0, math.Min(1.0, score))

	return &CRVResult{
		Phase:   PhaseVerify,
		Content: answer,
		Score:   score,
		Issues:  issues,
	}, nil
}

// alignWithModelCapacity alinha resposta com capacidade do modelo
func (crv *CRVFramework) alignWithModelCapacity(answer string) string {
	// Para modelos pequenos: simplificar raciocínio complexo
	// Para modelos grandes: permitir raciocínio mais elaborado

	// Heurística: se muito longo para modelo pequeno, simplificar
	if len(answer) > 500 {
		// Manter apenas partes essenciais
		parts := strings.Split(answer, "\n")
		if len(parts) > 5 {
			answer = strings.Join(parts[:5], "\n") + "\n... (simplified for model capacity)"
		}
	}

	return answer
}

// findBestResult encontra melhor resultado no histórico
func (crv *CRVFramework) findBestResult() *CRVResult {
	if len(crv.History) == 0 {
		return nil
	}

	best := &crv.History[0]
	for i := 1; i < len(crv.History); i++ {
		if crv.History[i].Score > best.Score {
			best = &crv.History[i]
		}
	}

	return best
}

// CogPOConfig configurações do algoritmo CogPO
type CogPOConfig struct {
	CognitiveAlignment float64 // Alinhamento com capacidade cognitiva (0-1)
	ExplorationRate    float64 // Taxa de exploração
	LearningRate       float64
	MaxEpochs          int
}

// CogPOAlgorithm implementa Cognitive-guided Policy Optimization
type CogPOAlgorithm struct {
	Model  *TransformerModel
	Config *CogPOConfig
}

// NewCogPOAlgorithm cria novo algoritmo CogPO
func NewCogPOAlgorithm(model *TransformerModel, config *CogPOConfig) *CogPOAlgorithm {
	return &CogPOAlgorithm{
		Model:  model,
		Config: config,
	}
}

// Train treina com CogPO
func (cogpo *CogPOAlgorithm) Train(
	data []string,
	labels []string,
) map[string]interface{} {
	// CogPO: otimização guiada por capacidade cognitiva

	lossHistory := make([]float64, 0)
	accuracyHistory := make([]float64, 0)

	for epoch := 0; epoch < cogpo.Config.MaxEpochs; epoch++ {
		// 1. Calcular loss base
		baseLoss := cogpo.calculateLoss(data, labels)

		// 2. Aplicar penalidade de alinhamento cognitivo
		cognitivePenalty := cogpo.cognitiveAlignmentPenalty(data)
		adjustedLoss := baseLoss + cognitivePenalty*cogpo.Config.CognitiveAlignment

		// 3. Exploration bonus
		explorationBonus := cogpo.Config.ExplorationRate * float64(epoch+1)
		finalLoss := adjustedLoss - explorationBonus*0.01

		lossHistory = append(lossHistory, finalLoss)

		// 4. Calcular accuracy
		accuracy := cogpo.evaluateAccuracy(data, labels)
		accuracyHistory = append(accuracyHistory, accuracy)

		// 5. Update model (simulated)
		cogpo.updateLoss(finalLoss)
	}

	// Estatísticas finais
	finalAccuracy := accuracyHistory[len(accuracyHistory)-1]
	avgLoss := 0.0
	for _, loss := range lossHistory {
		avgLoss += loss
	}
	avgLoss /= float64(len(lossHistory))

	return map[string]interface{}{
		"final_accuracy":   finalAccuracy,
		"average_loss":     avgLoss,
		"loss_history":     lossHistory,
		"accuracy_history": accuracyHistory,
		"epochs":           cogpo.Config.MaxEpochs,
	}
}

// calculateLoss calcula loss base
func (cogpo *CogPOAlgorithm) calculateLoss(data []string, labels []string) float64 {
	if len(data) == 0 {
		return 0.0
	}

	totalLoss := 0.0
	for i := range data {
		// Simulated loss calculation
		prediction := cogpo.predict(data[i])
		if prediction != labels[i] {
			totalLoss += 1.0
		}
	}

	return totalLoss / float64(len(data))
}

// cognitiveAlignmentPenalty calcula penalidade de alinhamento
func (cogpo *CogPOAlgorithm) cognitiveAlignmentPenalty(data []string) float64 {
	// Penalizar tarefas muito difíceis para capacidade do modelo
	// Modelos pequenos focam em tarefas fáceis primeiro

	avgDifficulty := cogpo.estimateTaskDifficulty(data)

	// Se tarefa muito difícil para modelo pequeno, penalizar
	if avgDifficulty > 0.7 {
		return (avgDifficulty - 0.7) * 2.0
	}

	return 0.0
}

// estimateTaskDifficulty estima dificuldade da tarefa
func (cogpo *CogPOAlgorithm) estimateTaskDifficulty(data []string) float64 {
	if len(data) == 0 {
		return 0.0
	}

	totalDifficulty := 0.0
	for _, item := range data {
		// Heurística: comprimento e complexidade
		difficulty := 0.3 // Base

		if len(item) > 100 {
			difficulty += 0.2
		}
		if strings.Contains(item, "prove") || strings.Contains(item, "theorem") {
			difficulty += 0.3
		}
		if strings.Contains(item, "calculate") || strings.Contains(item, "solve") {
			difficulty += 0.2
		}

		totalDifficulty += difficulty
	}

	return totalDifficulty / float64(len(data))
}

// predict faz predição (simulated)
func (cogpo *CogPOAlgorithm) predict(input string) string {
	// Simplified prediction
	if len(input) > 50 {
		return "complex_answer"
	}
	return "simple_answer"
}

// evaluateAccuracy avalia acurácia
func (cogpo *CogPOAlgorithm) evaluateAccuracy(data []string, labels []string) float64 {
	if len(data) == 0 {
		return 0.0
	}

	correct := 0
	for i := range data {
		prediction := cogpo.predict(data[i])
		if prediction == labels[i] {
			correct++
		}
	}

	return float64(correct) / float64(len(data))
}

// updateModel atualiza modelo (simulated)
func (cogpo *CogPOAlgorithm) updateLoss(loss float64) {
	// Em produção: backpropagation
	// Aqui: simulated
}

// ReasoningBank implementa memória de trabalho de raciocínios
type ReasoningBank struct {
	Reasonings []ReasoningEntry
	MaxSize    int
	Index      map[string][]int // Index por keywords
}

// ReasoningEntry entrada no banco de raciocínios
type ReasoningEntry struct {
	ID          string
	Problem     string
	Solution    string
	Strategy    string
	Difficulty  float64
	Tags        []string
	UsageCount  int
	SuccessRate float64
}

// NewReasoningBank cria novo banco de raciocínios
func NewReasoningBank(maxSize int) *ReasoningBank {
	return &ReasoningBank{
		Reasonings: make([]ReasoningEntry, 0),
		MaxSize:    maxSize,
		Index:      make(map[string][]int),
	}
}

// Add adiciona raciocínio ao banco
func (bank *ReasoningBank) Add(entry ReasoningEntry) error {
	if len(bank.Reasonings) >= bank.MaxSize {
		// Remover menos usado
		bank.removeLeastUsed()
	}

	bank.Reasonings = append(bank.Reasonings, entry)
	idx := len(bank.Reasonings) - 1

	// Indexar por tags
	for _, tag := range entry.Tags {
		bank.Index[tag] = append(bank.Index[tag], idx)
	}

	return nil
}

// Query consulta raciocínios similares
func (bank *ReasoningBank) Query(problem string, maxResults int) []ReasoningEntry {
	// Encontrar raciocínios relevantes
	scores := make(map[int]float64)

	// 1. Match por tags
	words := strings.Fields(strings.ToLower(problem))
	for _, word := range words {
		if indices, exists := bank.Index[word]; exists {
			for _, idx := range indices {
				scores[idx] += 1.0
			}
		}
	}

	// 2. Match por similaridade de texto
	for i, entry := range bank.Reasonings {
		similarity := bank.calculateSimilarity(problem, entry.Problem)
		scores[i] += similarity * 2.0
	}

	// 3. Boost por success rate
	for i, entry := range bank.Reasonings {
		scores[i] += entry.SuccessRate * 0.5
	}

	// Ordenar por score
	type scoredEntry struct {
		Entry ReasoningEntry
		Score float64
	}

	scored := make([]scoredEntry, 0)
	for idx, score := range scores {
		scored = append(scored, scoredEntry{bank.Reasonings[idx], score})
	}

	sort.Slice(scored, func(i, j int) bool {
		return scored[i].Score > scored[j].Score
	})

	// Retornar top results
	results := make([]ReasoningEntry, 0)
	for i := 0; i < len(scored) && i < maxResults; i++ {
		results = append(results, scored[i].Entry)
	}

	return results
}

// calculateSimilarity calcula similaridade entre problemas
func (bank *ReasoningBank) calculateSimilarity(p1, p2 string) float64 {
	// Simple Jaccard similarity
	words1 := make(map[string]bool)
	for _, word := range strings.Fields(strings.ToLower(p1)) {
		words1[word] = true
	}

	words2 := make(map[string]bool)
	for _, word := range strings.Fields(strings.ToLower(p2)) {
		words2[word] = true
	}

	intersection := 0
	for word := range words1 {
		if words2[word] {
			intersection++
		}
	}

	union := len(words1) + len(words2) - intersection
	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// removeLeastUsed remove raciocínio menos usado
func (bank *ReasoningBank) removeLeastUsed() {
	if len(bank.Reasonings) == 0 {
		return
	}

	minIdx := 0
	minUsage := bank.Reasonings[0].UsageCount

	for i := 1; i < len(bank.Reasonings); i++ {
		if bank.Reasonings[i].UsageCount < minUsage {
			minUsage = bank.Reasonings[i].UsageCount
			minIdx = i
		}
	}

	// Remover
	bank.Reasonings = append(bank.Reasonings[:minIdx], bank.Reasonings[minIdx+1:]...)

	// Reconstruir index
	bank.rebuildIndex()
}

// rebuildIndex reconstrói índice
func (bank *ReasoningBank) rebuildIndex() {
	bank.Index = make(map[string][]int)

	for i, entry := range bank.Reasonings {
		for _, tag := range entry.Tags {
			bank.Index[tag] = append(bank.Index[tag], i)
		}
	}
}

// IncrementUsage incrementa contador de uso
func (bank *ReasoningBank) IncrementUsage(entryID string) {
	for i := range bank.Reasonings {
		if bank.Reasonings[i].ID == entryID {
			bank.Reasonings[i].UsageCount++
			break
		}
	}
}

// GetStats retorna estatísticas do banco
func (bank *ReasoningBank) GetStats() map[string]interface{} {
	if len(bank.Reasonings) == 0 {
		return map[string]interface{}{
			"total_entries": 0,
		}
	}

	totalUsage := 0
	avgSuccess := 0.0

	for _, entry := range bank.Reasonings {
		totalUsage += entry.UsageCount
		avgSuccess += entry.SuccessRate
	}

	avgSuccess /= float64(len(bank.Reasonings))

	return map[string]interface{}{
		"total_entries":    len(bank.Reasonings),
		"max_size":         bank.MaxSize,
		"total_usage":      totalUsage,
		"avg_success_rate": avgSuccess,
		"index_size":       len(bank.Index),
	}
}

// SLMScaleLaw implementa leis de escala para modelos pequenos
type SLMScaleLaw struct {
	ModelParams    int // Número de parâmetros
	TrainingTokens int
	ComputeBudget  float64
}

// NewSLMScaleLaw cria nova instância de scaling law
func NewSLMScaleLaw(params int, tokens int, compute float64) *SLMScaleLaw {
	return &SLMScaleLaw{
		ModelParams:    params,
		TrainingTokens: tokens,
		ComputeBudget:  compute,
	}
}

// PredictPerformance prediz performance baseada em scaling laws
func (slm *SLMScaleLaw) PredictPerformance(taskDifficulty float64) map[string]interface{} {
	// Modelos <20M se comportam diferente
	isVerySmall := slm.ModelParams < 20_000_000

	// Chirikov scaling: modelos pequenos concentram em tarefas fáceis
	effectiveCapacity := slm.calculateEffectiveCapacity(taskDifficulty, isVerySmall)

	// Performance prediction
	performance := effectiveCapacity * slm.computeEfficiency()

	// Learning rate optimal
	optimalLR := slm.calculateOptimalLearningRate()

	// Training recommendations
	recommendations := slm.generateRecommendations(isVerySmall)

	return map[string]interface{}{
		"model_params":          slm.ModelParams,
		"is_very_small":         isVerySmall,
		"task_difficulty":       taskDifficulty,
		"effective_capacity":    effectiveCapacity,
		"predicted_performance": performance,
		"optimal_learning_rate": optimalLR,
		"recommendations":       recommendations,
	}
}

// calculateEffectiveCapacity calcula capacidade efetiva
func (slm *SLMScaleLaw) calculateEffectiveCapacity(difficulty float64, isVerySmall bool) float64 {
	if isVerySmall {
		// Modelos muito pequenos: concentram em tarefas fáceis
		if difficulty < 0.3 {
			return 0.8 // Good at easy tasks
		} else if difficulty < 0.6 {
			return 0.5 // Moderate at medium tasks
		} else {
			return 0.2 // Poor at hard tasks (ignored)
		}
	} else {
		// Modelos maiores: distribuição mais uniforme
		return 0.9 - difficulty*0.3
	}
}

// computeEfficiency calcula eficiência computacional
func (slm *SLMScaleLaw) computeEfficiency() float64 {
	if slm.ComputeBudget <= 0 {
		return 0.5
	}

	// Efficiency scales sublinearly
	return math.Log10(slm.ComputeBudget+1) / 10.0
}

// calculateOptimalLearningRate calcula learning rate ótimo
func (slm *SLMScaleLaw) calculateOptimalLearningRate() float64 {
	// Smaller models need higher learning rates
	if slm.ModelParams < 10_000_000 {
		return 0.01
	} else if slm.ModelParams < 50_000_000 {
		return 0.005
	} else {
		return 0.001
	}
}

// generateRecommendations gera recomendações de treinamento
func (slm *SLMScaleLaw) generateRecommendations(isVerySmall bool) []string {
	recommendations := make([]string, 0)

	if isVerySmall {
		recommendations = append(recommendations,
			"Focus on easy tasks first (<30% difficulty)",
			"Use higher learning rate (0.01)",
			"Increase training tokens by 2-3x",
			"Use curriculum learning (easy → hard)",
			"Avoid complex reasoning tasks initially",
			"Prioritize pattern matching over abstraction",
		)
	} else {
		recommendations = append(recommendations,
			"Balanced task difficulty distribution",
			"Standard learning rate (0.001-0.005)",
			"Can handle moderate complexity tasks",
			"Use standard training schedule",
		)
	}

	return recommendations
}

// FirstStageTrainingConfig configurações do primeiro estágio
type FirstStageTrainingConfig struct {
	Phase1Tokens    int
	Phase2Tokens    int
	EasyTaskRatio   float64
	CurriculumSteps int
}

// FirstStageTrainingDesigner planeja primeiro estágio de treinamento
type FirstStageTrainingDesigner struct {
	Config *FirstStageTrainingConfig
}

// NewFirstStageTrainingDesigner cria novo designer
func NewFirstStageTrainingDesigner(config *FirstStageTrainingConfig) *FirstStageTrainingDesigner {
	return &FirstStageTrainingDesigner{
		Config: config,
	}
}

// DesignPhase1 desenha fase 1 (tarefas fáceis)
func (designer *FirstStageTrainingDesigner) DesignPhase1(data []string) []string {
	// Filtrar tarefas fáceis
	easyTasks := make([]string, 0)

	for _, item := range data {
		difficulty := designer.estimateDifficulty(item)
		if difficulty < 0.4 {
			easyTasks = append(easyTasks, item)
		}
	}

	// Garantir ratio
	targetCount := int(float64(len(data)) * designer.Config.EasyTaskRatio)
	if len(easyTasks) > targetCount {
		easyTasks = easyTasks[:targetCount]
	}

	return easyTasks
}

// DesignPhase2 desenha fase 2 (tarefas moderadas)
func (designer *FirstStageTrainingDesigner) DesignPhase2(data []string) []string {
	// Filtrar tarefas moderadas
	mediumTasks := make([]string, 0)

	for _, item := range data {
		difficulty := designer.estimateDifficulty(item)
		if difficulty >= 0.4 && difficulty < 0.7 {
			mediumTasks = append(mediumTasks, item)
		}
	}

	return mediumTasks
}

// DesignCurriculum desenha currículo completo
func (designer *FirstStageTrainingDesigner) DesignCurriculum(data []string) [][]string {
	curriculum := make([][]string, designer.Config.CurriculumSteps)

	// Step size
	stepSize := 1.0 / float64(designer.Config.CurriculumSteps)

	for step := 0; step < designer.Config.CurriculumSteps; step++ {
		minDiff := float64(step) * stepSize
		maxDiff := float64(step+1) * stepSize

		stepData := make([]string, 0)
		for _, item := range data {
			difficulty := designer.estimateDifficulty(item)
			if difficulty >= minDiff && difficulty < maxDiff {
				stepData = append(stepData, item)
			}
		}

		curriculum[step] = stepData
	}

	return curriculum
}

// estimateDifficulty estima dificuldade de tarefa
func (designer *FirstStageTrainingDesigner) estimateDifficulty(task string) float64 {
	difficulty := 0.2 // Base

	// Comprimento
	if len(task) > 100 {
		difficulty += 0.2
	}

	// Complexidade
	if strings.Contains(task, "prove") || strings.Contains(task, "theorem") {
		difficulty += 0.3
	}
	if strings.Contains(task, "calculate") || strings.Contains(task, "solve") {
		difficulty += 0.2
	}
	if strings.Contains(task, "analyze") || strings.Contains(task, "evaluate") {
		difficulty += 0.1
	}

	return math.Min(1.0, difficulty)
}
