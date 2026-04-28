package model

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// AgenticStrategy tipo de estratégia agêntica
type AgenticStrategy string

const (
	StrategyChainOfThought    AgenticStrategy = "chain_of_thought"
	StrategyTreeOfThoughts    AgenticStrategy = "tree_of_thoughts"
	StrategySelfConsistency   AgenticStrategy = "self_consistency"
	StrategyGraphOfThoughts   AgenticStrategy = "graph_of_thoughts"
	StrategyReAct             AgenticStrategy = "react"
	StrategyReflexion         AgenticStrategy = "reflexion"
	StrategyLeastToMost       AgenticStrategy = "least_to_most"
	StrategyPlanAndSolve      AgenticStrategy = "plan_and_solve"
	StrategyStepBack          AgenticStrategy = "step_back"
	StrategySkeletonOfThought AgenticStrategy = "skeleton_of_thought"
	StrategyCubicThinking     AgenticStrategy = "cubic_thinking"
	StrategyEmotionPrompting  AgenticStrategy = "emotion_prompting"
	StrategyRolePrompting     AgenticStrategy = "role_prompting"
	StrategyContrastiveCoT    AgenticStrategy = "contrastive_cot"
	StrategyActivePrompting   AgenticStrategy = "active_prompting"
	StrategyAutomaticCoT      AgenticStrategy = "automatic_cot"
)

// ThoughtNode nó em uma árvore de pensamentos
type ThoughtNode struct {
	ID       string
	Content  string
	Score    float64
	Depth    int
	Parent   *ThoughtNode
	Children []*ThoughtNode
	Metadata map[string]interface{}
}

// AgenticTask tarefa para modelo seguidor
type AgenticTask struct {
	ID             string
	Description    string
	Context        string
	Constraints    []string
	ExpectedOutput string
	Priority       int
}

// AgenticResult resultado de uma tarefa
type AgenticResult struct {
	TaskID     string
	Output     string
	Confidence float64
	Metadata   map[string]interface{}
}

// AgenticConfig configurações para raciocínio agêntico
type AgenticConfig struct {
	Strategy     AgenticStrategy
	MaxDepth     int    // Profundidade máxima da árvore
	BranchFactor int    // Fator de ramificação
	NumSamples   int    // Número de amostras para self-consistency
	VotingMethod string // "majority", "weighted", "consensus"
	UseBrain     bool   // Usar modelo cérebro (DisCIPL)
	MaxFollowers int    // Número máximo de seguidores
	Timeout      int    // Timeout em segundos
	Verbose      bool
}

// TreeOfThoughts implementa estratégia ToT
type TreeOfThoughts struct {
	Config *AgenticConfig
	Model  *TransformerModel
	Root   *ThoughtNode
}

// NewTreeOfThoughts cria nova instância ToT
func NewTreeOfThoughts(model *TransformerModel, config *AgenticConfig) *TreeOfThoughts {
	return &TreeOfThoughts{
		Config: config,
		Model:  model,
		Root: &ThoughtNode{
			ID:       "root",
			Content:  "",
			Score:    0.0,
			Depth:    0,
			Children: make([]*ThoughtNode, 0),
			Metadata: make(map[string]interface{}),
		},
	}
}

// Solve resolve problema usando Tree of Thoughts
func (tot *TreeOfThoughts) Solve(problem string) (*ThoughtNode, error) {
	// 1. Criar raiz da árvore
	tot.Root.Content = problem
	tot.Root.Metadata["type"] = "problem"

	// 2. Expandir árvore
	err := tot.expandTree(tot.Root, 0)
	if err != nil {
		return nil, err
	}

	// 3. Encontrar melhor caminho
	bestLeaf := tot.findBestLeaf(tot.Root)

	return bestLeaf, nil
}

// expandTree expande a árvore de pensamentos
func (tot *TreeOfThoughts) expandTree(node *ThoughtNode, depth int) error {
	// Verificar profundidade máxima
	if depth >= tot.Config.MaxDepth {
		return nil
	}

	// Gerar pensamentos filhos
	for i := 0; i < tot.Config.BranchFactor; i++ {
		childContent := tot.generateThought(node.Content, i)

		child := &ThoughtNode{
			ID:       fmt.Sprintf("%s_child_%d", node.ID, i),
			Content:  childContent,
			Score:    tot.evaluateThought(childContent),
			Depth:    depth + 1,
			Parent:   node,
			Children: make([]*ThoughtNode, 0),
			Metadata: make(map[string]interface{}),
		}

		node.Children = append(node.Children, child)

		// Recursivamente expandir
		err := tot.expandTree(child, depth+1)
		if err != nil {
			return err
		}
	}

	return nil
}

// generateThought gera um pensamento
func (tot *TreeOfThoughts) generateThought(context string, variant int) string {
	// Em produção: usar modelo para gerar pensamento
	// Aqui: simulação simplificada
	thoughts := []string{
		fmt.Sprintf("Approach 1: Analyze the problem step by step from %s", context),
		fmt.Sprintf("Approach 2: Consider alternative perspectives on %s", context),
		fmt.Sprintf("Approach 3: Break down %s into sub-problems", context),
		fmt.Sprintf("Approach 4: Use analogy to solve %s", context),
	}

	if variant < len(thoughts) {
		return thoughts[variant]
	}
	return fmt.Sprintf("Approach %d: Creative solution for %s", variant+1, context)
}

// evaluateThought avalia qualidade de um pensamento
func (tot *TreeOfThoughts) evaluateThought(thought string) float64 {
	// Em produção: usar modelo de scoring ou heurísticas
	// Aqui: simulação baseada em comprimento e palavras-chave

	score := 0.5 // Base score

	// Bonus por comprimento (pensamentos mais detalhados)
	if len(thought) > 50 {
		score += 0.2
	}

	// Bonus por palavras-chave de raciocínio
	reasoningWords := []string{"step", "analyze", "therefore", "because", "thus"}
	for _, word := range reasoningWords {
		if strings.Contains(strings.ToLower(thought), word) {
			score += 0.1
		}
	}

	return math.Min(1.0, score)
}

// findBestLeaf encontra folha com maior score
func (tot *TreeOfThoughts) findBestLeaf(node *ThoughtNode) *ThoughtNode {
	// Se é folha, retornar
	if len(node.Children) == 0 {
		return node
	}

	// Recursivamente encontrar melhor filho
	bestChild := node.Children[0]
	for _, child := range node.Children[1:] {
		bestGrandchild := tot.findBestLeaf(child)
		if bestGrandchild.Score > bestChild.Score {
			bestChild = bestGrandchild
		}
	}

	return bestChild
}

// SelfConsistency implementa votação por consistência
type SelfConsistency struct {
	Config *AgenticConfig
	Model  *TransformerModel
}

// NewSelfConsistency cria nova instância
func NewSelfConsistency(model *TransformerModel, config *AgenticConfig) *SelfConsistency {
	return &SelfConsistency{
		Config: config,
		Model:  model,
	}
}

// Solve resolve com self-consistency
func (sc *SelfConsistency) Solve(problem string) (string, float64) {
	// 1. Gerar múltiplas respostas
	answers := make([]string, 0, sc.Config.NumSamples)
	for i := 0; i < sc.Config.NumSamples; i++ {
		answer := sc.generateAnswer(problem, i)
		answers = append(answers, answer)
	}

	// 2. Votar
	winner, confidence := sc.vote(answers)

	return winner, confidence
}

// generateAnswer gera uma resposta
func (sc *SelfConsistency) generateAnswer(problem string, seed int) string {
	// Em produção: sample do modelo com diferentes seeds
	// Aqui: simulação
	responses := []string{
		fmt.Sprintf("Answer to %s: Solution A (reasoning path 1)", problem),
		fmt.Sprintf("Answer to %s: Solution A (reasoning path 2)", problem),
		fmt.Sprintf("Answer to %s: Solution B (reasoning path 3)", problem),
		fmt.Sprintf("Answer to %s: Solution A (reasoning path 4)", problem),
		fmt.Sprintf("Answer to %s: Solution C (reasoning path 5)", problem),
	}

	if seed < len(responses) {
		return responses[seed]
	}
	return fmt.Sprintf("Answer to %s: Solution A (reasoning path %d)", problem, seed+1)
}

// vote realiza votação
func (sc *SelfConsistency) vote(answers []string) (string, float64) {
	if len(answers) == 0 {
		return "", 0.0
	}

	// Contar frequência de cada resposta
	counts := make(map[string]int)
	for _, answer := range answers {
		// Extrair solução base
		solution := extractBaseSolution(answer)
		counts[solution]++
	}

	// Encontrar maioria
	majority := ""
	maxCount := 0
	for solution, count := range counts {
		if count > maxCount {
			maxCount = count
			majority = solution
		}
	}

	confidence := float64(maxCount) / float64(len(answers))

	return majority, confidence
}

// extractBaseSolution extrai solução base de resposta
func extractBaseSolution(answer string) string {
	// Extrair "Solution X" da resposta
	parts := strings.Split(answer, ":")
	if len(parts) > 1 {
		solutionPart := strings.TrimSpace(parts[1])
		// Pegar até parênteses
		if idx := strings.Index(solutionPart, "("); idx != -1 {
			return strings.TrimSpace(solutionPart[:idx])
		}
		return solutionPart
	}
	return answer
}

// DisCIPLBrain cérebro do framework DisCIPL
type DisCIPLBrain struct {
	Model          *TransformerModel
	FollowerModels []*TransformerModel
	Config         *AgenticConfig
}

// NewDisCIPLBrain cria cérebro DisCIPL
func NewDisCIPLBrain(brainModel *TransformerModel, config *AgenticConfig) *DisCIPLBrain {
	return &DisCIPLBrain{
		Model:          brainModel,
		Config:         config,
		FollowerModels: make([]*TransformerModel, 0),
	}
}

// AddFollower adiciona modelo seguidor
func (brain *DisCIPLBrain) AddFollower(follower *TransformerModel) {
	brain.FollowerModels = append(brain.FollowerModels, follower)
}

// PlanAndDistribute planeja e distribui tarefas (DisCIPL)
func (brain *DisCIPLBrain) PlanAndDistribute(problem string) ([]AgenticResult, error) {
	// 1. Cérebro analisa problema e cria plano
	tasks := brain.createPlan(problem)

	// 2. Distribuir tarefas para seguidores
	results := make([]AgenticResult, 0)

	for i, task := range tasks {
		// Selecionar seguidor (round-robin)
		followerIdx := i % len(brain.FollowerModels)
		follower := brain.FollowerModels[followerIdx]

		// Seguidor executa tarefa
		result := brain.executeTask(follower, task)
		results = append(results, result)
	}

	// 3. Cérebro sintetiza resultados
	finalResult := brain.synthesizeResults(results)

	return []AgenticResult{finalResult}, nil
}

// createPlan cria plano de execução
func (brain *DisCIPLBrain) createPlan(problem string) []AgenticTask {
	// Em produção: usar LLM forte para criar plano
	// Aqui: heurística simplificada

	tasks := []AgenticTask{
		{
			ID:             "task_1",
			Description:    "Understand the problem",
			Context:        problem,
			ExpectedOutput: "Problem analysis",
			Priority:       1,
		},
		{
			ID:             "task_2",
			Description:    "Identify key components",
			Context:        problem,
			ExpectedOutput: "Component list",
			Priority:       2,
		},
		{
			ID:             "task_3",
			Description:    "Develop solution strategy",
			Context:        problem,
			ExpectedOutput: "Solution approach",
			Priority:       3,
		},
		{
			ID:             "task_4",
			Description:    "Execute solution",
			Context:        problem,
			ExpectedOutput: "Final answer",
			Priority:       4,
		},
	}

	return tasks
}

// executeTask executa tarefa com seguidor
func (brain *DisCIPLBrain) executeTask(follower *TransformerModel, task AgenticTask) AgenticResult {
	// Em produção: usar modelo seguidor
	// Aqui: simulação
	output := fmt.Sprintf("Follower result for task '%s': %s", task.ID, task.ExpectedOutput)

	return AgenticResult{
		TaskID:     task.ID,
		Output:     output,
		Confidence: 0.85,
		Metadata: map[string]interface{}{
			"task_description": task.Description,
			"model_used":       "follower",
		},
	}
}

// synthesizeResults sintetiza resultados
func (brain *DisCIPLBrain) synthesizeResults(results []AgenticResult) AgenticResult {
	// Cérebro combina resultados dos seguidores
	combined := "Brain synthesis of follower results:\n"
	for _, result := range results {
		combined += fmt.Sprintf("- %s\n", result.Output)
	}

	return AgenticResult{
		TaskID:     "synthesis",
		Output:     combined,
		Confidence: 0.90,
		Metadata: map[string]interface{}{
			"num_tasks": len(results),
			"method":    "discipl_synthesis",
		},
	}
}

// MultiStrategyOrchestrator orquestrador de múltiplas estratégias
type MultiStrategyOrchestrator struct {
	Config     *AgenticConfig
	Strategies map[AgenticStrategy]interface{}
}

// NewMultiStrategyOrchestrator cria orquestrador
func NewMultiStrategyOrchestrator(config *AgenticConfig) *MultiStrategyOrchestrator {
	return &MultiStrategyOrchestrator{
		Config:     config,
		Strategies: make(map[AgenticStrategy]interface{}),
	}
}

// RegisterStrategy registra estratégia
func (orch *MultiStrategyOrchestrator) RegisterStrategy(
	strategy AgenticStrategy,
	impl interface{},
) {
	orch.Strategies[strategy] = impl
}

// Solve resolve usando estratégia específica
func (orch *MultiStrategyOrchestrator) Solve(
	problem string,
	strategy AgenticStrategy,
) (interface{}, error) {
	impl, ok := orch.Strategies[strategy]
	if !ok {
		return nil, fmt.Errorf("strategy %s not registered", strategy)
	}

	switch s := impl.(type) {
	case *TreeOfThoughts:
		return s.Solve(problem)
	case *SelfConsistency:
		answer, confidence := s.Solve(problem)
		return map[string]interface{}{
			"answer":     answer,
			"confidence": confidence,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported strategy type")
	}
}

// SolveWithEnsemble resolve com ensemble de estratégias
func (orch *MultiStrategyOrchestrator) SolveWithEnsemble(
	problem string,
	strategies []AgenticStrategy,
) (string, error) {
	results := make([]string, 0)

	for _, strategy := range strategies {
		result, err := orch.Solve(problem, strategy)
		if err != nil {
			continue
		}

		// Extrair resposta
		switch r := result.(type) {
		case *ThoughtNode:
			results = append(results, r.Content)
		case map[string]interface{}:
			if answer, ok := r["answer"].(string); ok {
				results = append(results, answer)
			}
		}
	}

	if len(results) == 0 {
		return "", fmt.Errorf("all strategies failed")
	}

	// Voting
	votes := make(map[string]int)
	for _, result := range results {
		votes[result]++
	}

	// Encontrar maioria
	winner := ""
	maxVotes := 0
	for result, count := range votes {
		if count > maxVotes {
			maxVotes = count
			winner = result
		}
	}

	return winner, nil
}

// GetAvailableStrategies retorna estratégias disponíveis
func GetAvailableStrategies() []AgenticStrategy {
	return []AgenticStrategy{
		StrategyChainOfThought,
		StrategyTreeOfThoughts,
		StrategySelfConsistency,
		StrategyGraphOfThoughts,
		StrategyReAct,
		StrategyReflexion,
		StrategyLeastToMost,
		StrategyPlanAndSolve,
		StrategyStepBack,
		StrategySkeletonOfThought,
		StrategyCubicThinking,
		StrategyEmotionPrompting,
		StrategyRolePrompting,
		StrategyContrastiveCoT,
		StrategyActivePrompting,
		StrategyAutomaticCoT,
	}
}

// GetStrategyDescription retorna descrição da estratégia
func GetStrategyDescription(strategy AgenticStrategy) string {
	descriptions := map[AgenticStrategy]string{
		StrategyChainOfThought:    "Linear step-by-step reasoning",
		StrategyTreeOfThoughts:    "Tree-based exploration of multiple reasoning paths",
		StrategySelfConsistency:   "Multiple samples with majority voting",
		StrategyGraphOfThoughts:   "Graph-based reasoning with cycles",
		StrategyReAct:             "Reasoning + Action interleaved",
		StrategyReflexion:         "Self-reflection and improvement",
		StrategyLeastToMost:       "Solve sub-problems from easy to hard",
		StrategyPlanAndSolve:      "Plan first, then execute",
		StrategyStepBack:          "Step back to broader context first",
		StrategySkeletonOfThought: "Generate skeleton, then flesh out",
		StrategyCubicThinking:     "3D thinking: depth, breadth, perspective",
		StrategyEmotionPrompting:  "Add emotional context to reasoning",
		StrategyRolePrompting:     "Adopt specific role/persona",
		StrategyContrastiveCoT:    "Compare correct vs incorrect reasoning",
		StrategyActivePrompting:   "Dynamically select best examples",
		StrategyAutomaticCoT:      "Automatically generate CoT examples",
	}

	if desc, ok := descriptions[strategy]; ok {
		return desc
	}
	return "Unknown strategy"
}

// SortThoughtNodesByScore ordena nós por score
func SortThoughtNodesByScore(nodes []*ThoughtNode) {
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].Score > nodes[j].Score
	})
}
