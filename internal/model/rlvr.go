package model

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// RLVRVerifier interface para verificadores de recompensa
type RLVRVerifier interface {
	// Verify verifica se a resposta está correta e retorna recompensa
	Verify(response string, context string) (float64, map[string]interface{})

	// GetName retorna nome do verificador
	GetName() string
}

// CodeExecutionVerifier verifica execução de código
type CodeExecutionVerifier struct {
	ExpectedOutput string
	Timeout        int
	Language       string
}

func (v *CodeExecutionVerifier) GetName() string {
	return "code_execution"
}

func (v *CodeExecutionVerifier) Verify(response string, context string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})

	// Extrair código da resposta (entre backticks ou tags)
	code := extractCode(response)
	if code == "" {
		details["error"] = "no_code_found"
		return 0.0, details
	}

	// Simular execução (em produção, usaria um sandbox real)
	executedOutput := simulateExecution(code, v.Language)

	// Comparar com output esperado
	if executedOutput == v.ExpectedOutput {
		details["executed_output"] = executedOutput
		details["match"] = true
		return 1.0, details
	}

	// Partial credit para output similar
	similarity := computeStringSimilarity(executedOutput, v.ExpectedOutput)
	details["executed_output"] = executedOutput
	details["expected_output"] = v.ExpectedOutput
	details["similarity"] = similarity
	details["match"] = false

	return similarity * 0.5, details
}

// MathQAVerifier verifica respostas de matemática/QA
type MathQAVerifier struct {
	ExpectedAnswer string
	AcceptFormats  []string // "decimal", "fraction", "percentage"
	Tolerance      float64  // Tolerância numérica
}

func (v *MathQAVerifier) GetName() string {
	return "math_qa"
}

func (v *MathQAVerifier) Verify(response string, context string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})

	// Extrair resposta numérica ou texto
	answer := extractAnswer(response)
	if answer == "" {
		details["error"] = "no_answer_found"
		return 0.0, details
	}

	// Verificar se é numérico
	expectedNum, errExpected := strconv.ParseFloat(v.ExpectedAnswer, 64)
	actualNum, errActual := strconv.ParseFloat(answer, 64)

	if errExpected == nil && errActual == nil {
		// Comparação numérica com tolerância
		diff := abs(expectedNum - actualNum)
		if diff <= v.Tolerance {
			details["expected"] = expectedNum
			details["actual"] = actualNum
			details["diff"] = diff
			details["match"] = true
			return 1.0, details
		}

		details["expected"] = expectedNum
		details["actual"] = actualNum
		details["diff"] = diff
		details["match"] = false

		// Partial credit baseado em quão perto está
		relativeError := diff / (abs(expectedNum) + 1e-10)
		return maxF(0.0, 1.0-relativeError), details
	}

	// Comparação textual (case-insensitive, normalized)
	expectedNorm := normalizeText(v.ExpectedAnswer)
	actualNorm := normalizeText(answer)

	if expectedNorm == actualNorm {
		details["expected"] = expectedNorm
		details["actual"] = actualNorm
		details["match"] = true
		return 1.0, details
	}

	// Fuzzy matching
	similarity := computeStringSimilarity(expectedNorm, actualNorm)
	details["expected"] = expectedNorm
	details["actual"] = actualNorm
	details["similarity"] = similarity
	details["match"] = false

	return similarity * 0.7, details
}

// LogicPuzzleVerifier verifica puzzles de lógica
type LogicPuzzleVerifier struct {
	ExpectedSteps []string
	Solution      string
}

func (v *LogicPuzzleVerifier) GetName() string {
	return "logic_puzzle"
}

func (v *LogicPuzzleVerifier) Verify(response string, context string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})

	// Verificar se resposta final está correta
	if strings.Contains(strings.ToLower(response), strings.ToLower(v.Solution)) {
		details["solution_correct"] = true
	} else {
		details["solution_correct"] = false
	}

	// Verificar se seguiu passos lógicos
	stepsFound := 0
	totalSteps := len(v.ExpectedSteps)

	for _, step := range v.ExpectedSteps {
		if strings.Contains(strings.ToLower(response), strings.ToLower(step)) {
			stepsFound++
		}
	}

	stepRatio := float64(stepsFound) / float64(totalSteps)
	details["steps_found"] = stepsFound
	details["total_steps"] = totalSteps
	details["step_ratio"] = stepRatio

	// Score: 70% solução + 30% raciocínio
	solutionScore := 0.0
	if details["solution_correct"].(bool) {
		solutionScore = 0.7
	}
	reasoningScore := stepRatio * 0.3

	totalScore := solutionScore + reasoningScore
	details["solution_score"] = solutionScore
	details["reasoning_score"] = reasoningScore
	details["total_score"] = totalScore

	return totalScore, details
}

// EnvironmentVerifier para jogos/ambientes interativos
type EnvironmentVerifier struct {
	TargetState     string
	MaxSteps        int
	RewardStructure map[string]float64
}

func (v *EnvironmentVerifier) GetName() string {
	return "environment"
}

func (v *EnvironmentVerifier) Verify(response string, context string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})

	// Parse actions from response
	actions := parseActions(response)

	// Simulate environment
	finalState, intermediateRewards := simulateEnvironment(actions, v.TargetState, v.MaxSteps, v.RewardStructure)

	totalReward := 0.0
	for _, r := range intermediateRewards {
		totalReward += r
	}

	// Normalize to [0, 1]
	maxPossibleReward := float64(v.MaxSteps) * 1.0
	normalizedReward := totalReward / maxPossibleReward

	details["actions"] = len(actions)
	details["final_state"] = finalState
	details["total_reward"] = totalReward
	details["normalized_reward"] = normalizedReward
	details["intermediate_rewards"] = intermediateRewards

	return minF(1.0, maxF(0.0, normalizedReward)), details
}

// CompositeVerifier combina múltiplos verificadores
type CompositeVerifier struct {
	Verifiers []RLVRVerifier
	Weights   []float64
}

func (v *CompositeVerifier) GetName() string {
	return "composite"
}

func (v *CompositeVerifier) Verify(response string, context string) (float64, map[string]interface{}) {
	details := make(map[string]interface{})
	totalScore := 0.0
	totalWeight := 0.0

	verifierResults := make([]map[string]interface{}, 0)

	for i, verifier := range v.Verifiers {
		score, verifierDetails := verifier.Verify(response, context)
		weight := 1.0
		if i < len(v.Weights) {
			weight = v.Weights[i]
		}

		totalScore += score * weight
		totalWeight += weight

		verifierDetails["score"] = score
		verifierDetails["weight"] = weight
		verifierResults = append(verifierResults, verifierDetails)
	}

	// Weighted average
	if totalWeight > 0 {
		totalScore /= totalWeight
	}

	details["verifiers"] = verifierResults
	details["total_score"] = totalScore
	details["total_weight"] = totalWeight

	return totalScore, details
}

// Funções auxiliares

func extractCode(response string) string {
	// Extrair código entre backticks
	re := regexp.MustCompile("```[\\w]*\\n([\\s\\S]*?)```")
	matches := re.FindStringSubmatch(response)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// Ou entre tags <code>
	re = regexp.MustCompile("<code>([\\s\\S]*?)</code>")
	matches = re.FindStringSubmatch(response)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	return ""
}

func simulateExecution(code string, language string) string {
	// Simulação simplificada
	// Em produção, usaria um sandbox real com Docker/gVisor

	if language == "python" {
		// Detectar print statements
		re := regexp.MustCompile(`print\\(([^)]+)\\)`)
		matches := re.FindAllStringSubmatch(code, -1)

		outputs := make([]string, 0)
		for _, match := range matches {
			if len(match) > 1 {
				content := strings.TrimSpace(match[1])

				// String literal
				if strings.HasPrefix(content, "\"") && strings.HasSuffix(content, "\"") {
					outputs = append(outputs, content[1:len(content)-1])
				} else if strings.HasPrefix(content, "'") && strings.HasSuffix(content, "'") {
					outputs = append(outputs, content[1:len(content)-1])
				} else {
					// Variável ou expressão - simular
					outputs = append(outputs, simulateExpression(content))
				}
			}
		}

		return strings.Join(outputs, "\n")
	}

	return "execution_not_supported"
}

func simulateExpression(expr string) string {
	// Simular expressões simples
	if _, err := strconv.ParseFloat(expr, 64); err == nil {
		return expr
	}

	// Expressões aritméticas simples
	re := regexp.MustCompile(`(\\d+)\\s*([+\\-*/])\\s*(\\d+)`)
	matches := re.FindStringSubmatch(expr)
	if len(matches) > 3 {
		a, _ := strconv.ParseFloat(matches[1], 64)
		b, _ := strconv.ParseFloat(matches[3], 64)
		op := matches[2]

		var result float64
		switch op {
		case "+":
			result = a + b
		case "-":
			result = a - b
		case "*":
			result = a * b
		case "/":
			if b != 0 {
				result = a / b
			}
		}

		return strconv.FormatFloat(result, 'f', -1, 64)
	}

	return "unknown"
}

func extractAnswer(response string) string {
	// Extrair resposta após "answer:", "therefore", etc.
	re := regexp.MustCompile("(?i)(?:answer|therefore|result|solution)\\s*:?\\s*([^\\n]+)")
	matches := re.FindStringSubmatch(response)
	if len(matches) > 1 {
		return strings.TrimSpace(matches[1])
	}

	// Ou último número na resposta
	re = regexp.MustCompile(`\d+\.?\d*`)
	matchesStr := re.FindAllString(response, -1)
	if len(matchesStr) > 0 {
		return matchesStr[len(matchesStr)-1]
	}

	return strings.TrimSpace(response)
}

func normalizeText(text string) string {
	text = strings.ToLower(text)
	text = regexp.MustCompile(`\\s+`).ReplaceAllString(text, " ")
	text = strings.TrimSpace(text)
	return text
}

func computeStringSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}

	// Levenshtein distance simplificada
	len1 := len(s1)
	len2 := len(s2)

	if len1 == 0 || len2 == 0 {
		return 0.0
	}

	// Count matching characters
	matches := 0
	minLen := min(len1, len2)
	for i := 0; i < minLen; i++ {
		if s1[i] == s2[i] {
			matches++
		}
	}

	return float64(matches) / float64(max(len1, len2))
}

func parseActions(response string) []string {
	// Parse actions delimitadas por novas linhas ou vírgulas
	lines := strings.Split(response, "\n")
	actions := make([]string, 0)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Split por vírgulas se necessário
		parts := strings.Split(line, ",")
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if part != "" {
				actions = append(actions, part)
			}
		}
	}

	return actions
}

func simulateEnvironment(actions []string, targetState string, maxSteps int, rewards map[string]float64) (string, []float64) {
	// Simulação simplificada de ambiente
	state := "initial"
	intermediateRewards := make([]float64, 0)

	for i, action := range actions {
		if i >= maxSteps {
			break
		}

		// Aplicar ação
		state = applyAction(state, action)

		// Calcular recompensa
		reward := 0.0
		if r, ok := rewards[action]; ok {
			reward = r
		}

		// Bonus por chegar ao target
		if state == targetState {
			reward += 5.0
		}

		intermediateRewards = append(intermediateRewards, reward)
	}

	return state, intermediateRewards
}

func applyAction(state string, action string) string {
	// Simular transição de estado
	return fmt.Sprintf("%s_after_%s", state, action)
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Helper para float min/max
func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func maxF(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
