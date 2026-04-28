package model

import (
	"fmt"
	"regexp"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// GrammarRule define uma regra gramatical
type GrammarRule struct {
	Name      string
	Pattern   string
	Required  bool
	MinLength int
	MaxLength int
	NextRules []string // Regras que podem seguir esta
}

// GrammarSchema define um schema gramatical completo
type GrammarSchema struct {
	Name        string
	Description string
	Rules       map[string]*GrammarRule
	StartRule   string
	Ordered     bool // Se true, regras devem aparecer em ordem
}

// StructuredReasoning raciocínio estruturado
type StructuredReasoning struct {
	Goal       string   // O que precisa ser resolvido
	Approach   string   // Como vai resolver
	Steps      []string // Passos do raciocínio
	Edges      []string // Casos especiais/edge cases
	Conclusion string   // Conclusão final
}

// GrammarConstrainedConfig configurações para geração com gramática
type GrammarConstrainedConfig struct {
	Schema         *GrammarSchema
	MaxTokens      int
	Temperature    float64
	EnforceGrammar bool // Se true, rejeita tokens que violam grammar
}

// GrammarConstrainedDecoder decoder com restrição gramatical
type GrammarConstrainedDecoder struct {
	Model         *TransformerModel
	Config        *GrammarConstrainedConfig
	CurrentRule   string
	GeneratedText string
	RulePositions map[string]int // Posição atual em cada regra
}

// PredefinedGrammarSchemas schemas predefinidos
var PredefinedGrammarSchemas = map[string]*GrammarSchema{
	"reasoning": {
		Name:        "Structured Reasoning",
		Description: "GOAL/APPROACH/STEPS/EDGES/CONCLUSION format",
		Rules: map[string]*GrammarRule{
			"GOAL": {
				Name:      "GOAL",
				Pattern:   `GOAL:\s*(.+?)`,
				Required:  true,
				MinLength: 10,
				MaxLength: 200,
				NextRules: []string{"APPROACH"},
			},
			"APPROACH": {
				Name:      "APPROACH",
				Pattern:   `APPROACH:\s*(.+?)`,
				Required:  true,
				MinLength: 20,
				MaxLength: 300,
				NextRules: []string{"STEPS"},
			},
			"STEPS": {
				Name:      "STEPS",
				Pattern:   `STEPS:\s*(.+?)`,
				Required:  true,
				MinLength: 30,
				MaxLength: 500,
				NextRules: []string{"EDGES"},
			},
			"EDGES": {
				Name:      "EDGES",
				Pattern:   `EDGES:\s*(.+?)`,
				Required:  false,
				MinLength: 10,
				MaxLength: 200,
				NextRules: []string{"CONCLUSION"},
			},
			"CONCLUSION": {
				Name:      "CONCLUSION",
				Pattern:   `CONCLUSION:\s*(.+?)`,
				Required:  true,
				MinLength: 10,
				MaxLength: 200,
				NextRules: []string{},
			},
		},
		StartRule: "GOAL",
		Ordered:   true,
	},
	"math_proof": {
		Name:        "Mathematical Proof",
		Description: "THEOREM/PROOF/QED format",
		Rules: map[string]*GrammarRule{
			"THEOREM": {
				Name:      "THEOREM",
				Pattern:   `THEOREM:\s*(.+?)`,
				Required:  true,
				MinLength: 20,
				MaxLength: 300,
				NextRules: []string{"PROOF"},
			},
			"PROOF": {
				Name:      "PROOF",
				Pattern:   `PROOF:\s*(.+?)`,
				Required:  true,
				MinLength: 50,
				MaxLength: 1000,
				NextRules: []string{"QED"},
			},
			"QED": {
				Name:      "QED",
				Pattern:   `QED:\s*(.+?)`,
				Required:  true,
				MinLength: 10,
				MaxLength: 100,
				NextRules: []string{},
			},
		},
		StartRule: "THEOREM",
		Ordered:   true,
	},
	"code_solution": {
		Name:        "Code Solution",
		Description: "PROBLEM/ALGORITHM/CODE/COMPLEXITY format",
		Rules: map[string]*GrammarRule{
			"PROBLEM": {
				Name:      "PROBLEM",
				Pattern:   `PROBLEM:\s*(.+?)`,
				Required:  true,
				MinLength: 20,
				MaxLength: 300,
				NextRules: []string{"ALGORITHM"},
			},
			"ALGORITHM": {
				Name:      "ALGORITHM",
				Pattern:   `ALGORITHM:\s*(.+?)`,
				Required:  true,
				MinLength: 30,
				MaxLength: 400,
				NextRules: []string{"CODE"},
			},
			"CODE": {
				Name:      "CODE",
				Pattern:   "CODE:\\s*```[\\w]*\\n(.+?)```",
				Required:  true,
				MinLength: 50,
				MaxLength: 2000,
				NextRules: []string{"COMPLEXITY"},
			},
			"COMPLEXITY": {
				Name:      "COMPLEXITY",
				Pattern:   `COMPLEXITY:\s*(.+?)`,
				Required:  true,
				MinLength: 20,
				MaxLength: 200,
				NextRules: []string{},
			},
		},
		StartRule: "PROBLEM",
		Ordered:   true,
	},
}

// NewGrammarConstrainedDecoder cria um novo decoder com restrição gramatical
func NewGrammarConstrainedDecoder(
	model *TransformerModel,
	schemaName string,
) (*GrammarConstrainedDecoder, error) {
	schema, ok := PredefinedGrammarSchemas[schemaName]
	if !ok {
		return nil, fmt.Errorf("grammar schema '%s' not found", schemaName)
	}

	config := &GrammarConstrainedConfig{
		Schema:         schema,
		MaxTokens:      500,
		Temperature:    0.7,
		EnforceGrammar: true,
	}

	return &GrammarConstrainedDecoder{
		Model:         model,
		Config:        config,
		CurrentRule:   schema.StartRule,
		GeneratedText: "",
		RulePositions: make(map[string]int),
	}, nil
}

// GenerateStructuredResponse gera resposta com estrutura gramatical
func (decoder *GrammarConstrainedDecoder) GenerateStructuredResponse(
	prompt string,
) (string, *StructuredReasoning, error) {
	// Construir prompt com estrutura esperada
	fullPrompt := decoder.buildStructuredPrompt(prompt)

	// Gerar texto com restrição gramatical
	generatedText, err := decoder.generateWithGrammar(fullPrompt)
	if err != nil {
		return "", nil, err
	}

	// Parsear estrutura
	structure, err := decoder.parseStructuredReasoning(generatedText)
	if err != nil {
		return generatedText, nil, err
	}

	return generatedText, structure, nil
}

// buildStructuredPrompt constrói prompt com estrutura esperada
func (decoder *GrammarConstrainedDecoder) buildStructuredPrompt(prompt string) string {
	schema := decoder.Config.Schema

	header := fmt.Sprintf("Please respond using the following structure:\n")

	for ruleName, rule := range schema.Rules {
		required := "Required"
		if !rule.Required {
			required = "Optional"
		}
		header += fmt.Sprintf("- %s (%s): %d-%d characters\n",
			ruleName, required, rule.MinLength, rule.MaxLength)
	}

	header += "\n"

	// Adicionar seções vazias para guiar o modelo
	if schema.Name == "Structured Reasoning" {
		header += fmt.Sprintf("%s\n\nAPPROACH:\n\nSTEPS:\n\nEDGES:\n\nCONCLUSION:\n",
			"GOAL:")
	}

	return prompt + "\n\n" + header
}

// generateWithGrammar gera texto com restrição gramatical
func (decoder *GrammarConstrainedDecoder) generateWithGrammar(
	prompt string,
) (string, error) {
	// Tokenizar prompt
	tokens := decoder.tokenizeText(prompt)

	// Gerar tokens com validação gramatical
	maxTokens := decoder.Config.MaxTokens

	for i := 0; i < maxTokens; i++ {
		if len(tokens) == 0 {
			break
		}

		// Forward pass
		hidden := decoder.Model.Forward(tokens)
		seqLen := len(tokens)

		// Última posição
		lastHidden := mat.NewDense(1, decoder.Model.DModel, nil)
		for j := 0; j < decoder.Model.DModel; j++ {
			lastHidden.Set(0, j, hidden.At(seqLen-1, j))
		}

		// Calcular logits
		logits := mat.NewDense(1, decoder.Model.VocabSize, nil)
		logits.Mul(lastHidden, decoder.Model.WOut.T())

		// Obter probabilidade do próximo token
		logitValues := make([]float64, decoder.Model.VocabSize)
		for j := 0; j < decoder.Model.VocabSize; j++ {
			logitValues[j] = logits.At(0, j) / decoder.Config.Temperature
		}

		// Aplicar restrição gramatical
		if decoder.Config.EnforceGrammar {
			logitValues = decoder.applyGrammarConstraint(logitValues, tokens)
		}

		// Softmax e sample
		probs := softmax(logitValues)
		nextToken := sampleFromDistribution(probs)

		tokens = append(tokens, nextToken)

		// Verificar se completou todas as regras
		if decoder.grammarComplete() {
			break
		}
	}

	// Converter para texto
	generatedText := decoder.tokensToText(tokens)
	decoder.GeneratedText = generatedText

	return generatedText, nil
}

// applyGrammarConstraint aplica restrição gramatical aos logits
func (decoder *GrammarConstrainedDecoder) applyGrammarConstraint(
	logits []float64,
	currentTokens []int,
) []float64 {
	// Obter texto gerado até agora
	currentText := decoder.tokensToText(currentTokens)

	// Verificar qual regra estamos
	currentRule := decoder.getCurrentRule(currentText)

	if currentRule == nil {
		return logits // Sem restrição
	}

	// Verificar se precisa iniciar próxima regra
	needsNextRule := decoder.shouldStartNextRule(currentText, currentRule)

	if needsNextRule {
		// Penalizar tokens que não iniciam próxima seção
		nextRuleName := decoder.getNextRuleName(currentRule)
		if nextRuleName != "" {
			// Em produção: restringir vocabulário para tokens que iniciam seção
			// Aqui: simulação simplificada
			decoder.CurrentRule = nextRuleName
		}
	}

	return logits
}

// getCurrentRule obtém regra atual baseada no texto gerado
func (decoder *GrammarConstrainedDecoder) getCurrentRule(text string) *GrammarRule {
	schema := decoder.Config.Schema

	for ruleName, rule := range schema.Rules {
		// Verificar se esta regra já foi completada
		pattern := regexp.MustCompile(ruleName + `:`)
		if pattern.MatchString(text) {
			// Verificar se próxima regra começou
			for _, nextRule := range rule.NextRules {
				nextPattern := regexp.MustCompile(nextRule + `:`)
				if nextPattern.MatchString(text) {
					// Próxima regra já começou
					continue
				}
			}
			return schema.Rules[ruleName]
		}
	}

	return schema.Rules[schema.StartRule]
}

// shouldStartNextRule verifica se deve iniciar próxima regra
func (decoder *GrammarConstrainedDecoder) shouldStartNextRule(
	text string,
	currentRule *GrammarRule,
) bool {
	// Verificar se conteúdo atual atende requisitos mínimos
	if len(text) < currentRule.MinLength {
		return false
	}

	// Verificar se excedeu máximo
	if len(text) > currentRule.MaxLength {
		return true
	}

	// Verificar se há indicadores de conclusão
	conclusionIndicators := []string{
		"\n\n",
		"Therefore",
		"Thus",
		"In conclusion",
	}

	for _, indicator := range conclusionIndicators {
		if strings.HasSuffix(text, indicator) {
			return true
		}
	}

	return false
}

// getNextRuleName obtém nome da próxima regra
func (decoder *GrammarConstrainedDecoder) getNextRuleName(rule *GrammarRule) string {
	if len(rule.NextRules) > 0 {
		return rule.NextRules[0]
	}
	return ""
}

// grammarComplete verifica se completou todas as regras
func (decoder *GrammarConstrainedDecoder) grammarComplete() bool {
	schema := decoder.Config.Schema
	currentText := decoder.GeneratedText

	// Verificar se todas as regras required foram satisfeitas
	for ruleName, rule := range schema.Rules {
		if rule.Required {
			pattern := regexp.MustCompile(ruleName + `:`)
			if !pattern.MatchString(currentText) {
				return false
			}
		}
	}

	// Se chegou aqui e a última regra não tem next, está completo
	currentRule := decoder.getCurrentRule(currentText)
	return currentRule != nil && len(currentRule.NextRules) == 0
}

// parseStructuredReasoning parseia texto para estrutura
func (decoder *GrammarConstrainedDecoder) parseStructuredReasoning(
	text string,
) (*StructuredReasoning, error) {
	reasoning := &StructuredReasoning{}

	// Extrair GOAL
	goalPattern := regexp.MustCompile(`GOAL:\s*(.+?)(?:APPROACH:|$)`)
	if matches := goalPattern.FindStringSubmatch(text); len(matches) > 1 {
		reasoning.Goal = strings.TrimSpace(matches[1])
	}

	// Extrair APPROACH
	approachPattern := regexp.MustCompile(`APPROACH:\s*(.+?)(?:STEPS:|$)`)
	if matches := approachPattern.FindStringSubmatch(text); len(matches) > 1 {
		reasoning.Approach = strings.TrimSpace(matches[1])
	}

	// Extrair STEPS
	stepsPattern := regexp.MustCompile(`STEPS:\s*(.+?)(?:EDGES:|CONCLUSION:|$)`)
	if matches := stepsPattern.FindStringSubmatch(text); len(matches) > 1 {
		stepsText := strings.TrimSpace(matches[1])
		reasoning.Steps = strings.Split(stepsText, "\n")
	}

	// Extrair EDGES
	edgesPattern := regexp.MustCompile(`EDGES:\s*(.+?)(?:CONCLUSION:|$)`)
	if matches := edgesPattern.FindStringSubmatch(text); len(matches) > 1 {
		edgesText := strings.TrimSpace(matches[1])
		reasoning.Edges = strings.Split(edgesText, "\n")
	}

	// Extrair CONCLUSION
	conclusionPattern := regexp.MustCompile(`CONCLUSION:\s*(.+?)$`)
	if matches := conclusionPattern.FindStringSubmatch(text); len(matches) > 1 {
		reasoning.Conclusion = strings.TrimSpace(matches[1])
	}

	return reasoning, nil
}

// tokenizeText tokeniza texto (simplificado)
func (decoder *GrammarConstrainedDecoder) tokenizeText(text string) []int {
	// Em produção: usar tokenizer real
	words := strings.Fields(text)
	tokens := make([]int, 0, len(words))

	for _, word := range words {
		token := hashStringForGrammar(word) % decoder.Model.VocabSize
		tokens = append(tokens, token)
	}

	return tokens
}

// tokensToText converte tokens para texto
func (decoder *GrammarConstrainedDecoder) tokensToText(tokens []int) string {
	words := make([]string, 0)
	for _, token := range tokens {
		if token < len(decoder.Model.Vocab) {
			words = append(words, decoder.Model.Vocab[token])
		}
	}
	return strings.Join(words, " ")
}

// ValidateGrammar valida se texto segue grammar schema
func ValidateGrammar(text string, schemaName string) (bool, []string) {
	schema, ok := PredefinedGrammarSchemas[schemaName]
	if !ok {
		return false, []string{fmt.Sprintf("Schema '%s' not found", schemaName)}
	}

	errors := make([]string, 0)

	// Verificar cada regra
	for ruleName, rule := range schema.Rules {
		pattern := regexp.MustCompile(ruleName + `:`)

		if rule.Required && !pattern.MatchString(text) {
			errors = append(errors, fmt.Sprintf("Missing required section: %s", ruleName))
		}

		// Verificar comprimento
		if pattern.MatchString(text) {
			sectionPattern := regexp.MustCompile(
				fmt.Sprintf(`%s:\s*(.+?)(?:%s:|$)`,
					ruleName,
					strings.Join(rule.NextRules, "|")),
			)
			if matches := sectionPattern.FindStringSubmatch(text); len(matches) > 1 {
				content := strings.TrimSpace(matches[1])
				if len(content) < rule.MinLength {
					errors = append(errors,
						fmt.Sprintf("%s too short: %d chars (min %d)",
							ruleName, len(content), rule.MinLength))
				}
				if len(content) > rule.MaxLength {
					errors = append(errors,
						fmt.Sprintf("%s too long: %d chars (max %d)",
							ruleName, len(content), rule.MaxLength))
				}
			}
		}
	}

	// Verificar ordem se necessário
	if schema.Ordered {
		lastPos := -1
		ruleNames := getOrderedRuleNames(schema)

		for _, ruleName := range ruleNames {
			pattern := regexp.MustCompile(ruleName + `:`)
			if matches := pattern.FindStringIndex(text); len(matches) > 0 {
				if matches[0] < lastPos {
					errors = append(errors,
						fmt.Sprintf("Rules out of order: %s appears before previous rule", ruleName))
				}
				lastPos = matches[0]
			}
		}
	}

	return len(errors) == 0, errors
}

// getOrderedRuleNames obtém nomes de regras em ordem
func getOrderedRuleNames(schema *GrammarSchema) []string {
	ordered := make([]string, 0)
	current := schema.StartRule

	visited := make(map[string]bool)
	for current != "" && !visited[current] {
		visited[current] = true
		ordered = append(ordered, current)

		rule := schema.Rules[current]
		if len(rule.NextRules) > 0 {
			current = rule.NextRules[0]
		} else {
			break
		}
	}

	return ordered
}

// hashStringForGrammar cria hash para gramática
func hashStringForGrammar(s string) int {
	hash := 0
	for _, c := range s {
		hash = hash*31 + int(c)
	}
	if hash < 0 {
		return -hash
	}
	return hash
}

// GetGrammarStats retorna estatísticas da gramática
func GetGrammarStats(schemaName string) map[string]interface{} {
	schema, ok := PredefinedGrammarSchemas[schemaName]
	if !ok {
		return nil
	}

	stats := map[string]interface{}{
		"name":        schema.Name,
		"description": schema.Description,
		"num_rules":   len(schema.Rules),
		"ordered":     schema.Ordered,
		"start_rule":  schema.StartRule,
	}

	ruleStats := make(map[string]interface{})
	for name, rule := range schema.Rules {
		ruleStats[name] = map[string]interface{}{
			"required":   rule.Required,
			"min_length": rule.MinLength,
			"max_length": rule.MaxLength,
			"next_rules": rule.NextRules,
		}
	}
	stats["rules"] = ruleStats

	return stats
}
