package model

import (
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strings"
)

// Softmax calcula a função softmax sobre um vetor de logits
func Softmax(x []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range x {
		if v > max {
			max = v
		}
	}

	res := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}

	for i := range res {
		res[i] /= sum
	}

	return res
}

// CrossEntropyLoss calcula a perda de cross-entropy
func CrossEntropyLoss(probs []float64, target int) float64 {
	if target < 0 || target >= len(probs) {
		return 0.0
	}
	// -log(p_target)
	p := probs[target]
	if p <= 0 {
		return 10.0 // Large penalty for zero probability
	}
	return -math.Log(p)
}

// Sample amostra um índice baseado nas probabilidades
func Sample(probs []float64) int {
	r := rand.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// SampleTopK amostra considerando apenas o top-K probabilidades mais altas
func SampleTopK(probs []float64, k int) int {
	if k >= len(probs) {
		return Sample(probs)
	}

	// Criar lista de (probabilidade, índice)
	type probIdx struct {
		prob float64
		idx  int
	}

	pairs := make([]probIdx, len(probs))
	for i, p := range probs {
		pairs[i] = probIdx{p, i}
	}

	// Ordenar por probabilidade (decrescente)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].prob > pairs[j].prob
	})

	// Pegar top-K
	topK := pairs[:k]

	// Renormalizar probabilidades
	sum := 0.0
	for _, p := range topK {
		sum += p.prob
	}

	if sum == 0 {
		return topK[rand.Intn(len(topK))].idx
	}

	// Amostrar do top-K renormalizado
	r := rand.Float64()
	cumsum := 0.0
	for _, p := range topK {
		cumsum += p.prob / sum
		if r <= cumsum {
			return p.idx
		}
	}

	return topK[len(topK)-1].idx
}

// ApplyRepetitionPenalty aplica penalidade de repetição aos logits
// Penaliza tokens que já aparecem no contexto gerado
// penalty > 1.0 reduz probabilidade de tokens repetidos
// penalty = 1.0 não aplica penalidade
// penalty < 1.0 incentiva repetição (raramente usado)
func ApplyRepetitionPenalty(logits []float64, generatedTokens []int, penalty float64) []float64 {
	if penalty <= 1.0 || len(generatedTokens) == 0 {
		return logits
	}

	// Contar frequência de cada token no contexto gerado
	tokenCounts := make(map[int]int)
	for _, token := range generatedTokens {
		tokenCounts[token]++
	}

	// Aplicar penalidade aos logits
	penalized := make([]float64, len(logits))
	copy(penalized, logits)

	for tokenID, count := range tokenCounts {
		if tokenID >= 0 && tokenID < len(penalized) {
			// Penalidade mais forte: subtrair um valor fixo por ocorrência
			// Isso é mais eficaz que divisão para evitar repetição
			penaltyAmount := float64(count) * (penalty - 1.0) * 10.0
			penalized[tokenID] -= penaltyAmount
		}
	}

	return penalized
}

// Generator helper para construção eficiente de strings
type Generator struct {
	strings.Builder
}

// NewGenerator cria um novo Generator
func NewGenerator() *Generator {
	return &Generator{}
}

// PreprocessText limpa e normaliza o texto para treinamento
func PreprocessText(text string) string {
	// Converter para minúsculas
	text = strings.ToLower(text)

	// Preservar apenas caracteres permitidos:
	// - Letras minúsculas (a-z)
	// - Letras acentuadas portuguesas (áéíóúãõâêôç)
	// - Espaços
	// - Pontuação básica (.,;:!?)
	allowedPattern := regexp.MustCompile(`[^a-záàâãéèêíïóôõöúçñ\s.,;:!?\-\"\']`)
	text = allowedPattern.ReplaceAllString(text, "")

	// Substituir múltiplos espaços por um único
	spacePattern := regexp.MustCompile(`\s+`)
	text = spacePattern.ReplaceAllString(text, " ")

	// Substituir múltiplas pontuações por uma única
	punctPattern := regexp.MustCompile(`([.,;:!?]){2,}`)
	text = punctPattern.ReplaceAllString(text, "$1")

	// Remover espaços antes de pontuação
	text = strings.ReplaceAll(text, " .", ".")
	text = strings.ReplaceAll(text, " ,", ",")
	text = strings.ReplaceAll(text, " !", "!")
	text = strings.ReplaceAll(text, " ?", "?")

	// Garantir espaço após pontuação (exceto no final)
	text = strings.ReplaceAll(text, ". ", ". ")
	text = strings.ReplaceAll(text, ", ", ", ")

	// Remover espaços no início e fim
	text = strings.TrimSpace(text)

	return text
}
