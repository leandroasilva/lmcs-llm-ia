package model

import (
	"math"
	"testing"
)

// TestApplyRepetitionPenalty_NoPenalty testa quando penalty=1.0 (sem penalidade)
func TestApplyRepetitionPenalty_NoPenalty(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{0, 1, 2}
	penalty := 1.0

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Com penalty=1.0, os logits não devem mudar
	for i := range logits {
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] mudou sem motivo: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_EmptyTokens testa com lista de tokens vazia
func TestApplyRepetitionPenalty_EmptyTokens(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{}
	penalty := 1.5

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Com tokens vazios, os logits não devem mudar
	for i := range logits {
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] mudou com tokens vazios: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_SingleToken testa penalidade com um token gerado uma vez
func TestApplyRepetitionPenalty_SingleToken(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{2} // Token 2 aparece uma vez
	penalty := 1.5

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Apenas o logit[2] deve ser penalizado
	// penaltyAmount = 1 * (1.5 - 1.0) * 10.0 = 5.0
	expectedPenalty := 5.0

	if math.Abs(result[2]-(logits[2]-expectedPenalty)) > 1e-10 {
		t.Errorf("Logit[2] não foi penalizado corretamente: esperado %f, obteve %f", logits[2]-expectedPenalty, result[2])
	}

	// Outros logits não devem mudar
	for i := range logits {
		if i == 2 {
			continue
		}
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] não deveria mudar: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_RepeatedToken testa penalidade com token repetido múltiplas vezes
func TestApplyRepetitionPenalty_RepeatedToken(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{3, 3, 3} // Token 3 aparece 3 vezes
	penalty := 1.3

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// penaltyAmount = 3 * (1.3 - 1.0) * 10.0 = 9.0
	expectedPenalty := 9.0
	expected := logits[3] - expectedPenalty

	if math.Abs(result[3]-expected) > 1e-10 {
		t.Errorf("Logit[3] não foi penalizado corretamente para 3 repetições: esperado %f, obteve %f", expected, result[3])
	}

	// Verificar que a penalidade é proporcional ao número de ocorrências
	// Token 3 aparece 3 vezes, então deve ter penalidade 3x maior que se aparecesse 1 vez
	singleOccurrenceResult := ApplyRepetitionPenalty(logits, []int{3}, penalty)
	expectedDiff := singleOccurrenceResult[3] - result[3]
	expectedDiffValue := 6.0 // 2 * (1.3 - 1.0) * 10.0 = 6.0

	if math.Abs(expectedDiff-expectedDiffValue) > 1e-10 {
		t.Errorf("Diferença de penalidade incorreta: esperado %f, obteve %f", expectedDiffValue, expectedDiff)
	}
}

// TestApplyRepetitionPenalty_MultipleTokens testa penalidade com múltiplos tokens diferentes
func TestApplyRepetitionPenalty_MultipleTokens(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{0, 1, 2, 0, 1} // Token 0 e 1 aparecem 2x, token 2 aparece 1x
	penalty := 1.2

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Token 0: aparece 2 vezes
	// penaltyAmount = 2 * (1.2 - 1.0) * 10.0 = 4.0
	expected0 := logits[0] - 4.0
	if math.Abs(result[0]-expected0) > 1e-10 {
		t.Errorf("Logit[0] incorreto: esperado %f, obteve %f", expected0, result[0])
	}

	// Token 1: aparece 2 vezes
	// penaltyAmount = 2 * (1.2 - 1.0) * 10.0 = 4.0
	expected1 := logits[1] - 4.0
	if math.Abs(result[1]-expected1) > 1e-10 {
		t.Errorf("Logit[1] incorreto: esperado %f, obteve %f", expected1, result[1])
	}

	// Token 2: aparece 1 vez
	// penaltyAmount = 1 * (1.2 - 1.0) * 10.0 = 2.0
	expected2 := logits[2] - 2.0
	if math.Abs(result[2]-expected2) > 1e-10 {
		t.Errorf("Logit[2] incorreto: esperado %f, obteve %f", expected2, result[2])
	}

	// Tokens 3 e 4: não aparecem, não devem mudar
	for i := 3; i <= 4; i++ {
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] não deveria mudar: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_StrongPenalty testa com penalidade forte
func TestApplyRepetitionPenalty_StrongPenalty(t *testing.T) {
	logits := []float64{5.0, 5.0, 5.0, 5.0, 5.0}
	generatedTokens := []int{0, 0, 0, 0} // Token 0 aparece 4 vezes
	penalty := 2.0

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// penaltyAmount = 4 * (2.0 - 1.0) * 10.0 = 40.0
	expectedPenalty := 40.0
	expected := logits[0] - expectedPenalty

	if math.Abs(result[0]-expected) > 1e-10 {
		t.Errorf("Logit[0] com penalidade forte incorreto: esperado %f, obteve %f", expected, result[0])
	}

	// O logit penalizado deve ser muito menor que os outros
	if result[0] >= result[1] {
		t.Errorf("Logit penalizado deveria ser menor: result[0]=%f, result[1]=%f", result[0], result[1])
	}
}

// TestApplyRepetitionPenalty_NegativeLogits testa com logits negativos
func TestApplyRepetitionPenalty_NegativeLogits(t *testing.T) {
	logits := []float64{-1.0, -2.0, -3.0, -4.0, -5.0}
	generatedTokens := []int{2, 2}
	penalty := 1.5

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// penaltyAmount = 2 * (1.5 - 1.0) * 10.0 = 10.0
	// Para logits negativos: subtrair torna mais negativo
	expectedPenalty := 10.0
	expected := logits[2] - expectedPenalty

	if math.Abs(result[2]-expected) > 1e-10 {
		t.Errorf("Logit[2] negativo incorreto: esperado %f, obteve %f", expected, result[2])
	}

	// O logit deve ficar mais negativo
	if result[2] >= logits[2] {
		t.Errorf("Logit negativo deveria ficar mais negativo: resultado=%f, original=%f", result[2], logits[2])
	}
}

// TestApplyRepetitionPenalty_PenaltyLessThanOne testa com penalty < 1.0 (deve ignorar)
func TestApplyRepetitionPenalty_PenaltyLessThanOne(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{0, 0, 0}
	penalty := 0.8

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Com penalty < 1.0, a função deve retornar logits originais
	for i := range logits {
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] mudou com penalty<1: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_LargeVocab testa com vocabulário grande
func TestApplyRepetitionPenalty_LargeVocab(t *testing.T) {
	vocabSize := 1000
	logits := make([]float64, vocabSize)
	for i := range logits {
		logits[i] = float64(i) * 0.1
	}

	generatedTokens := []int{10, 20, 30, 10, 20} // Tokens 10 e 20 aparecem 2x, token 30 aparece 1x
	penalty := 1.3

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Verificar tokens penalizados
	// Token 10: 2 * (1.3 - 1.0) * 10.0 = 6.0
	if math.Abs(result[10]-(logits[10]-6.0)) > 1e-10 {
		t.Errorf("Token 10 incorreto: esperado %f, obteve %f", logits[10]-6.0, result[10])
	}

	// Token 20: 2 * (1.3 - 1.0) * 10.0 = 6.0
	if math.Abs(result[20]-(logits[20]-6.0)) > 1e-10 {
		t.Errorf("Token 20 incorreto: esperado %f, obteve %f", logits[20]-6.0, result[20])
	}

	// Token 30: 1 * (1.3 - 1.0) * 10.0 = 3.0
	if math.Abs(result[30]-(logits[30]-3.0)) > 1e-10 {
		t.Errorf("Token 30 incorreto: esperado %f, obteve %f", logits[30]-3.0, result[30])
	}

	// Token não gerado não deve mudar
	if math.Abs(result[500]-logits[500]) > 1e-10 {
		t.Errorf("Token 500 não deveria mudar: esperado %f, obteve %f", logits[500], result[500])
	}
}

// TestApplyRepetitionPenalty_DoesNotModifyOriginal testa que a função não modifica o slice original
func TestApplyRepetitionPenalty_DoesNotModifyOriginal(t *testing.T) {
	original := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	logits := make([]float64, len(original))
	copy(logits, original)

	generatedTokens := []int{0, 1, 2}
	penalty := 1.5

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// O slice original não deve ter sido modificado
	for i := range original {
		if math.Abs(original[i]-float64(i+1.0)) > 1e-10 {
			t.Errorf("Original[%d] foi modificado: esperado %f, obteve %f", i, float64(i+1.0), original[i])
		}
	}

	// O slice logits também não deve ter sido modificado (a função deve criar um novo slice)
	for i := range logits {
		if math.Abs(logits[i]-float64(i+1.0)) > 1e-10 {
			t.Errorf("Logits[%d] foi modificado: esperado %f, obteve %f", i, float64(i+1.0), logits[i])
		}
	}

	// O resultado deve ser diferente do original para tokens penalizados
	for i := 0; i < 3; i++ {
		if math.Abs(result[i]-original[i]) < 1e-10 {
			t.Errorf("Result[%d] deveria ser diferente do original após penalidade", i)
		}
	}
}

// TestApplyRepetitionPenalty_OutOfBoundsToken testa com token ID fora dos limites
func TestApplyRepetitionPenalty_OutOfBoundsToken(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	generatedTokens := []int{0, 10, -1, 100} // Tokens 10 e -1 e 100 estão fora dos limites
	penalty := 1.5

	// Não deve panic
	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Apenas token 0 deve ser penalizado
	// penaltyAmount = 1 * (1.5 - 1.0) * 10.0 = 5.0
	expected0 := logits[0] - 5.0
	if math.Abs(result[0]-expected0) > 1e-10 {
		t.Errorf("Logit[0] incorreto: esperado %f, obteve %f", expected0, result[0])
	}

	// Outros tokens válidos não devem mudar
	for i := 1; i < len(logits); i++ {
		if math.Abs(result[i]-logits[i]) > 1e-10 {
			t.Errorf("Logit[%d] não deveria mudar: esperado %f, obteve %f", i, logits[i], result[i])
		}
	}
}

// TestApplyRepetitionPenalty_AllTokensRepeated testa quando todos os tokens foram gerados
func TestApplyRepetitionPenalty_AllTokensRepeated(t *testing.T) {
	logits := []float64{5.0, 4.0, 3.0, 2.0, 1.0}
	generatedTokens := []int{0, 1, 2, 3, 4, 0, 1, 2, 3, 4} // Todos aparecem 2 vezes
	penalty := 1.2

	result := ApplyRepetitionPenalty(logits, generatedTokens, penalty)

	// Todos os tokens devem ter penalidade de: 2 * (1.2 - 1.0) * 10.0 = 4.0
	expectedPenalty := 4.0
	for i := range logits {
		expected := logits[i] - expectedPenalty
		if math.Abs(result[i]-expected) > 1e-10 {
			t.Errorf("Logit[%d] incorreto: esperado %f, obteve %f", i, expected, result[i])
		}
	}

	// A ordem relativa deve ser mantida (todos penalizados igualmente)
	// logits[0] ainda deve ser o maior
	for i := 1; i < len(result); i++ {
		if result[0] <= result[i] {
			t.Errorf("Ordem relativa não mantida: result[0]=%f deveria ser > result[%d]=%f", result[0], i, result[i])
		}
	}
}

// BenchmarkApplyRepetitionPenalty mede a performance da função
func BenchmarkApplyRepetitionPenalty(b *testing.B) {
	vocabSize := 8000 // Tamanho típico do vocabulário
	logits := make([]float64, vocabSize)
	for i := range logits {
		logits[i] = float64(i) * 0.01
	}

	// Simular 100 tokens gerados
	generatedTokens := make([]int, 100)
	for i := range generatedTokens {
		generatedTokens[i] = i % 50 // Alguns tokens repetidos
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyRepetitionPenalty(logits, generatedTokens, 1.3)
	}
}

// BenchmarkApplyRepetitionPenalty_SmallVocab benchmark com vocabulário pequeno
func BenchmarkApplyRepetitionPenalty_SmallVocab(b *testing.B) {
	logits := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}
	generatedTokens := []int{0, 1, 2, 0, 1}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyRepetitionPenalty(logits, generatedTokens, 1.3)
	}
}

// BenchmarkApplyRepetitionPenalty_LargeGenerated benchmark com muitos tokens gerados
func BenchmarkApplyRepetitionPenalty_LargeGenerated(b *testing.B) {
	vocabSize := 8000
	logits := make([]float64, vocabSize)
	for i := range logits {
		logits[i] = 1.0
	}

	// 500 tokens gerados
	generatedTokens := make([]int, 500)
	for i := range generatedTokens {
		generatedTokens[i] = i % 200
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ApplyRepetitionPenalty(logits, generatedTokens, 1.3)
	}
}
