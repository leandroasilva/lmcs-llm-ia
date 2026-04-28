package model

import (
	"fmt"
	"math"
	"sort"

	"gonum.org/v1/gonum/mat"
)

// MoEConfig configurações para Mixture of Experts
type MoEConfig struct {
	DModel     int     // Dimensão do modelo
	NumExperts int     // Número total de experts
	TopK       int     // Número de experts ativos por token (k)
	FFHidden   int     // Hidden size de cada expert FFN
	UseLoadBal bool    // Usar load balancing loss
	LambdaAux  float64 // Peso do auxiliary loss (load balancing)
}

// MoELayer representa uma camada Mixture of Experts
type MoELayer struct {
	// Gating network (router)
	WGate *mat.Dense // [num_experts, d_model] - Gate weights

	// Experts (FFNs especializados)
	Experts []ExpertFFN

	// Configurações
	Config *MoEConfig

	// Auxiliares para load balancing
	ExpertUsage []float64 // Contagem de uso por expert (para monitoramento)

	// Gradientes
	GradWGate *mat.Dense
}

// ExpertFFN representa um Feed-Forward Network especialista
type ExpertFFN struct {
	W1 *mat.Dense // [ff_hidden, d_model]
	B1 *mat.Dense // [ff_hidden, 1]
	W2 *mat.Dense // [d_model, ff_hidden]
	B2 *mat.Dense // [d_model, 1]

	// Gradientes
	GradW1, GradB1 *mat.Dense
	GradW2, GradB2 *mat.Dense
}

// RoutingResult resultado do routing para um batch de tokens
type RoutingResult struct {
	// Gates: [batch_size, num_experts] - weights para cada expert
	Gates *mat.Dense

	// SelectedExperts: [batch_size, top_k] - indices dos experts selecionados
	SelectedExperts [][]int

	// GateValues: [batch_size, top_k] - valores dos gates para experts selecionados
	GateValues [][]float64

	// ExpertWeights: [num_experts] - peso total atribuído a cada expert (para load balancing)
	ExpertWeights []float64
}

// NewMoEConfig cria configuração padrão para MoE
func NewMoEConfig(dModel, numExperts, topK, ffHidden int) *MoEConfig {
	return &MoEConfig{
		DModel:     dModel,
		NumExperts: numExperts,
		TopK:       topK,
		FFHidden:   ffHidden,
		UseLoadBal: true,
		LambdaAux:  0.01, // Valor típico usado em modelos como DeepSeek
	}
}

// NewMoELayer cria uma nova camada MoE
func NewMoELayer(cfg *MoEConfig) *MoELayer {
	layer := &MoELayer{
		Config:      cfg,
		Experts:     make([]ExpertFFN, cfg.NumExperts),
		ExpertUsage: make([]float64, cfg.NumExperts),
	}

	// Inicializar gating network
	scaleGate := math.Sqrt(2.0 / float64(cfg.DModel+cfg.NumExperts))
	layer.WGate = transformerRandomMatrix(cfg.NumExperts, cfg.DModel, scaleGate)

	// Inicializar cada expert FFN
	for i := 0; i < cfg.NumExperts; i++ {
		layer.Experts[i] = newExpertFFN(cfg.DModel, cfg.FFHidden)
	}

	return layer
}

// newExpertFFN cria um novo expert FFN
func newExpertFFN(dModel, ffHidden int) ExpertFFN {
	expert := ExpertFFN{}

	// FFN weights com Xavier initialization
	// W1: [d_model, ff_hidden] para x @ W1
	scale1 := math.Sqrt(2.0 / float64(dModel+ffHidden))
	expert.W1 = transformerRandomMatrix(dModel, ffHidden, scale1)
	expert.B1 = mat.NewDense(ffHidden, 1, make([]float64, ffHidden))

	// W2: [ff_hidden, d_model] para hidden @ W2
	scale2 := math.Sqrt(2.0 / float64(ffHidden+dModel))
	expert.W2 = transformerRandomMatrix(ffHidden, dModel, scale2)
	expert.B2 = mat.NewDense(dModel, 1, make([]float64, dModel))

	return expert
}

// MixtureOfExperts aplica MoE forward pass com top-k routing
// X: [seq_len, d_model]
// Returns: [seq_len, d_model], RoutingResult
func MixtureOfExperts(layer *MoELayer, X *mat.Dense, seqLen int) (*mat.Dense, *RoutingResult) {
	// 1. Routing: calcular gates para cada token
	routingResult := computeGates(layer, X, seqLen)

	// 2. Aplicar experts selecionados e combinar resultados
	output := mat.NewDense(seqLen, layer.Config.DModel, nil)

	for i := 0; i < seqLen; i++ {
		// Extrair vetor do token i
		xRow := mat.NewDense(1, layer.Config.DModel, nil)
		for j := 0; j < layer.Config.DModel; j++ {
			xRow.Set(0, j, X.At(i, j))
		}

		// Combinar outputs dos top-k experts
		combinedOutput := mat.NewDense(1, layer.Config.DModel, nil)
		selectedExperts := routingResult.SelectedExperts[i]
		gateValues := routingResult.GateValues[i]

		for k := 0; k < len(selectedExperts); k++ {
			expertIdx := selectedExperts[k]
			gateValue := gateValues[k]

			// Aplicar expert
			expertOutput := applyExpert(&layer.Experts[expertIdx], xRow)

			// Weighted sum: gate_value * expert_output
			expertOutput.Scale(gateValue, expertOutput)
			combinedOutput.Add(combinedOutput, expertOutput)

			// Monitorar uso do expert
			layer.ExpertUsage[expertIdx]++
		}

		// Copiar para output
		for j := 0; j < layer.Config.DModel; j++ {
			output.Set(i, j, combinedOutput.At(0, j))
		}
	}

	return output, routingResult
}

// computeGates calcula os gates e faz top-k routing
func computeGates(layer *MoELayer, X *mat.Dense, seqLen int) *RoutingResult {
	result := &RoutingResult{
		SelectedExperts: make([][]int, seqLen),
		GateValues:      make([][]float64, seqLen),
		ExpertWeights:   make([]float64, layer.Config.NumExperts),
	}

	// Calcular logits: X @ WGate^T
	// [seq_len, d_model] @ [d_model, num_experts] = [seq_len, num_experts]
	gateLogits := mat.NewDense(seqLen, layer.Config.NumExperts, nil)
	gateLogits.Mul(X, layer.WGate.T())

	// Aplicar softmax por token e selecionar top-k
	result.Gates = mat.NewDense(seqLen, layer.Config.NumExperts, nil)

	for i := 0; i < seqLen; i++ {
		// Extrair logits do token i
		logits := make([]float64, layer.Config.NumExperts)
		for j := 0; j < layer.Config.NumExperts; j++ {
			logits[j] = gateLogits.At(i, j)
		}

		// Softmax
		softmaxLogits := softmax(logits)

		// Top-k selection
		topKIndices, topKValues := topK(softmaxLogits, layer.Config.TopK)

		result.SelectedExperts[i] = topKIndices
		result.GateValues[i] = topKValues

		// Normalizar gate values (para que somem 1)
		sum := 0.0
		for _, v := range topKValues {
			sum += v
		}
		if sum > 0 {
			for k := range topKValues {
				topKValues[k] /= sum
			}
		}

		// Preencher gates
		for j := 0; j < layer.Config.NumExperts; j++ {
			result.Gates.Set(i, j, softmaxLogits[j])
		}

		// Acumular expert weights para load balancing
		for k := 0; k < len(topKIndices); k++ {
			result.ExpertWeights[topKIndices[k]] += topKValues[k]
		}
	}

	return result
}

// applyExpert aplica um expert FFN a um token
func applyExpert(expert *ExpertFFN, x *mat.Dense) *mat.Dense {
	// FFN: x @ W1 + b1 -> ReLU -> @ W2 + b2

	// Hidden: x @ W1 + b1
	// [1, d_model] @ [d_model, ff_hidden] = [1, ff_hidden]
	xRows, xCols := x.Dims()
	w1Rows, w1Cols := expert.W1.Dims()

	// Verificar dimensões
	if xCols != w1Rows {
		panic(fmt.Sprintf("Dimension mismatch: x is %dx%d, W1 is %dx%d", xRows, xCols, w1Rows, w1Cols))
	}

	hidden := mat.NewDense(1, w1Cols, nil)
	hidden.Mul(x, expert.W1)

	// Adicionar bias [ff_hidden, 1] -> extrair valores
	for i := 0; i < w1Cols; i++ {
		val := hidden.At(0, i) + expert.B1.At(i, 0)
		hidden.Set(0, i, val)
	}

	// ReLU activation
	reluInPlace(hidden)

	// Output: hidden @ W2 + b2
	// [1, ff_hidden] @ [ff_hidden, d_model] = [1, d_model]
	hiddenRows, hiddenCols := hidden.Dims()
	w2Rows, w2Cols := expert.W2.Dims()

	// Verificar dimensões
	if hiddenCols != w2Rows {
		panic(fmt.Sprintf("Dimension mismatch: hidden is %dx%d, W2 is %dx%d", hiddenRows, hiddenCols, w2Rows, w2Cols))
	}

	output := mat.NewDense(1, w2Cols, nil)
	output.Mul(hidden, expert.W2)

	// Adicionar bias
	for i := 0; i < w2Cols; i++ {
		val := output.At(0, i) + expert.B2.At(i, 0)
		output.Set(0, i, val)
	}

	return output
}

// topK retorna os índices e valores dos k maiores elementos
func topK(values []float64, k int) ([]int, []float64) {
	// Criar pares (index, value)
	type pair struct {
		index int
		value float64
	}

	pairs := make([]pair, len(values))
	for i, v := range values {
		pairs[i] = pair{i, v}
	}

	// Ordenar por valor (decrescente)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})

	// Pegar top-k
	indices := make([]int, k)
	topValues := make([]float64, k)
	for i := 0; i < k && i < len(pairs); i++ {
		indices[i] = pairs[i].index
		topValues[i] = pairs[i].value
	}

	return indices, topValues
}

// softmax aplica softmax a um vetor
func softmax(values []float64) []float64 {
	// Subtrair max para estabilidade numérica
	maxVal := values[0]
	for _, v := range values {
		if v > maxVal {
			maxVal = v
		}
	}

	// Calcular exp e sum
	exp := make([]float64, len(values))
	sum := 0.0
	for i, v := range values {
		exp[i] = math.Exp(v - maxVal)
		sum += exp[i]
	}

	// Normalizar
	result := make([]float64, len(values))
	for i := range exp {
		result[i] = exp[i] / sum
	}

	return result
}

// ComputeLoadBalancingLoss calcula auxiliary loss para balancear carga entre experts
// Inspirado no load balancing loss do Switch Transformer / DeepSeek
func ComputeLoadBalancingLoss(gates *mat.Dense, seqLen, numExperts int) float64 {
	if !gates.IsEmpty() {
		// Calcular fração de tokens atribuídos a cada expert
		expertFractions := make([]float64, numExperts)
		gateMeans := make([]float64, numExperts)

		for i := 0; i < numExperts; i++ {
			count := 0
			sum := 0.0
			for j := 0; j < seqLen; j++ {
				val := gates.At(j, i)
				if val > 0 {
					count++
				}
				sum += val
			}
			expertFractions[i] = float64(count) / float64(seqLen)
			gateMeans[i] = sum / float64(seqLen)
		}

		// Load balancing loss: num_experts * sum(fraction_i * mean_gate_i)
		loss := 0.0
		for i := 0; i < numExperts; i++ {
			loss += expertFractions[i] * gateMeans[i]
		}
		loss *= float64(numExperts)

		return loss
	}
	return 0.0
}

// GetExpertUsage retorna estatísticas de uso dos experts
func GetExpertUsage(layer *MoELayer) map[string]interface{} {
	total := 0.0
	for _, usage := range layer.ExpertUsage {
		total += usage
	}

	if total == 0 {
		return map[string]interface{}{
			"expert_usage":   layer.ExpertUsage,
			"total_tokens":   0,
			"sparsity_ratio": 0.0,
		}
	}

	// Calcular estatísticas
	usagePercents := make([]float64, len(layer.ExpertUsage))
	for i, usage := range layer.ExpertUsage {
		usagePercents[i] = (usage / total) * 100
	}

	// Sparsity: fração de experts não utilizados
	activeExperts := 0
	for _, usage := range layer.ExpertUsage {
		if usage > 0 {
			activeExperts++
		}
	}
	sparsityRatio := float64(len(layer.ExpertUsage)-activeExperts) / float64(len(layer.ExpertUsage)) * 100

	return map[string]interface{}{
		"expert_usage":      layer.ExpertUsage,
		"expert_usage_pct":  usagePercents,
		"total_tokens":      total,
		"active_experts":    activeExperts,
		"total_experts":     len(layer.ExpertUsage),
		"sparsity_ratio":    sparsityRatio,
		"parameters_active": float64(activeExperts) / float64(len(layer.ExpertUsage)) * 100,
	}
}

// GetMoEParameterCount calcula contagem de parâmetros do MoE
func GetMoEParameterCount(cfg *MoEConfig) map[string]interface{} {
	// Gating network: num_experts * d_model
	gateParams := cfg.NumExperts * cfg.DModel

	// Cada expert: (ff_hidden * d_model) + (d_model * ff_hidden) + biases
	expertParams := cfg.FFHidden*cfg.DModel + cfg.DModel*cfg.FFHidden + cfg.FFHidden + cfg.DModel

	// Total de parâmetros MoE
	totalParams := gateParams + (cfg.NumExperts * expertParams)

	// Parâmetros ativos por token (top-k experts)
	activeParams := gateParams + (cfg.TopK * expertParams)

	// Sparsity de parâmetros
	paramSparsity := float64(activeParams) / float64(totalParams) * 100

	return map[string]interface{}{
		"gate_params":             gateParams,
		"params_per_expert":       expertParams,
		"total_params":            totalParams,
		"active_params_per_token": activeParams,
		"parameter_sparsity_pct":  paramSparsity,
		"total_experts":           cfg.NumExperts,
		"active_experts":          cfg.TopK,
	}
}

// ResetExpertUsage reseta contadores de uso dos experts
func ResetExpertUsage(layer *MoELayer) {
	for i := range layer.ExpertUsage {
		layer.ExpertUsage[i] = 0
	}
}

// reluInPlace aplica ReLU in-place em uma matriz
func reluInPlace(m *mat.Dense) {
	rows, cols := m.Dims()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val := m.At(i, j)
			if val < 0 {
				m.Set(i, j, 0)
			}
		}
	}
}
