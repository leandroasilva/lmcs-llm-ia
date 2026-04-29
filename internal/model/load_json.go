package model

import (
	"encoding/json"
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
)

// loadTrainedModelFromJSON implementa o carregamento de modelo JSON
func loadTrainedModelFromJSON(path string) (*TransformerModel, error) {
	// Ler arquivo JSON
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("erro ao ler arquivo JSON: %w", err)
	}

	// Parse JSON structure
	var trainedModel struct {
		Metadata map[string]interface{} `json:"metadata"`
		Config   map[string]interface{} `json:"config"`
		Weights  map[string]interface{} `json:"weights"`
	}

	if err := json.Unmarshal(data, &trainedModel); err != nil {
		return nil, fmt.Errorf("erro ao parsear JSON: %w", err)
	}

	// Extrair configuração
	config := trainedModel.Config
	vocabSize := int(config["vocab_size"].(float64))
	dModel := int(config["d_model"].(float64))
	nHeads := int(config["n_heads"].(float64))
	nLayers := int(config["n_layers"].(float64))
	maxSeqLen := int(config["max_seq_len"].(float64))
	ffHidden := int(config["ff_hidden"].(float64))
	dropout := 0.1
	if d, ok := config["dropout"].(float64); ok {
		dropout = d
	}

	// Criar modelo
	model := NewTransformerModel(vocabSize, dModel, nHeads, nLayers, maxSeqLen, ffHidden, 0.001, dropout, 0.01)

	// Carregar pesos
	if err := loadJSONWeights(model, trainedModel.Weights, vocabSize, dModel, nLayers, maxSeqLen, ffHidden); err != nil {
		return nil, fmt.Errorf("erro ao carregar pesos: %w", err)
	}

	// Carregar vocabulário do arquivo
	if err := loadVocabularyForModel(model); err != nil {
		// Se falhar, usar vocabulário placeholder
		fmt.Printf("Aviso: Não foi possível carregar vocabulário: %v\n", err)
		populatePlaceholderVocab(model, vocabSize)
	}

	return model, nil
}

// loadJSONWeights carrega pesos do JSON para o modelo
func loadJSONWeights(m *TransformerModel, weights map[string]interface{}, vocabSize, dModel, nLayers, maxSeqLen, ffHidden int) error {
	// Token embedding
	if err := loadJSONMatrix2D(m.TokenEmbedding, weights, "token_embedding.weight", vocabSize, dModel); err != nil {
		return err
	}

	// Position embedding
	if err := loadJSONMatrix2D(m.PositionEmbedding, weights, "position_embedding.weight", maxSeqLen, dModel); err != nil {
		return err
	}

	// Transformer layers
	for i := 0; i < nLayers; i++ {
		layer := &m.TransformerLayers[i]
		prefix := fmt.Sprintf("transformer.layers.%d.", i)

		// PyTorch in_proj_weight: [3*d_model, d_model] - Q, K, V concatenados
		inProjWeight := getJSONMatrix2D(weights, prefix+"self_attn.in_proj_weight", dModel*3, dModel)
		inProjBias := getJSONVector1D(weights, prefix+"self_attn.in_proj_bias", dModel*3)

		// Separar Q, K, V
		extractJSONSubMatrix(layer.WQ, inProjWeight, inProjBias, 0, dModel)
		extractJSONSubMatrix(layer.WK, inProjWeight, inProjBias, 1, dModel)
		extractJSONSubMatrix(layer.WV, inProjWeight, inProjBias, 2, dModel)

		// Output projection
		if err := loadJSONMatrix2D(layer.WO, weights, prefix+"self_attn.out_proj.weight", dModel, dModel); err != nil {
			return err
		}

		// FFN
		if err := loadJSONMatrix2D(layer.W1, weights, prefix+"linear1.weight", ffHidden, dModel); err != nil {
			return err
		}
		if err := loadJSONMatrix1D(layer.B1, weights, prefix+"linear1.bias", ffHidden); err != nil {
			return err
		}
		if err := loadJSONMatrix2D(layer.W2, weights, prefix+"linear2.weight", dModel, ffHidden); err != nil {
			return err
		}
		if err := loadJSONMatrix1D(layer.B2, weights, prefix+"linear2.bias", dModel); err != nil {
			return err
		}

		// Layer norms
		if err := loadJSONVector(layer.LN1Weight, weights, prefix+"norm1.weight", dModel); err != nil {
			return err
		}
		if err := loadJSONVector(layer.LN1Bias, weights, prefix+"norm1.bias", dModel); err != nil {
			return err
		}
		if err := loadJSONVector(layer.LN2Weight, weights, prefix+"norm2.weight", dModel); err != nil {
			return err
		}
		if err := loadJSONVector(layer.LN2Bias, weights, prefix+"norm2.bias", dModel); err != nil {
			return err
		}
	}

	// Output projection - usar token_embedding transpose (tied weights)
	m.WOut.CloneFrom(m.TokenEmbedding)
	m.BOut.Zero()

	return nil
}

// Helper functions

func loadJSONMatrix2D(target *mat.Dense, weights map[string]interface{}, key string, rows, cols int) error {
	data := getJSONMatrix2D(weights, key, rows, cols)
	target.CloneFrom(data)
	return nil
}

func loadJSONMatrix1D(target *mat.Dense, weights map[string]interface{}, key string, size int) error {
	data := getJSONVector1D(weights, key, size)
	for i := 0; i < size; i++ {
		target.Set(i, 0, data[i])
	}
	return nil
}

func loadJSONVector(target *mat.Dense, weights map[string]interface{}, key string, size int) error {
	data := getJSONVector1D(weights, key, size)
	for i := 0; i < size; i++ {
		target.Set(i, 0, data[i])
	}
	return nil
}

func getJSONMatrix2D(weights map[string]interface{}, key string, rows, cols int) *mat.Dense {
	data, ok := weights[key]
	if !ok {
		return mat.NewDense(rows, cols, make([]float64, rows*cols))
	}

	matrix, ok := data.([]interface{})
	if !ok {
		return mat.NewDense(rows, cols, make([]float64, rows*cols))
	}

	flatData := make([]float64, rows*cols)
	for i := 0; i < rows && i < len(matrix); i++ {
		rowData, ok := matrix[i].([]interface{})
		if !ok {
			continue
		}
		for j := 0; j < cols && j < len(rowData); j++ {
			if val, ok := rowData[j].(float64); ok {
				flatData[i*cols+j] = val
			}
		}
	}

	return mat.NewDense(rows, cols, flatData)
}

func getJSONVector1D(weights map[string]interface{}, key string, size int) []float64 {
	data, ok := weights[key]
	if !ok {
		return make([]float64, size)
	}

	vector, ok := data.([]interface{})
	if !ok {
		return make([]float64, size)
	}

	result := make([]float64, size)
	for i := 0; i < size && i < len(vector); i++ {
		if val, ok := vector[i].(float64); ok {
			result[i] = val
		}
	}

	return result
}

func extractJSONSubMatrix(target *mat.Dense, source *mat.Dense, bias []float64, blockIndex, blockSize int) {
	rows, cols := source.Dims()
	if rows < blockSize {
		return
	}

	startRow := blockIndex * blockSize
	for i := 0; i < blockSize && i < rows-startRow; i++ {
		for j := 0; j < cols; j++ {
			val := source.At(startRow+i, j)
			if bias != nil && len(bias) > startRow+i {
				val += bias[startRow+i]
			}
			target.Set(i, j, val)
		}
	}
}
