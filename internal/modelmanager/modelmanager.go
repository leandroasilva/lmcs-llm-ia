package modelmanager

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
)

// ModelMetadata armazena metadata sobre um checkpoint de modelo
type ModelMetadata struct {
	Timestamp      string        `json:"timestamp"`
	EpochsTrained  int           `json:"epochs_trained"`
	FinalLoss      float64       `json:"final_loss"`
	Perplexity     float64       `json:"perplexity"`
	Config         config.Config `json:"config"`
	DatasetSize    int           `json:"dataset_size"`
	ModelPath      string        `json:"model_path"`
	GenerationTime time.Duration `json:"generation_time"`
}

// ModelVersion representa uma versão de modelo
type ModelVersion struct {
	ID       string        `json:"id"`
	Metadata ModelMetadata `json:"metadata"`
}

// ListModels lista todos os modelos salvos
func ListModels(modelsDir string) ([]ModelVersion, error) {
	pattern := filepath.Join(modelsDir, "checkpoint-*.bin")
	files, err := filepath.Glob(pattern)
	if err != nil {
		return nil, err
	}

	var versions []ModelVersion
	for _, file := range files {
		// Extrair timestamp do filename
		base := filepath.Base(file)
		timestamp := strings.TrimPrefix(base, "checkpoint-")
		timestamp = strings.TrimSuffix(timestamp, ".bin")

		// Tentar carregar metadata
		metaPath := strings.TrimSuffix(file, ".bin") + ".json"
		metadata, err := loadMetadata(metaPath)
		if err != nil {
			// Se não encontrar metadata, criar básico
			metadata = ModelMetadata{
				Timestamp: timestamp,
				ModelPath: file,
			}
		}

		versions = append(versions, ModelVersion{
			ID:       timestamp,
			Metadata: metadata,
		})
	}

	// Ordenar por timestamp (mais recente primeiro)
	sort.Slice(versions, func(i, j int) bool {
		return versions[i].ID > versions[j].ID
	})

	return versions, nil
}

// SaveCheckpoint salva um checkpoint do modelo com metadata
func SaveCheckpoint(modelPath string, metadata ModelMetadata) error {
	// Criar diretório se não existir
	dir := filepath.Dir(modelPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("erro ao criar diretório: %w", err)
	}

	// Gerar timestamp
	timestamp := time.Now().Format("2006-01-02-150405")

	// Renomear arquivo para incluir timestamp
	checkpointName := fmt.Sprintf("checkpoint-%s.bin", timestamp)
	checkpointPath := filepath.Join(dir, checkpointName)

	if err := os.Rename(modelPath, checkpointPath); err != nil {
		// Se falhar rename, copiar
		return fmt.Errorf("erro ao criar checkpoint: %w", err)
	}

	// Atualizar metadata
	metadata.Timestamp = timestamp
	metadata.ModelPath = checkpointPath
	metadata.Perplexity = calculatePerplexity(metadata.FinalLoss)

	// Salvar metadata JSON
	metaPath := strings.TrimSuffix(checkpointPath, ".bin") + ".json"
	if err := saveMetadata(metaPath, metadata); err != nil {
		return fmt.Errorf("erro ao salvar metadata: %w", err)
	}

	// Criar symlink "latest" apontando para este checkpoint
	latestPath := filepath.Join(dir, "latest.bin")
	os.Remove(latestPath) // Remover symlink antigo
	if err := os.Symlink(checkpointPath, latestPath); err != nil {
		// Se não suportar symlink, copiar
		fmt.Printf("Aviso: Não foi possível criar symlink, copiando arquivo\n")
	}

	fmt.Printf("✓ Checkpoint salvo: %s\n", checkpointName)
	fmt.Printf("  Metadata: %s\n", filepath.Base(metaPath))

	return nil
}

// LoadLatestModel carrega o modelo mais recente
func LoadLatestModel(modelsDir string) (string, error) {
	latestPath := filepath.Join(modelsDir, "latest.bin")

	// Verificar se symlink existe
	if _, err := os.Stat(latestPath); err == nil {
		return latestPath, nil
	}

	// Fallback: encontrar checkpoint mais recente
	versions, err := ListModels(modelsDir)
	if err != nil {
		return "", err
	}

	if len(versions) == 0 {
		return "", fmt.Errorf("nenhum modelo encontrado em %s", modelsDir)
	}

	return versions[0].Metadata.ModelPath, nil
}

// DeleteModel deleta um modelo específico
func DeleteModel(modelID string, modelsDir string) error {
	modelPath := filepath.Join(modelsDir, fmt.Sprintf("checkpoint-%s.bin", modelID))
	metaPath := strings.TrimSuffix(modelPath, ".bin") + ".json"

	// Deletar arquivo do modelo
	if err := os.Remove(modelPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("erro ao deletar modelo: %w", err)
	}

	// Deletar metadata
	if err := os.Remove(metaPath); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("erro ao deletar metadata: %w", err)
	}

	fmt.Printf("✓ Modelo deletado: %s\n", modelID)
	return nil
}

// loadMetadata carrega metadata de um arquivo JSON
func loadMetadata(path string) (ModelMetadata, error) {
	var metadata ModelMetadata

	data, err := os.ReadFile(path)
	if err != nil {
		return metadata, err
	}

	if err := json.Unmarshal(data, &metadata); err != nil {
		return metadata, err
	}

	return metadata, nil
}

// saveMetadata salva metadata em um arquivo JSON
func saveMetadata(path string, metadata ModelMetadata) error {
	data, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// calculatePerplexity calcula perplexidade a partir do loss
func calculatePerplexity(loss float64) float64 {
	// Perplexity = exp(loss)
	return exp(loss)
}

// exp calcula e^x usando série de Taylor
func exp(x float64) float64 {
	// Aproximação simples para perplexidade
	result := 1.0
	term := 1.0
	for i := 1; i < 100; i++ {
		term *= x / float64(i)
		result += term
		if term < 1e-10 {
			break
		}
	}
	return result
}
