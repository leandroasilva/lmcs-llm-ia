package config

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config armazena todas as configurações da aplicação
type Config struct {
	Training TrainingConfig `json:"training"`
	Server   ServerConfig   `json:"server"`
	Paths    PathsConfig    `json:"paths"`
}

// TrainingConfig configurações de treinamento do modelo
type TrainingConfig struct {
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	BatchSize    int     `json:"batch_size"`
	Temperature  float64 `json:"temperature"`
	TopK         int     `json:"top_k"` // Top-K sampling
	// Transformer
	DModel    int `json:"d_model"`     // Dimension do modelo
	NHeads    int `json:"n_heads"`     // Número de attention heads
	NumLayers int `json:"num_layers"`  // Número de transformer layers
	MaxSeqLen int `json:"max_seq_len"` // Tamanho máximo da sequência
	FFHidden  int `json:"ff_hidden"`   // Hidden size do feed-forward
	MaxVocab  int `json:"max_vocab"`   // Tamanho máximo do vocabulário
}

// ServerConfig configurações do servidor HTTP
type ServerConfig struct {
	Port string `json:"port"`
	Host string `json:"host"`
}

// PathsConfig caminhos de arquivos
type PathsConfig struct {
	ModelPath string `json:"model_path"`
	InputFile string `json:"input_file"`
}

// DefaultConfig retorna configurações padrão
func DefaultConfig() *Config {
	return &Config{
		Training: TrainingConfig{
			Epochs:       30,
			LearningRate: 0.001,
			BatchSize:    16,
			Temperature:  0.7,
			TopK:         30,
			DModel:       128,
			NHeads:       4,
			NumLayers:    2,
			MaxSeqLen:    256,
			FFHidden:     256,
			MaxVocab:     5000,
		},
		Server: ServerConfig{
			Port: ":8080",
			Host: "localhost",
		},
		Paths: PathsConfig{
			ModelPath: "lmcs-model.bin",
			InputFile: "dataset/data/train_enriched.txt",
		},
	}
}

// LoadFromFile carrega configurações de um arquivo JSON
func LoadFromFile(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("erro ao ler arquivo de config: %w", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("erro ao parsear JSON: %w", err)
	}

	return &cfg, nil
}

// SaveToFile salva configurações em um arquivo JSON
func (c *Config) SaveToFile(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("erro ao serializar config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("erro ao escrever arquivo: %w", err)
	}

	return nil
}

// Validate valida as configurações
func (c *Config) Validate() error {
	if c.Training.Epochs <= 0 {
		return fmt.Errorf("epochs deve ser maior que 0")
	}
	if c.Training.LearningRate <= 0 || c.Training.LearningRate > 1.0 {
		return fmt.Errorf("learning_rate deve estar entre 0 e 1")
	}
	if c.Training.BatchSize <= 0 {
		return fmt.Errorf("batch_size deve ser maior que 0")
	}
	if c.Training.Temperature <= 0 || c.Training.Temperature > 2.0 {
		return fmt.Errorf("temperature deve estar entre 0 e 2")
	}
	if c.Training.TopK <= 0 || c.Training.TopK > 100 {
		return fmt.Errorf("top_k deve estar entre 1 e 100")
	}
	// Transformer validations
	if c.Training.DModel <= 0 {
		return fmt.Errorf("d_model deve ser maior que 0")
	}
	if c.Training.NHeads <= 0 {
		return fmt.Errorf("n_heads deve ser maior que 0")
	}
	if c.Training.NumLayers <= 0 || c.Training.NumLayers > 12 {
		return fmt.Errorf("num_layers deve estar entre 1 e 12")
	}
	if c.Training.MaxSeqLen <= 0 || c.Training.MaxSeqLen > 1024 {
		return fmt.Errorf("max_seq_len deve estar entre 1 e 1024")
	}
	if c.Training.FFHidden <= 0 {
		return fmt.Errorf("ff_hidden deve ser maior que 0")
	}
	if c.Training.MaxVocab <= 0 {
		return fmt.Errorf("max_vocab deve ser maior que 0")
	}
	if c.Server.Port == "" {
		return fmt.Errorf("port não pode estar vazio")
	}

	return nil
}
