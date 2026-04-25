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
	ContextSize  int     `json:"context_size"` // Tamanho do contexto (n-gramas)
	TopK         int     `json:"top_k"`        // Top-K sampling
	HiddenSize   int     `json:"hidden_size"`  // Tamanho da camada oculta (LSTM)
	NumLayers    int     `json:"num_layers"`   // Número de camadas LSTM
	UseLSTM      bool    `json:"use_lstm"`     // Usar arquitetura LSTM
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
			Epochs:       100,
			LearningRate: 0.001,
			BatchSize:    32,
			Temperature:  0.7,
			ContextSize:  15,
			TopK:         40,
			HiddenSize:   128,
			NumLayers:    1,
			UseLSTM:      true,
		},
		Server: ServerConfig{
			Port: ":8080",
			Host: "localhost",
		},
		Paths: PathsConfig{
			ModelPath: "lmcs-model.bin",
			InputFile: "dataset/data/train.txt",
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
	if c.Training.ContextSize <= 0 || c.Training.ContextSize > 500 {
		return fmt.Errorf("context_size deve estar entre 1 e 500")
	}
	if c.Training.NumLayers <= 0 || c.Training.NumLayers > 5 {
		return fmt.Errorf("num_layers deve estar entre 1 e 5")
	}
	if c.Training.TopK <= 0 || c.Training.TopK > 100 {
		return fmt.Errorf("top_k deve estar entre 1 e 100")
	}
	if c.Server.Port == "" {
		return fmt.Errorf("port não pode estar vazio")
	}

	return nil
}
