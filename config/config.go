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
			Epochs:       50,
			LearningRate: 0.01,
			BatchSize:    32,
			Temperature:  0.8,
		},
		Server: ServerConfig{
			Port: ":8080",
			Host: "localhost",
		},
		Paths: PathsConfig{
			ModelPath: "modelo_treinado.bin",
			InputFile: "livro.txt",
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
	if c.Server.Port == "" {
		return fmt.Errorf("port não pode estar vazio")
	}

	return nil
}
