package config

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Environment representa o ambiente de execução
type Environment string

const (
	EnvDevelopment Environment = "development"
	EnvStaging     Environment = "staging"
	EnvProduction  Environment = "production"
	EnvTest        Environment = "test"
)

// Config armazena todas as configurações da aplicação
type Config struct {
	Environment Environment    `json:"environment"`
	Training    TrainingConfig `json:"training"`
	Server      ServerConfig   `json:"server"`
	Paths       PathsConfig    `json:"paths"`
}

// TrainingConfig configurações de treinamento do modelo
type TrainingConfig struct {
	Epochs       int     `json:"epochs"`
	LearningRate float64 `json:"learning_rate"`
	BatchSize    int     `json:"batch_size"`
	Temperature  float64 `json:"temperature"`
	TopK         int     `json:"top_k"` // Top-K sampling
	// Transformer
	DModel      int     `json:"d_model"`      // Dimension do modelo
	NHeads      int     `json:"n_heads"`      // Número de attention heads
	NumLayers   int     `json:"num_layers"`   // Número de transformer layers
	MaxSeqLen   int     `json:"max_seq_len"`  // Tamanho máximo da sequência
	FFHidden    int     `json:"ff_hidden"`    // Hidden size do feed-forward
	MaxVocab    int     `json:"max_vocab"`    // Tamanho máximo do vocabulário
	DropoutRate float64 `json:"dropout_rate"` // Taxa de dropout para regularização
	WeightDecay float64 `json:"weight_decay"` // Weight decay (L2 regularization)
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

// DefaultConfig retorna configurações padrão para desenvolvimento
func DefaultConfig() *Config {
	return &Config{
		Environment: EnvDevelopment,
		Training: TrainingConfig{
			Epochs:       300,
			LearningRate: 0.001,
			BatchSize:    16,
			Temperature:  0.7,
			TopK:         30,
			DModel:       512,
			NHeads:       8,
			NumLayers:    6,
			MaxSeqLen:    256,
			FFHidden:     1024,
			MaxVocab:     5000,
			DropoutRate:  0.1,
			WeightDecay:  0.01,
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

	// Aplicar variáveis de ambiente (sobrescrevem o arquivo)
	cfg.applyEnvironmentVariables()

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

// Validate valida as configurações com verificações robustas
func (c *Config) Validate() error {
	var errors []string

	// Validar ambiente
	if c.Environment == "" {
		c.Environment = EnvDevelopment
	}
	if !isValidEnvironment(c.Environment) {
		errors = append(errors, fmt.Sprintf("environment inválido: %s (deve ser: development, staging, production, test)", c.Environment))
	}

	// Validar training
	if c.Training.Epochs <= 0 {
		errors = append(errors, "epochs deve ser maior que 0")
	}
	if c.Training.Epochs > 10000 {
		errors = append(errors, "epochs não deve exceder 10000")
	}
	if c.Training.LearningRate <= 0 || c.Training.LearningRate > 1.0 {
		errors = append(errors, "learning_rate deve estar entre 0 e 1")
	}
	if c.Training.BatchSize <= 0 {
		errors = append(errors, "batch_size deve ser maior que 0")
	}
	if c.Training.BatchSize > 256 {
		errors = append(errors, "batch_size não deve exceder 256")
	}
	if c.Training.Temperature <= 0 || c.Training.Temperature > 2.0 {
		errors = append(errors, "temperature deve estar entre 0 e 2")
	}
	if c.Training.TopK <= 0 || c.Training.TopK > 100 {
		errors = append(errors, "top_k deve estar entre 1 e 100")
	}

	// Validar arquitetura Transformer
	if c.Training.DModel <= 0 {
		errors = append(errors, "d_model deve ser maior que 0")
	}
	if c.Training.DModel > 2048 {
		errors = append(errors, "d_model não deve exceder 2048")
	}
	if c.Training.DModel%64 != 0 {
		errors = append(errors, "d_model deve ser múltiplo de 64")
	}
	if c.Training.NHeads <= 0 {
		errors = append(errors, "n_heads deve ser maior que 0")
	}
	if c.Training.DModel%c.Training.NHeads != 0 {
		errors = append(errors, fmt.Sprintf("d_model (%d) deve ser divisível por n_heads (%d)", c.Training.DModel, c.Training.NHeads))
	}
	if c.Training.NumLayers <= 0 || c.Training.NumLayers > 24 {
		errors = append(errors, "num_layers deve estar entre 1 e 24")
	}
	if c.Training.MaxSeqLen <= 0 || c.Training.MaxSeqLen > 2048 {
		errors = append(errors, "max_seq_len deve estar entre 1 e 2048")
	}
	if c.Training.MaxSeqLen%32 != 0 {
		errors = append(errors, "max_seq_len deve ser múltiplo de 32")
	}
	if c.Training.FFHidden <= 0 {
		errors = append(errors, "ff_hidden deve ser maior que 0")
	}
	if c.Training.FFHidden < c.Training.DModel {
		errors = append(errors, "ff_hidden deve ser >= d_model")
	}

	// Validar vocabulário
	if c.Training.MaxVocab < 100 {
		errors = append(errors, "max_vocab deve ser >= 100")
	}
	if c.Training.MaxVocab > 50000 {
		errors = append(errors, "max_vocab não deve exceder 50000")
	}

	// Validar regularização
	if c.Training.DropoutRate < 0 || c.Training.DropoutRate > 0.5 {
		errors = append(errors, "dropout_rate deve estar entre 0 e 0.5")
	}
	if c.Training.WeightDecay < 0 || c.Training.WeightDecay > 0.5 {
		errors = append(errors, "weight_decay deve estar entre 0 e 0.5")
	}

	// Validar servidor
	if c.Server.Port == "" {
		errors = append(errors, "port não pode estar vazio")
	}
	if !strings.HasPrefix(c.Server.Port, ":") {
		errors = append(errors, "port deve começar com ':'")
	}

	// Validar paths
	if c.Paths.ModelPath == "" {
		errors = append(errors, "model_path não pode estar vazio")
	}
	if c.Paths.InputFile == "" {
		errors = append(errors, "input_file não pode estar vazio")
	}

	// Retornar todos os erros encontrados
	if len(errors) > 0 {
		return fmt.Errorf("configurações inválidas:\n  - %s", strings.Join(errors, "\n  - "))
	}

	return nil
}

// isValidEnvironment verifica se o ambiente é válido
func isValidEnvironment(env Environment) bool {
	switch env {
	case EnvDevelopment, EnvStaging, EnvProduction, EnvTest:
		return true
	default:
		return false
	}
}

// applyEnvironmentVariables aplica variáveis de ambiente à configuração
func (c *Config) applyEnvironmentVariables() {
	// Ambiente
	if env := os.Getenv("LMCS_ENV"); env != "" {
		c.Environment = Environment(env)
	}

	// Training
	if val := os.Getenv("LMCS_EPOCHS"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.Epochs = v
		}
	}
	if val := os.Getenv("LMCS_LEARNING_RATE"); val != "" {
		if v, err := strconv.ParseFloat(val, 64); err == nil {
			c.Training.LearningRate = v
		}
	}
	if val := os.Getenv("LMCS_BATCH_SIZE"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.BatchSize = v
		}
	}
	if val := os.Getenv("LMCS_TEMPERATURE"); val != "" {
		if v, err := strconv.ParseFloat(val, 64); err == nil {
			c.Training.Temperature = v
		}
	}
	if val := os.Getenv("LMCS_TOP_K"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.TopK = v
		}
	}
	if val := os.Getenv("LMCS_D_MODEL"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.DModel = v
		}
	}
	if val := os.Getenv("LMCS_N_HEADS"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.NHeads = v
		}
	}
	if val := os.Getenv("LMCS_NUM_LAYERS"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.NumLayers = v
		}
	}
	if val := os.Getenv("LMCS_MAX_SEQ_LEN"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.MaxSeqLen = v
		}
	}
	if val := os.Getenv("LMCS_FF_HIDDEN"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.FFHidden = v
		}
	}
	if val := os.Getenv("LMCS_MAX_VOCAB"); val != "" {
		if v, err := strconv.Atoi(val); err == nil {
			c.Training.MaxVocab = v
		}
	}
	if val := os.Getenv("LMCS_DROPOUT_RATE"); val != "" {
		if v, err := strconv.ParseFloat(val, 64); err == nil {
			c.Training.DropoutRate = v
		}
	}
	if val := os.Getenv("LMCS_WEIGHT_DECAY"); val != "" {
		if v, err := strconv.ParseFloat(val, 64); err == nil {
			c.Training.WeightDecay = v
		}
	}

	// Server
	if val := os.Getenv("LMCS_PORT"); val != "" {
		c.Server.Port = val
	}
	if val := os.Getenv("LMCS_HOST"); val != "" {
		c.Server.Host = val
	}

	// Paths
	if val := os.Getenv("LMCS_MODEL_PATH"); val != "" {
		c.Paths.ModelPath = val
	}
	if val := os.Getenv("LMCS_INPUT_FILE"); val != "" {
		c.Paths.InputFile = val
	}
}

// LoadFromEnvironment carrega configurações baseadas no ambiente
func LoadFromEnvironment() *Config {
	cfg := DefaultConfig()

	// Detectar ambiente
	if env := os.Getenv("LMCS_ENV"); env != "" {
		cfg.Environment = Environment(env)
	}

	// Aplicar configurações específicas por ambiente
	switch cfg.Environment {
	case EnvProduction:
		// Produção: mais épocas, modelo maior
		cfg.Training.Epochs = 500
		cfg.Training.DModel = 512
		cfg.Training.NumLayers = 6
		cfg.Training.DropoutRate = 0.1
		cfg.Server.Host = "0.0.0.0"
	case EnvStaging:
		// Staging: configuração intermediária
		cfg.Training.Epochs = 200
		cfg.Training.DModel = 256
		cfg.Training.NumLayers = 4
		cfg.Training.DropoutRate = 0.15
	case EnvTest:
		// Test: mínimo para testes rápidos
		cfg.Training.Epochs = 10
		cfg.Training.DModel = 64
		cfg.Training.NumLayers = 2
		cfg.Training.MaxVocab = 1000
		cfg.Training.BatchSize = 8
	case EnvDevelopment:
		// Development: defaults já estão configurados
		cfg.Training.DropoutRate = 0.05 // Menos dropout para debug
	}

	// Aplicar variáveis de ambiente (sobrescrevem presets)
	cfg.applyEnvironmentVariables()

	return cfg
}
