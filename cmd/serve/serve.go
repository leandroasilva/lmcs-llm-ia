package serve

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/api"
	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
	"github.com/leandroasilva/lmcs-llm-ia/internal/middleware"
	"github.com/leandroasilva/lmcs-llm-ia/internal/model"
	"github.com/leandroasilva/lmcs-llm-ia/internal/training"
)

func RunServe(configPath string) error {
	// Configurar número de threads/cores para máxima performance
	numCPUs := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPUs)
	log.Printf("Usando %d threads/cores (GOMAXPROCS)\n", numCPUs)

	// Carregar configurações
	cfg := config.DefaultConfig()

	if _, err := os.Stat(configPath); err == nil {
		log.Printf("Carregando configurações de %s...\n", configPath)
		if loadedCfg, err := config.LoadFromFile(configPath); err != nil {
			log.Printf("Aviso: Erro ao carregar %s, usando padrões: %v\n", configPath, err)
		} else {
			cfg = loadedCfg
		}
	}

	// Validar configurações
	if err := cfg.Validate(); err != nil {
		return err
	}

	log.Printf("Configuração: %+v\n", cfg)

	// Carregar modelo
	if _, err := os.Stat(cfg.Paths.ModelPath); err != nil {
		return fmt.Errorf("modelo não encontrado em %s: %w", cfg.Paths.ModelPath, err)
	}

	log.Printf("Carregando modelo de %s...\n", cfg.Paths.ModelPath)

	var transformerMdl *model.TransformerModel
	var err error

	// Verificar se é JSON exportado ou binário Go
	if strings.HasSuffix(cfg.Paths.ModelPath, ".json") {
		transformerMdl, err = model.LoadTrainedModelFromJSON(cfg.Paths.ModelPath)
	} else {
		transformerMdl, err = model.LoadTransformerModel(cfg.Paths.ModelPath)
	}

	if err != nil {
		return err
	}

	log.Printf("Modelo carregado com sucesso! (épocas treinadas: %d)\n", transformerMdl.EpochsTrained)

	// Configurar servidor HTTP
	mux := http.NewServeMux()
	handler := api.NewHandler(transformerMdl)
	handler.RegisterRoutes(mux)

	// Registrar rotas de treinamento
	trainingHandler := api.NewTrainingHandler(training.GlobalMetrics)
	mux.HandleFunc("/api/training/status", trainingHandler.HandleTrainingStatus)          // SSE
	mux.HandleFunc("/api/training/status/json", trainingHandler.HandleTrainingStatusJSON) // JSON

	// Aplicar middlewares de segurança
	var httpHandler http.Handler = mux

	// 1. Security Headers
	httpHandler = middleware.SecurityHeaders(httpHandler)

	// 2. CORS (configurável)
	allowedOrigins := []string{"*"} // Em produção, especificar origens
	httpHandler = middleware.CORS(allowedOrigins)(httpHandler)

	// 3. Rate Limiting (100 req/min por IP)
	rateLimiter := middleware.NewRateLimiter(100, time.Minute)
	httpHandler = middleware.RateLimit(rateLimiter)(httpHandler)

	// 4. Timeout (30s por requisição)
	httpHandler = middleware.Timeout(30 * time.Second)(httpHandler)

	log.Printf("Servidor rodando em http://%s%s\n", cfg.Server.Host, cfg.Server.Port)
	log.Println("Frontend: http://localhost" + cfg.Server.Port)
	log.Println("API Endpoints:")
	log.Println("  GET  /api/health")
	log.Println("  POST /api/ask")
	log.Println("Security:")
	log.Println("  ✓ CORS enabled")
	log.Println("  ✓ Rate limiting: 100 req/min")
	log.Println("  ✓ Timeout: 30s")
	log.Println("  ✓ Security headers")

	// Iniciar servidor
	if err := http.ListenAndServe(cfg.Server.Port, httpHandler); err != nil {
		return err
	}

	return nil
}
