package serve

import (
	"log"
	"net/http"
	"os"
	"runtime"

	"github.com/leandroasilva/lmcs-llm-ia/internal/api"
	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
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
		return nil
	}

	log.Printf("Carregando modelo de %s...\n", cfg.Paths.ModelPath)
	transformerMdl, err := model.LoadTransformerModel(cfg.Paths.ModelPath)
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

	log.Printf("Servidor rodando em http://%s%s\n", cfg.Server.Host, cfg.Server.Port)
	log.Println("Frontend: http://localhost" + cfg.Server.Port)
	log.Println("API Endpoints:")
	log.Println("  GET  /api/health")
	log.Println("  POST /api/ask")

	// Iniciar servidor
	if err := http.ListenAndServe(cfg.Server.Port, mux); err != nil {
		return err
	}

	return nil
}
