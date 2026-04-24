package main

import (
	"log"
	"net/http"
	"os"

	"github.com/leandroasilva/lmcs-llm-ia/api"
	"github.com/leandroasilva/lmcs-llm-ia/config"
	"github.com/leandroasilva/lmcs-llm-ia/model"
)

func main() {
	log.Println("=== LMCS LLM IA ===")

	// Carregar configurações
	cfg := config.DefaultConfig()

	// Tentar carregar de arquivo se existir
	if _, err := os.Stat("config.json"); err == nil {
		log.Println("Carregando configurações de config.json...")
		if loadedCfg, err := config.LoadFromFile("config.json"); err != nil {
			log.Printf("Aviso: Erro ao carregar config.json, usando padrões: %v\n", err)
		} else {
			cfg = loadedCfg
		}
	}

	// Validar configurações
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Configurações inválidas: %v\n", err)
	}

	log.Printf("Configuração: %+v\n", cfg)

	// Carregar texto de treinamento
	data, err := os.ReadFile(cfg.Paths.InputFile)
	var content string
	if err != nil {
		log.Printf("Aviso: Não foi possível ler %s, usando texto padrão\n", cfg.Paths.InputFile)
		content = "o rato roeu a roupa do rei de roma. o rei mandou buscar outro rato."
	} else {
		content = string(data)
		log.Printf("Texto carregado: %d caracteres\n", len(content))
	}

	// Carregar ou criar modelo
	var mdl *model.LmcsLLM
	if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
		log.Printf("Carregando modelo existente de %s...\n", cfg.Paths.ModelPath)
		mdl, err = model.Load(cfg.Paths.ModelPath)
		if err != nil {
			log.Fatalf("Erro ao carregar modelo: %v\n", err)
		}
		log.Println("Modelo carregado com sucesso!")
	} else {
		mdl = model.New(content)
	}

	// Treinar modelo
	mdl.Train(
		content,
		cfg.Training.Epochs,
		cfg.Training.LearningRate,
		cfg.Training.BatchSize,
	)

	// Salvar modelo treinado
	if err := mdl.Save(cfg.Paths.ModelPath); err != nil {
		log.Printf("Erro ao salvar modelo: %v\n", err)
	} else {
		log.Printf("Modelo salvo em %s\n", cfg.Paths.ModelPath)
	}

	// Configurar servidor HTTP
	mux := http.NewServeMux()
	handler := api.NewHandler(mdl)
	handler.RegisterRoutes(mux)

	log.Printf("API rodando em http://%s%s\n", cfg.Server.Host, cfg.Server.Port)
	log.Println("Endpoints:")
	log.Println("  GET  /ask?q=o&length=100&temperature=0.8")
	log.Println("  POST /ask {\"seed\": \"o\", \"length\": 100, \"temperature\": 0.8}")
	log.Println("  GET  /health")

	// Iniciar servidor
	if err := http.ListenAndServe(cfg.Server.Port, mux); err != nil {
		log.Fatalf("Erro ao iniciar servidor: %v\n", err)
	}
}
