package serve

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/api"
	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
	"github.com/leandroasilva/lmcs-llm-ia/internal/middleware"
	"github.com/leandroasilva/lmcs-llm-ia/internal/model"
)

// spaHandler serves a Single Page Application from a directory,
// falling back to index.html for unknown paths.
type spaHandler struct {
	staticPath string
	indexPath  string
}

func newSPAHandler(staticPath string) *spaHandler {
	return &spaHandler{
		staticPath: staticPath,
		indexPath:  "index.html",
	}
}

func (h *spaHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path

	// API routes should not be handled by SPA
	if strings.HasPrefix(path, "/api/") {
		http.NotFound(w, r)
		return
	}

	// Try to serve static file
	fp := filepath.Join(h.staticPath, filepath.Clean(path))
	info, err := os.Stat(fp)
	if err == nil && !info.IsDir() {
		http.ServeFile(w, r, fp)
		return
	}

	// Fallback to index.html for SPA routing
	indexFile := filepath.Join(h.staticPath, h.indexPath)
	http.ServeFile(w, r, indexFile)
}

func RunServe(configPath string) error {
	// Configurar número de threads/cores para máxima performance
	numCPUs := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPUs)
	log.Printf("Usando %d threads/cores (GOMAXPROCS)\n", numCPUs)

	// Carregar configurações
	cfg := config.DefaultConfig()

	configIsModelJSON := false
	if _, err := os.Stat(configPath); err == nil {
		log.Printf("Carregando configurações de %s...\n", configPath)
		if loadedCfg, err := config.LoadFromFile(configPath); err != nil {
			// Se falhar ao parsear como config e é um arquivo .json,
			// pode ser o modelo exportado diretamente
			if strings.HasSuffix(configPath, ".json") {
				log.Printf("Arquivo JSON detectado como modelo exportado: %s\n", configPath)
				configIsModelJSON = true
				cfg.Paths.ModelPath = configPath
			} else {
				log.Printf("Aviso: Erro ao carregar %s, usando padrões: %v\n", configPath, err)
			}
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
		if configIsModelJSON {
			return fmt.Errorf("modelo JSON não encontrado em %s: %w", cfg.Paths.ModelPath, err)
		}
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

	// Servir frontend React build (SPA fallback)
	spaHandler := newSPAHandler("frontend/dist")
	mux.Handle("/", spaHandler)

	// Aplicar middlewares de segurança
	var httpHandler http.Handler = mux

	// 1. Security Headers
	httpHandler = middleware.SecurityHeaders(httpHandler)

	// 2. CORS (configurável)
	allowedOrigins := []string{"*", "http://localhost:5173", "http://localhost:3000"}
	httpHandler = middleware.CORS(allowedOrigins)(httpHandler)

	// 3. Rate Limiting (100 req/min por IP)
	rateLimiter := middleware.NewRateLimiter(100, time.Minute)
	httpHandler = middleware.RateLimit(rateLimiter)(httpHandler)

	// 4. Timeout (60s por requisição para streaming)
	httpHandler = middleware.Timeout(60 * time.Second)(httpHandler)

	log.Printf("Servidor rodando em http://%s%s\n", cfg.Server.Host, cfg.Server.Port)
	log.Println("Frontend: http://localhost" + cfg.Server.Port)
	log.Println("API Endpoints:")
	log.Println("  GET  /api/health")
	log.Println("  POST /api/ask")
	log.Println("  POST /api/ask/stream")
	log.Println("Security:")
	log.Println("  ✓ CORS enabled")
	log.Println("  ✓ Rate limiting: 100 req/min")
	log.Println("  ✓ Timeout: 60s")
	log.Println("  ✓ Security headers")

	// Iniciar servidor
	if err := http.ListenAndServe(cfg.Server.Port, httpHandler); err != nil {
		return err
	}

	return nil
}
