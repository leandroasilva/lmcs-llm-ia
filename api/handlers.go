package api

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/leandroasilva/lmcs-llm-ia/model"
)

// GenerateRequest representa uma requisição de geração de texto
type GenerateRequest struct {
	Seed        string  `json:"seed"`
	Length      int     `json:"length"`
	Temperature float64 `json:"temperature"`
}

// GenerateResponse representa uma resposta da API
type GenerateResponse struct {
	Success bool   `json:"success"`
	Result  string `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
}

// ErrorResponse representa uma resposta de erro genérica
type ErrorResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
}

// Handler agrupa os handlers da API
type Handler struct {
	model *model.LmcsLLM
}

// NewHandler cria um novo handler com o modelo fornecido
func NewHandler(m *model.LmcsLLM) *Handler {
	return &Handler{model: m}
}

// RegisterRoutes registra todas as rotas da API
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	// Rotas da API
	mux.HandleFunc("/api/ask", h.handleAsk)
	mux.HandleFunc("/api/health", h.handleHealth)

	// Arquivos estáticos
	fs := http.FileServer(http.Dir("static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))

	// Redirecionar raiz para o frontend
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.ServeFile(w, r, "static/index.html")
		}
	})
}

// handleAsk handler para geração de texto
func (h *Handler) handleAsk(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")

	// Handle preflight
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}

	var req GenerateRequest

	// Tentar parse JSON primeiro (POST)
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("Erro ao parsear JSON: %v", err)
			h.sendError(w, "JSON inválido: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		// Fallback para query params (GET)
		req.Seed = r.URL.Query().Get("q")
		req.Length = 100
		req.Temperature = 0.8
	}

	// Validações
	if req.Seed == "" {
		req.Seed = "o"
	}
	if req.Length <= 0 || req.Length > 1000 {
		req.Length = 100
	}
	if req.Temperature <= 0 || req.Temperature > 2.0 {
		req.Temperature = 0.8
	}

	// Extrair primeiro caractere da seed
	seedRunes := []rune(req.Seed)
	if len(seedRunes) == 0 {
		h.sendError(w, "Seed inválida", http.StatusBadRequest)
		return
	}
	seed := seedRunes[0]

	// Gerar texto
	resp := h.model.Generate(seed, req.Length, req.Temperature)

	h.sendJSON(w, GenerateResponse{
		Success: true,
		Result:  resp,
	}, http.StatusOK)
}

// handleHealth handler para health check
func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	h.sendJSON(w, map[string]string{
		"status": "ok",
		"model":  "LMCS LLM",
	}, http.StatusOK)
}

// sendJSON helper para enviar respostas JSON
func (h *Handler) sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		log.Printf("Erro ao codificar resposta JSON: %v", err)
	}
}

// sendError helper para enviar respostas de erro
func (h *Handler) sendError(w http.ResponseWriter, message string, status int) {
	h.sendJSON(w, ErrorResponse{
		Success: false,
		Error:   message,
	}, status)
}
