package api

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/model"
)

// AskRequest representa uma requisição de geração de texto
type AskRequest struct {
	Question    string  `json:"question"`
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
}

// AskResponse representa uma resposta da API
type AskResponse struct {
	Answer      string  `json:"answer"`
	Model       string  `json:"model"`
	ElapsedMs   int64   `json:"elapsed_ms"`
	VocabSize   int     `json:"vocab_size"`
	DModel      int     `json:"d_model"`
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
}

// ErrorResponse representa uma resposta de erro genérica
type ErrorResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error"`
}

// Handler agrupa os handlers da API
type Handler struct {
	model *model.TransformerModel
}

// NewHandler cria um novo handler com o modelo Transformer
func NewHandler(m *model.TransformerModel) *Handler {
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

	var req AskRequest

	// Tentar parse JSON primeiro (POST)
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("Erro ao parsear JSON: %v", err)
			h.sendError(w, "JSON inválido: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		// Fallback para query params (GET)
		req.Question = r.URL.Query().Get("q")
	}

	// Validações e defaults
	if req.Question == "" {
		req.Question = "oi"
	}
	if req.Temperature <= 0 || req.Temperature > 2.0 {
		req.Temperature = 0.7
	}
	if req.TopK <= 0 || req.TopK > 100 {
		req.TopK = 30
	}

	log.Printf("Recebida pergunta: %s (temp=%.2f, topk=%d)\n", req.Question, req.Temperature, req.TopK)

	startTime := time.Now()

	// Gerar resposta
	answer := h.model.Generate(req.Question, 100, req.Temperature, req.TopK)

	elapsed := time.Since(startTime)
	log.Printf("Resposta gerada em %s: %s\n", elapsed, answer)

	h.sendJSON(w, AskResponse{
		Answer:      answer,
		Model:       "Transformer",
		ElapsedMs:   elapsed.Milliseconds(),
		VocabSize:   len(h.model.Vocab),
		DModel:      h.model.DModel,
		Temperature: req.Temperature,
		TopK:        req.TopK,
	}, http.StatusOK)
}

// handleHealth handler para health check
func (h *Handler) handleHealth(w http.ResponseWriter, r *http.Request) {
	h.sendJSON(w, map[string]interface{}{
		"status":  "ok",
		"model":   "Transformer",
		"vocab":   len(h.model.Vocab),
		"d_model": h.model.DModel,
		"heads":   h.model.NHeads,
		"layers":  h.model.NLayers,
		"epochs":  h.model.EpochsTrained,
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
