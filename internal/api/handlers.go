package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/model"
	"github.com/leandroasilva/lmcs-llm-ia/internal/sanitizer"
	"github.com/leandroasilva/lmcs-llm-ia/internal/validation"
)

// AskRequest representa uma requisição de geração de texto
type AskRequest struct {
	Question    string  `json:"question"`
	Temperature float64 `json:"temperature"`
	TopK        int     `json:"top_k"`
	MaxTokens   int     `json:"max_tokens"`
}

// StreamResponse representa um chunk de resposta streaming
type StreamResponse struct {
	Token   string `json:"token"`
	Done    bool   `json:"done"`
	Elapsed int64  `json:"elapsed_ms,omitempty"`
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
	mux.HandleFunc("/api/ask/stream", h.handleAskStream) // Streaming SSE
	mux.HandleFunc("/api/health", h.handleHealth)

	// Arquivos estáticos legados (fallback se frontend/dist não existir)
	fs := http.FileServer(http.Dir("static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))
}

// handleAsk handler para geração de texto
func (h *Handler) handleAsk(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

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

	// Sanitizar input
	req.Question = sanitizer.SanitizePrompt(req.Question)

	// Validar input
	val := validation.NewValidator()
	val.ValidateString("question", req.Question, 1, 10000)
	if val.HasErrors() {
		h.sendError(w, fmt.Sprintf("Validation failed: %v", val.GetErrors()), http.StatusBadRequest)
		return
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
	if req.MaxTokens <= 0 || req.MaxTokens > 512 {
		req.MaxTokens = 150
	}

	log.Printf("Recebida pergunta: %s (temp=%.2f, topk=%d, max_tokens=%d)\n", req.Question, req.Temperature, req.TopK, req.MaxTokens)

	startTime := time.Now()

	// Formatar como conversação para o modelo
	conversationalPrompt := fmt.Sprintf("Usuário: %s\nAssistente: ", req.Question)

	// Gerar resposta usando KV cache para speedup
	fullResponse := h.model.GenerateWithKVCache(conversationalPrompt, req.MaxTokens, req.Temperature, req.TopK)

	// Extrair apenas a resposta do assistente
	answer := extractAssistantResponse(fullResponse)

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

// handleAskStream handler para geração de texto com streaming SSE
func (h *Handler) handleAskStream(w http.ResponseWriter, r *http.Request) {
	// Headers para SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Verificar se suporta flushing
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming não suportado", http.StatusInternalServerError)
		return
	}

	// Parse request
	var req AskRequest
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			sendSSEError(w, "JSON inválido: "+err.Error())
			return
		}
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
	if req.MaxTokens <= 0 || req.MaxTokens > 512 {
		req.MaxTokens = 150
	}

	log.Printf("[STREAM] Recebida pergunta: %s (temp=%.2f, topk=%d, max_tokens=%d)\n", req.Question, req.Temperature, req.TopK, req.MaxTokens)

	startTime := time.Now()

	// Formatar prompt
	conversationalPrompt := fmt.Sprintf("Usuário: %s\nAssistente: ", req.Question)

	// Gerar com streaming usando KV cache (mesma lógica do GenerateWithKVCache)
	h.model.GenerateStream(conversationalPrompt, req.MaxTokens, req.Temperature, req.TopK, func(token string) bool {
		data := StreamResponse{
			Token: token,
			Done:  false,
		}
		if err := sendSSE(w, data); err != nil {
			log.Printf("[STREAM] Erro ao enviar: %v\n", err)
			return false
		}
		flusher.Flush()
		time.Sleep(30 * time.Millisecond)
		return true
	})

	// Enviar sinal de done
	elapsed := time.Since(startTime)
	data := StreamResponse{
		Token:   "",
		Done:    true,
		Elapsed: elapsed.Milliseconds(),
	}
	sendSSE(w, data)
	flusher.Flush()

	log.Printf("[STREAM] Resposta completada em %s\n", elapsed)
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

// extractAssistantResponse extrai apenas a resposta do assistente
func extractAssistantResponse(fullResponse string) string {
	// Procurar por "Assistente: " e pegar o texto após
	if idx := strings.Index(fullResponse, "Assistente: "); idx != -1 {
		response := fullResponse[idx+len("Assistente: "):]

		// Parar no próximo "Usuário: " se existir
		if userIdx := strings.Index(response, "\nUsuário: "); userIdx != -1 {
			response = response[:userIdx]
		}

		// Limpar espaços excessivos
		response = strings.TrimSpace(response)

		return response
	}

	// Se não encontrar padrão, retornar texto limpo
	return strings.TrimSpace(fullResponse)
}

// sendSSE envia um evento SSE
func sendSSE(w http.ResponseWriter, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	fmt.Fprintf(w, "data: %s\n\n", jsonData)
	return nil
}

// sendSSEError envia um erro via SSE
func sendSSEError(w http.ResponseWriter, message string) {
	data := StreamResponse{
		Token: fmt.Sprintf("[ERRO] %s", message),
		Done:  true,
	}
	sendSSE(w, data)
}
