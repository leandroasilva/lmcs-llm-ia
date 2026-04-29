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

	log.Printf("Recebida pergunta: %s (temp=%.2f, topk=%d)\n", req.Question, req.Temperature, req.TopK)

	startTime := time.Now()

	// Formatar como conversação para o modelo
	conversationalPrompt := fmt.Sprintf("Usuário: %s\nAssistente: ", req.Question)

	// Gerar resposta usando KV cache para speedup
	fullResponse := h.model.GenerateWithKVCache(conversationalPrompt, 150, req.Temperature, req.TopK)

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

	log.Printf("[STREAM] Recebida pergunta: %s (temp=%.2f, topk=%d)\n", req.Question, req.Temperature, req.TopK)

	startTime := time.Now()

	// Tokenizar prompt
	conversationalPrompt := fmt.Sprintf("Usuário: %s\nAssistente: ", req.Question)
	promptTokens := h.model.Tokenize(conversationalPrompt)

	// Gerar token por token com beam search
	generatedTokens := h.generateTokensStreaming(promptTokens, 150, req.Temperature)

	// Enviar tokens via SSE
	for _, tokenID := range generatedTokens {
		if token, ok := h.model.IDToWord[tokenID]; ok {
			if tokenID != h.model.SpecialTokens["<PAD>"] &&
				tokenID != h.model.SpecialTokens["<BOS>"] &&
				tokenID != h.model.SpecialTokens["<EOS>"] {
				// Enviar token
				data := StreamResponse{
					Token: token + " ",
					Done:  false,
				}
				if err := sendSSE(w, data); err != nil {
					log.Printf("[STREAM] Erro ao enviar: %v\n", err)
					return
				}
				flusher.Flush()

				// Pequena pausa para simular digitação (opcional)
				time.Sleep(30 * time.Millisecond)
			}
		}
	}

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

// generateTokensStreaming gera tokens um por um para streaming
func (h *Handler) generateTokensStreaming(promptTokens []int, maxTokens int, temperature float64) []int {
	var generated []int
	currentTokens := make([]int, len(promptTokens))
	copy(currentTokens, promptTokens)

	for i := 0; i < maxTokens; i++ {
		// Forward pass
		output := h.model.Forward(currentTokens)

		// Pegar última posição
		seqLen := output.RawMatrix().Rows
		lastRow := make([]float64, h.model.DModel)
		for j := 0; j < h.model.DModel; j++ {
			lastRow[j] = output.At(seqLen-1, j)
		}

		// Calcular logits
		logits := make([]float64, h.model.VocabSize)
		for v := 0; v < h.model.VocabSize; v++ {
			logits[v] = h.model.BOut.At(v, 0)
			for j := 0; j < h.model.DModel; j++ {
				logits[v] += h.model.WOut.At(v, j) * lastRow[j]
			}
		}

		// Aplicar temperatura aos logits antes do softmax
		if temperature > 0 && temperature != 1.0 {
			for i := range logits {
				logits[i] /= temperature
			}
		}

		// Aplicar softmax
		probs := model.Softmax(logits)

		// Sample token com top-k
		nextToken := model.SampleTopK(probs, 30)

		// Verificar se é token de fim
		if nextToken == h.model.SpecialTokens["<EOS>"] {
			break
		}

		generated = append(generated, nextToken)
		currentTokens = append(currentTokens, nextToken)

		// Limitar tamanho da sequência
		if len(currentTokens) > h.model.MaxSeqLen {
			currentTokens = currentTokens[1:]
		}
	}

	return generated
}
