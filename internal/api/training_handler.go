package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/training"
)

// TrainingHandler gerencia o streaming de métricas de treinamento
type TrainingHandler struct {
	metrics *training.TrainingMetrics
}

// NewTrainingHandler cria um novo handler de treinamento
func NewTrainingHandler(metrics *training.TrainingMetrics) *TrainingHandler {
	return &TrainingHandler{metrics: metrics}
}

// HandleTrainingStatus endpoint SSE para streaming de métricas
func (h *TrainingHandler) HandleTrainingStatus(w http.ResponseWriter, r *http.Request) {
	// Headers SSE
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming não suportado", http.StatusInternalServerError)
		return
	}

	// Enviar métricas iniciais
	snapshot := h.metrics.GetSnapshot()
	if err := h.sendSSE(w, snapshot); err != nil {
		return
	}
	flusher.Flush()

	// Enviar atualizações a cada segundo
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		snapshot := h.metrics.GetSnapshot()

		if err := h.sendSSE(w, snapshot); err != nil {
			return
		}
		flusher.Flush()

		// Parar se não estiver mais treinando
		if !snapshot.IsTraining {
			break
		}
	}
}

// HandleTrainingStatusJSON endpoint JSON simples para status
func (h *TrainingHandler) HandleTrainingStatusJSON(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	snapshot := h.metrics.GetSnapshot()
	json.NewEncoder(w).Encode(snapshot)
}

// sendSSE envia evento SSE
func (h *TrainingHandler) sendSSE(w http.ResponseWriter, data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", jsonData)
	return err
}
