package training

import (
	"fmt"
	"math"
	"sync"
	"time"
)

// TrainingMetrics armazena métricas de treinamento em tempo real
type TrainingMetrics struct {
	mu sync.RWMutex

	// Status atual
	IsTraining   bool      `json:"is_training"`
	StartTime    time.Time `json:"start_time"`
	CurrentEpoch int       `json:"current_epoch"`
	TotalEpochs  int       `json:"total_epochs"`

	// Métricas de loss
	CurrentLoss float64   `json:"current_loss"`
	AvgLoss     float64   `json:"avg_loss"`
	BestLoss    float64   `json:"best_loss"`
	LossHistory []float64 `json:"loss_history"`

	// Métricas calculadas
	Perplexity     float64 `json:"perplexity"`
	BestPerplexity float64 `json:"best_perplexity"`

	// Performance
	SamplesPerSecond   float64 `json:"samples_per_second"`
	ElapsedTime        string  `json:"elapsed_time"`
	EstimatedRemaining string  `json:"estimated_remaining"`

	// Dataset
	TotalSamples     int `json:"total_samples"`
	ProcessedSamples int `json:"processed_samples"`

	// Configuração atual
	LearningRate float64 `json:"learning_rate"`
	BatchSize    int     `json:"batch_size"`
}

// GlobalMetrics é a instância global de métricas
var GlobalMetrics = NewTrainingMetrics()

// NewTrainingMetrics cria uma nova instância de métricas
func NewTrainingMetrics() *TrainingMetrics {
	return &TrainingMetrics{
		BestLoss:       math.MaxFloat64,
		BestPerplexity: math.MaxFloat64,
		LossHistory:    make([]float64, 0, 1000),
	}
}

// StartTraining inicia o registro de treinamento
func (tm *TrainingMetrics) StartTraining(totalEpochs int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.IsTraining = true
	tm.StartTime = time.Now()
	tm.TotalEpochs = totalEpochs
	tm.CurrentEpoch = 0
	tm.LossHistory = tm.LossHistory[:0]
}

// UpdateEpoch atualiza métricas de uma época
func (tm *TrainingMetrics) UpdateEpoch(epoch int, loss float64, samplesProcessed int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.CurrentEpoch = epoch
	tm.CurrentLoss = loss
	tm.ProcessedSamples = samplesProcessed

	// Calcular perplexidade: exp(loss)
	tm.Perplexity = tm.exp(loss)

	// Atualizar melhor loss
	if loss < tm.BestLoss {
		tm.BestLoss = loss
		tm.BestPerplexity = tm.Perplexity
	}

	// Manter histórico (últimos 1000 epochs)
	if len(tm.LossHistory) < 1000 {
		tm.LossHistory = append(tm.LossHistory, loss)
	} else {
		// Calcular média móvel
		tm.LossHistory = append(tm.LossHistory[1:], loss)
	}

	// Calcular loss média
	tm.AvgLoss = tm.average(tm.LossHistory)

	// Calcular samples per second
	elapsed := time.Since(tm.StartTime).Seconds()
	if elapsed > 0 {
		tm.SamplesPerSecond = float64(samplesProcessed) / elapsed
	}

	// Formatar tempo decorrido
	tm.ElapsedTime = tm.formatDuration(time.Since(tm.StartTime))

	// Estimar tempo restante
	if epoch > 0 {
		timePerEpoch := time.Since(tm.StartTime) / time.Duration(epoch)
		remainingEpochs := tm.TotalEpochs - epoch
		estimatedRemaining := timePerEpoch * time.Duration(remainingEpochs)
		tm.EstimatedRemaining = tm.formatDuration(estimatedRemaining)
	}
}

// EndTraining finaliza o treinamento
func (tm *TrainingMetrics) EndTraining() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.IsTraining = false
}

// GetSnapshot retorna uma cópia snapshot das métricas
func (tm *TrainingMetrics) GetSnapshot() *TrainingMetrics {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	return &TrainingMetrics{
		IsTraining:         tm.IsTraining,
		StartTime:          tm.StartTime,
		CurrentEpoch:       tm.CurrentEpoch,
		TotalEpochs:        tm.TotalEpochs,
		CurrentLoss:        tm.CurrentLoss,
		AvgLoss:            tm.AvgLoss,
		BestLoss:           tm.BestLoss,
		LossHistory:        tm.LossHistory,
		Perplexity:         tm.Perplexity,
		BestPerplexity:     tm.BestPerplexity,
		SamplesPerSecond:   tm.SamplesPerSecond,
		ElapsedTime:        tm.ElapsedTime,
		EstimatedRemaining: tm.EstimatedRemaining,
		TotalSamples:       tm.TotalSamples,
		ProcessedSamples:   tm.ProcessedSamples,
		LearningRate:       tm.LearningRate,
		BatchSize:          tm.BatchSize,
	}
}

// exp calcula e^x usando série de Taylor
func (tm *TrainingMetrics) exp(x float64) float64 {
	if x > 20 {
		x = 20 // Prevenir overflow
	}
	if x < -20 {
		return 0.000000002
	}

	result := 1.0
	term := 1.0
	for i := 1; i < 100; i++ {
		term *= x / float64(i)
		result += term
		if math.Abs(term) < 1e-10 {
			break
		}
	}
	return result
}

// average calcula a média de um slice
func (tm *TrainingMetrics) average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// formatDuration formata duração em string legível
func (tm *TrainingMetrics) formatDuration(d time.Duration) string {
	if d < time.Minute {
		return d.Round(time.Second).String()
	} else if d < time.Hour {
		minutes := int(d.Minutes())
		seconds := int(d.Seconds()) % 60
		return formatTime(minutes, seconds)
	} else {
		hours := int(d.Hours())
		minutes := int(d.Minutes()) % 60
		seconds := int(d.Seconds()) % 60
		return formatTime(hours*60+minutes, seconds)
	}
}

// formatTime formata minutos e segundos
func formatTime(minutes, seconds int) string {
	if minutes < 60 {
		return fmt.Sprintf("%dm%ds", minutes, seconds)
	}
	hours := minutes / 60
	mins := minutes % 60
	return fmt.Sprintf("%dh%dm%ds", hours, mins, seconds)
}
