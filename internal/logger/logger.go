package logger

import (
	"context"
	"log/slog"
	"os"
)

var logger *slog.Logger

// InitLogger inicializa o logger estruturado
func InitLogger(level slog.Level, format string) {
	var handler slog.Handler

	opts := &slog.HandlerOptions{
		Level: level,
	}

	if format == "json" {
		handler = slog.NewJSONHandler(os.Stdout, opts)
	} else {
		handler = slog.NewTextHandler(os.Stdout, opts)
	}

	logger = slog.New(handler)
	slog.SetDefault(logger)
}

// GetLogger retorna o logger global
func GetLogger() *slog.Logger {
	if logger == nil {
		InitLogger(slog.LevelInfo, "text")
	}
	return logger
}

// Helper functions para logging estruturado

// Info log de informação
func Info(msg string, args ...any) {
	GetLogger().Info(msg, args...)
}

// Error log de erro
func Error(msg string, args ...any) {
	GetLogger().Error(msg, args...)
}

// Warn log de aviso
func Warn(msg string, args ...any) {
	GetLogger().Warn(msg, args...)
}

// Debug log de debug
func Debug(msg string, args ...any) {
	GetLogger().Debug(msg, args...)
}

// With cria um logger com campos adicionais
func With(args ...any) *slog.Logger {
	return GetLogger().With(args...)
}

// Context com contexto
func InfoCtx(ctx context.Context, msg string, args ...any) {
	GetLogger().InfoContext(ctx, msg, args...)
}

func ErrorCtx(ctx context.Context, msg string, args ...any) {
	GetLogger().ErrorContext(ctx, msg, args...)
}

// Training logger helper
func TrainingLog(epoch int, totalEpochs int, loss float64, perplexity float64, args ...any) {
	allArgs := append([]any{
		"epoch", epoch,
		"total_epochs", totalEpochs,
		"loss", loss,
		"perplexity", perplexity,
		"progress_pct", float64(epoch) / float64(totalEpochs) * 100,
	}, args...)

	GetLogger().Info("Training epoch completed", allArgs...)
}

// Model logger helper
func ModelLog(action string, modelPath string, args ...any) {
	allArgs := append([]any{
		"action", action,
		"model_path", modelPath,
	}, args...)

	GetLogger().Info("Model operation", allArgs...)
}

// API logger helper
func APILog(method string, path string, status int, args ...any) {
	allArgs := append([]any{
		"method", method,
		"path", path,
		"status", status,
	}, args...)

	GetLogger().Info("API request", allArgs...)
}
