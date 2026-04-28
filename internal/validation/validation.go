package validation

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Validator para validação de inputs
type Validator struct {
	Errors map[string]string
}

// NewValidator cria um novo validador
func NewValidator() *Validator {
	return &Validator{
		Errors: make(map[string]string),
	}
}

// HasErrors verifica se há erros
func (v *Validator) HasErrors() bool {
	return len(v.Errors) > 0
}

// GetErrors retorna todos os erros
func (v *Validator) GetErrors() map[string]string {
	return v.Errors
}

// ValidateString valida string
func (v *Validator) ValidateString(field, value string, minLength, maxLength int) {
	if value == "" {
		v.Errors[field] = "is required"
		return
	}

	length := utf8.RuneCountInString(value)
	if minLength > 0 && length < minLength {
		v.Errors[field] = "must be at least " + strconv.Itoa(minLength) + " characters"
	}
	if maxLength > 0 && length > maxLength {
		v.Errors[field] = "must be at most " + strconv.Itoa(maxLength) + " characters"
	}
}

// ValidateInt valida inteiro
func (v *Validator) ValidateInt(field string, value, min, max int) {
	if min > 0 && value < min {
		v.Errors[field] = "must be at least " + strconv.Itoa(min)
	}
	if max > 0 && value > max {
		v.Errors[field] = "must be at most " + strconv.Itoa(max)
	}
}

// ValidateFloat valida float
func (v *Validator) ValidateFloat(field string, value, min, max float64) {
	if min > 0 && value < min {
		v.Errors[field] = "must be at least " + fmt.Sprintf("%.2f", min)
	}
	if max > 0 && value > max {
		v.Errors[field] = "must be at most " + fmt.Sprintf("%.2f", max)
	}
}

// ValidateEmail valida email
func (v *Validator) ValidateEmail(field, email string) {
	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(email) {
		v.Errors[field] = "is not a valid email"
	}
}

// ValidatePattern valida com regex
func (v *Validator) ValidatePattern(field, value, pattern string) {
	re := regexp.MustCompile(pattern)
	if !re.MatchString(value) {
		v.Errors[field] = "has invalid format"
	}
}

// ValidateInList valida se está em uma lista
func (v *Validator) ValidateInList(field, value string, allowed []string) {
	for _, a := range allowed {
		if a == value {
			return
		}
	}
	v.Errors[field] = "must be one of: " + strings.Join(allowed, ", ")
}

// TrainingConfigValidator valida configuração de treino
type TrainingConfigValidator struct{}

// Validate valida configuração de treino
func (v *TrainingConfigValidator) Validate(config interface {
	GetEpochs() int
	GetLearningRate() float64
	GetBatchSize() int
	GetMaxVocab() int
	GetDropoutRate() float64
}) *Validator {
	val := NewValidator()

	val.ValidateInt("epochs", config.GetEpochs(), 1, 10000)
	val.ValidateFloat("learning_rate", config.GetLearningRate(), 0.0001, 1.0)
	val.ValidateInt("batch_size", config.GetBatchSize(), 1, 1024)
	val.ValidateInt("max_vocab", config.GetMaxVocab(), 100, 50000)
	val.ValidateFloat("dropout_rate", config.GetDropoutRate(), 0.0, 0.9)

	return val
}

// GenerationRequestValidator valida requisição de geração
type GenerationRequestValidator struct{}

// Validate valida requisição de geração
func (v *GenerationRequestValidator) Validate(prompt string, maxTokens int, temperature float64) *Validator {
	val := NewValidator()

	val.ValidateString("prompt", prompt, 1, 10000)
	val.ValidateInt("max_tokens", maxTokens, 1, 1000)
	val.ValidateFloat("temperature", temperature, 0.1, 2.0)

	return val
}
