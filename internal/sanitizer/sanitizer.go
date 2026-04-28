package sanitizer

import (
	"html"
	"regexp"
	"strings"
)

// SanitizeInput remove caracteres perigosos e faz escape HTML
func SanitizeInput(input string) string {
	// Trim whitespace
	input = strings.TrimSpace(input)

	// Escape HTML para prevenir XSS
	input = html.EscapeString(input)

	// Remover scripts
	scriptRegex := regexp.MustCompile(`(?i)<script[^>]*>.*?</script>`)
	input = scriptRegex.ReplaceAllString(input, "")

	// Remover tags HTML perigosas
	dangerousTags := []string{
		"script", "iframe", "object", "embed", "form", "input",
		"textarea", "button", "select", "link", "meta", "style",
	}

	for _, tag := range dangerousTags {
		tagRegex := regexp.MustCompile(`(?i)</?` + tag + `[^>]*>`)
		input = tagRegex.ReplaceAllString(input, "")
	}

	return input
}

// SanitizePrompt sanitiza prompt de geração de texto
func SanitizePrompt(prompt string) string {
	// Remover caracteres de controle
	controlChars := regexp.MustCompile(`[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]`)
	prompt = controlChars.ReplaceAllString(prompt, "")

	// Limitar tamanho
	if len(prompt) > 10000 {
		prompt = prompt[:10000]
	}

	return SanitizeInput(prompt)
}

// SanitizeFilename sanitiza nome de arquivo
func SanitizeFilename(filename string) string {
	// Remover caracteres perigosos
	dangerous := regexp.MustCompile(`[<>:"/\\|?*\x00-\x1F]`)
	filename = dangerous.ReplaceAllString(filename, "")

	// Remover paths
	filename = strings.ReplaceAll(filename, "../", "")
	filename = strings.ReplaceAll(filename, "..\\", "")

	// Limitar tamanho
	if len(filename) > 255 {
		filename = filename[:255]
	}

	return filename
}

// SanitizeSQL previne SQL injection (básico)
func SanitizeSQL(input string) string {
	// Remover caracteres perigosos para SQL
	dangerous := []string{"'", "\"", ";", "--", "/*", "*/", "xp_", "exec"}

	for _, char := range dangerous {
		input = strings.ReplaceAll(input, char, "")
	}

	return input
}

// EscapeJSON escape caracteres especiais em JSON
func EscapeJSON(input string) string {
	input = strings.ReplaceAll(input, "\\", "\\\\")
	input = strings.ReplaceAll(input, "\"", "\\\"")
	input = strings.ReplaceAll(input, "\n", "\\n")
	input = strings.ReplaceAll(input, "\r", "\\r")
	input = strings.ReplaceAll(input, "\t", "\\t")
	return input
}

// ValidateInputLength valida tamanho do input
func ValidateInputLength(input string, min, max int) bool {
	length := len(input)
	return length >= min && length <= max
}

// ContainsOnlyAlphanumeric verifica se contém apenas alfanuméricos
func ContainsOnlyAlphanumeric(input string) bool {
	alphanumeric := regexp.MustCompile(`^[a-zA-Z0-9]+$`)
	return alphanumeric.MatchString(input)
}

// ContainsOnlyLetters verifica se contém apenas letras
func ContainsOnlyLetters(input string) bool {
	letters := regexp.MustCompile(`^[a-zA-Z]+$`)
	return letters.MatchString(input)
}

// RemoveEmoji remove emojis
func RemoveEmoji(input string) string {
	emojiRegex := regexp.MustCompile(`[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F700}-\x{1F77F}\x{1F780}-\x{1F7FF}\x{1F800}-\x{1F8FF}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{FE00}-\x{FE0F}\x{1F000}-\x{1FFFF}]`)
	return emojiRegex.ReplaceAllString(input, "")
}
