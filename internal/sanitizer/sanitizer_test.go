package sanitizer

import (
	"strings"
	"testing"
)

func TestSanitizeInput_XSS(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "script tag",
			input:    "<script>alert('xss')</script>",
			expected: "&lt;script&gt;alert(&#39;xss&#39;)&lt;/script&gt;",
		},
		{
			name:     "iframe tag",
			input:    "<iframe src='evil.com'></iframe>",
			expected: "&lt;iframe src=&#39;evil.com&#39;&gt;&lt;/iframe&gt;",
		},
		{
			name:     "HTML entities",
			input:    "<div>Hello</div>",
			expected: "&lt;div&gt;Hello&lt;/div&gt;",
		},
		{
			name:     "normal text",
			input:    "Hello World",
			expected: "Hello World",
		},
		{
			name:     "whitespace",
			input:    "  Hello  ",
			expected: "Hello",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SanitizeInput(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestSanitizePrompt_ControlChars(t *testing.T) {
	input := "Hello\x00World\x01Test"
	result := SanitizePrompt(input)

	if strings.ContainsAny(result, "\x00\x01") {
		t.Errorf("Control characters not removed: %q", result)
	}

	t.Logf("Sanitized: %q", result)
}

func TestSanitizePrompt_LengthLimit(t *testing.T) {
	// Create very long input
	input := strings.Repeat("A", 15000)
	result := SanitizePrompt(input)

	if len(result) > 10000 {
		t.Errorf("Result too long: %d chars", len(result))
	}

	t.Logf("Original: %d, Sanitized: %d", len(input), len(result))
}

func TestSanitizeFilename_DangerousChars(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "path traversal",
			input:    "../../../etc/passwd",
			expected: "......etcpasswd",
		},
		{
			name:     "special chars",
			input:    "file<>name.txt",
			expected: "filename.txt",
		},
		{
			name:     "windows path",
			input:    "C:\\Windows\\System32",
			expected: "CWindowsSystem32",
		},
		{
			name:     "normal filename",
			input:    "document.pdf",
			expected: "document.pdf",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SanitizeFilename(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestSanitizeFilename_LengthLimit(t *testing.T) {
	input := strings.Repeat("a", 300) + ".txt"
	result := SanitizeFilename(input)

	if len(result) > 255 {
		t.Errorf("Filename too long: %d chars", len(result))
	}
}

func TestSanitizeSQL_Injection(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"quote injection", "'; DROP TABLE users; --"},
		{"comment injection", "SELECT * FROM users /* comment */"},
		{"exec injection", "'; EXEC xp_cmdshell('dir'); --"},
		{"union injection", "' UNION SELECT * FROM passwords --"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := SanitizeSQL(tt.input)

			// Check that dangerous characters are removed
			if strings.Contains(result, "'") {
				t.Error("Single quotes not removed")
			}
			if strings.Contains(result, ";") {
				t.Error("Semicolons not removed")
			}
			if strings.Contains(result, "--") {
				t.Error("SQL comments not removed")
			}

			t.Logf("Original: %s, Sanitized: %s", tt.input, result)
		})
	}
}

func TestEscapeJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "backslash",
			input:    "path\\to\\file",
			expected: "path\\\\to\\\\file",
		},
		{
			name:     "quotes",
			input:    "He said \"hello\"",
			expected: "He said \\\"hello\\\"",
		},
		{
			name:     "newlines",
			input:    "line1\nline2",
			expected: "line1\\nline2",
		},
		{
			name:     "tabs",
			input:    "col1\tcol2",
			expected: "col1\\tcol2",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EscapeJSON(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestValidateInputLength(t *testing.T) {
	tests := []struct {
		name  string
		input string
		min   int
		max   int
		valid bool
	}{
		{"valid length", "hello", 1, 10, true},
		{"too short", "ab", 5, 10, false},
		{"too long", "very long string", 1, 5, false},
		{"exact min", "abc", 3, 10, true},
		{"exact max", "abcde", 1, 5, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ValidateInputLength(tt.input, tt.min, tt.max)
			if result != tt.valid {
				t.Errorf("Expected %v, got %v", tt.valid, result)
			}
		})
	}
}

func TestContainsOnlyAlphanumeric(t *testing.T) {
	tests := []struct {
		name  string
		input string
		valid bool
	}{
		{"alphanumeric", "abc123", true},
		{"with space", "abc 123", false},
		{"with special", "abc@123", false},
		{"uppercase", "ABC", true},
		{"numbers only", "123456", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsOnlyAlphanumeric(tt.input)
			if result != tt.valid {
				t.Errorf("Expected %v for '%s', got %v", tt.valid, tt.input, result)
			}
		})
	}
}

func TestContainsOnlyLetters(t *testing.T) {
	tests := []struct {
		name  string
		input string
		valid bool
	}{
		{"letters only", "abcdef", true},
		{"with numbers", "abc123", false},
		{"uppercase", "ABCDEF", true},
		{"mixed case", "AbCdEf", true},
		{"with space", "abc def", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContainsOnlyLetters(tt.input)
			if result != tt.valid {
				t.Errorf("Expected %v for '%s', got %v", tt.valid, tt.input, result)
			}
		})
	}
}

func TestRemoveEmoji(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "smiley face",
			input:    "Hello 😀 World",
			expected: "Hello  World",
		},
		{
			name:     "heart",
			input:    "I ❤️ Go",
			expected: "I  Go",
		},
		{
			name:     "no emoji",
			input:    "Just text",
			expected: "Just text",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := RemoveEmoji(tt.input)
			if result != tt.expected {
				t.Errorf("Expected '%s', got '%s'", tt.expected, result)
			}
		})
	}
}

func TestSanitizeInput_DangerousTags(t *testing.T) {
	dangerousTags := []string{
		"script", "iframe", "object", "embed", "form",
		"input", "textarea", "button", "select", "link",
		"meta", "style",
	}

	for _, tag := range dangerousTags {
		t.Run(tag+" tag", func(t *testing.T) {
			input := "<" + tag + ">evil content</" + tag + ">"
			result := SanitizeInput(input)

			if strings.Contains(strings.ToLower(result), "<"+tag) {
				t.Errorf("Dangerous tag <%s> not removed from: %s", tag, result)
			}
		})
	}
}

func TestSanitizePrompt_Comprehensive(t *testing.T) {
	// Test with mixed malicious content
	input := `<script>alert('xss')</script>Hello\x00World<img src="x" onerror="alert('hack')">`
	result := SanitizePrompt(input)

	// Should not contain script tags
	if strings.Contains(strings.ToLower(result), "<script>") {
		t.Error("Script tag not removed")
	}

	// Should not contain control characters
	if strings.ContainsAny(result, "\x00") {
		t.Error("Control characters not removed")
	}

	t.Logf("Sanitized prompt: %s", result)
}
