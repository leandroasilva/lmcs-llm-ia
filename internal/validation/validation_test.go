package validation

import (
	"testing"
)

func TestValidator_ValidateString(t *testing.T) {
	tests := []struct {
		name      string
		field     string
		value     string
		min       int
		max       int
		wantError bool
	}{
		{"valid string", "name", "hello", 1, 100, false},
		{"empty string", "name", "", 1, 100, true},
		{"too short", "name", "ab", 5, 100, true},
		{"too long", "name", "this is a very long string", 1, 10, true},
		{"exact min length", "name", "abcde", 5, 100, false},
		{"exact max length", "name", "abcdefghij", 1, 10, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := NewValidator()
			v.ValidateString(tt.field, tt.value, tt.min, tt.max)

			if tt.wantError && !v.HasErrors() {
				t.Error("Expected validation error, got none")
			}

			if !tt.wantError && v.HasErrors() {
				t.Errorf("Expected no error, got: %v", v.GetErrors())
			}
		})
	}
}

func TestValidator_ValidateInt(t *testing.T) {
	tests := []struct {
		name      string
		field     string
		value     int
		min       int
		max       int
		wantError bool
	}{
		{"valid int", "age", 25, 0, 150, false},
		{"below min (positive)", "age", 5, 10, 150, true},
		{"above max", "age", 200, 0, 150, true},
		{"at min", "age", 10, 10, 150, false},
		{"at max", "age", 150, 0, 150, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := NewValidator()
			v.ValidateInt(tt.field, tt.value, tt.min, tt.max)

			if tt.wantError && !v.HasErrors() {
				t.Error("Expected validation error, got none")
			}

			if !tt.wantError && v.HasErrors() {
				t.Errorf("Expected no error, got: %v", v.GetErrors())
			}
		})
	}
}

func TestValidator_ValidateFloat(t *testing.T) {
	tests := []struct {
		name      string
		field     string
		value     float64
		min       float64
		max       float64
		wantError bool
	}{
		{"valid float", "temp", 0.7, 0.0, 1.0, false},
		{"below min (positive)", "temp", 0.05, 0.1, 1.0, true},
		{"above max", "temp", 1.5, 0.0, 1.0, true},
		{"at min", "temp", 0.1, 0.1, 1.0, false},
		{"at max", "temp", 1.0, 0.0, 1.0, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := NewValidator()
			v.ValidateFloat(tt.field, tt.value, tt.min, tt.max)

			if tt.wantError && !v.HasErrors() {
				t.Error("Expected validation error, got none")
			}

			if !tt.wantError && v.HasErrors() {
				t.Errorf("Expected no error, got: %v", v.GetErrors())
			}
		})
	}
}

func TestValidator_ValidateEmail(t *testing.T) {
	tests := []struct {
		name      string
		email     string
		wantError bool
	}{
		{"valid email", "user@example.com", false},
		{"missing @", "userexample.com", true},
		{"missing domain", "user@", true},
		{"missing user", "@example.com", true},
		{"invalid chars", "user@exam ple.com", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := NewValidator()
			v.ValidateEmail("email", tt.email)

			if tt.wantError && !v.HasErrors() {
				t.Error("Expected validation error, got none")
			}

			if !tt.wantError && v.HasErrors() {
				t.Errorf("Expected no error, got: %v", v.GetErrors())
			}
		})
	}
}

func TestValidator_ValidatePattern(t *testing.T) {
	v := NewValidator()
	v.ValidatePattern("phone", "123-456-7890", `^\d{3}-\d{3}-\d{4}$`)

	if v.HasErrors() {
		t.Errorf("Expected valid pattern, got error: %v", v.GetErrors())
	}

	v2 := NewValidator()
	v2.ValidatePattern("phone", "1234567890", `^\d{3}-\d{3}-\d{4}$`)

	if !v2.HasErrors() {
		t.Error("Expected pattern validation error, got none")
	}
}

func TestValidator_ValidateInList(t *testing.T) {
	allowed := []string{"red", "green", "blue"}

	v := NewValidator()
	v.ValidateInList("color", "red", allowed)

	if v.HasErrors() {
		t.Errorf("Expected 'red' to be valid, got: %v", v.GetErrors())
	}

	v2 := NewValidator()
	v2.ValidateInList("color", "yellow", allowed)

	if !v2.HasErrors() {
		t.Error("Expected 'yellow' to be invalid, got no error")
	}
}

func TestValidator_MultipleErrors(t *testing.T) {
	v := NewValidator()
	v.ValidateString("name", "", 1, 100)
	v.ValidateInt("age", 5, 10, 150) // below min
	v.ValidateFloat("score", 10.0, 0.0, 5.0)

	if !v.HasErrors() {
		t.Error("Expected multiple errors, got none")
	}

	errors := v.GetErrors()
	if len(errors) != 3 {
		t.Errorf("Expected 3 errors, got %d", len(errors))
	}

	t.Logf("Errors: %v", errors)
}

func TestTrainingConfigValidator(t *testing.T) {
	// TrainingConfigValidator requires interface implementation
	// This test verifies the validator can be instantiated
	validator := &TrainingConfigValidator{}

	if validator == nil {
		t.Error("Failed to create TrainingConfigValidator")
	}

	t.Log("TrainingConfigValidator exists and can be instantiated")
}

func TestGenerationRequestValidator(t *testing.T) {
	validator := &GenerationRequestValidator{}

	// Valid request
	v1 := validator.Validate("hello world", 100, 0.7)
	if v1.HasErrors() {
		t.Errorf("Expected valid request, got errors: %v", v1.GetErrors())
	}

	// Invalid: empty prompt
	v2 := validator.Validate("", 100, 0.7)
	if !v2.HasErrors() {
		t.Error("Expected error for empty prompt")
	}

	// Invalid: maxTokens too high
	v3 := validator.Validate("hello", 2000, 0.7)
	if !v3.HasErrors() {
		t.Error("Expected error for maxTokens > 1000")
	}

	// Invalid: temperature out of range
	v4 := validator.Validate("hello", 100, 3.0)
	if !v4.HasErrors() {
		t.Error("Expected error for temperature > 2.0")
	}

	t.Log("GenerationRequestValidator tests passed")
}

func TestValidator_GetErrors(t *testing.T) {
	v := NewValidator()

	// Should return empty map initially
	if len(v.GetErrors()) != 0 {
		t.Error("New validator should have no errors")
	}

	// Add an error
	v.ValidateString("field", "", 1, 100)

	// Should return map with error
	errors := v.GetErrors()
	if len(errors) != 1 {
		t.Errorf("Expected 1 error, got %d", len(errors))
	}

	if _, ok := errors["field"]; !ok {
		t.Error("Expected error for 'field'")
	}
}
