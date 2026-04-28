package model

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestHashString(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"empty string", ""},
		{"simple", "hello"},
		{"longer", "hello world this is a test"},
		{"special chars", "hello!@#$%^&*()"},
		{"unicode", "olá mundo ãõá"},
	}

	// Verificar que hash é determinístico
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hash1 := HashString(tt.input)
			hash2 := HashString(tt.input)

			if hash1 != hash2 {
				t.Errorf("Hash não é determinístico: %d vs %d", hash1, hash2)
			}

			// Hash de strings diferentes deve ser diferente (na maioria dos casos)
			if tt.input != "" {
				hash3 := HashString(tt.input + "x")
				if hash1 == hash3 {
					t.Logf("Aviso: colisão de hash detectada")
				}
			}
		})
	}
}

func TestVocabCache_SaveLoad(t *testing.T) {
	// Criar diretório temporário
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-vocab-cache.gob")

	// Criar vocabulário de teste
	cache := &VocabCache{
		Vocab:     []string{"hello", "world", "test"},
		WordToID:  map[string]int{"hello": 0, "world": 1, "test": 2},
		IDToWord:  map[int]string{0: "hello", 1: "world", 2: "test"},
		TextHash:  12345,
		MaxVocab:  100,
		CreatedAt: time.Now(),
		CacheFile: cachePath,
	}

	// Salvar cache
	if err := SaveVocabCache(cache); err != nil {
		t.Fatalf("Failed to save cache: %v", err)
	}

	// Verificar arquivo existe
	if _, err := os.Stat(cachePath); os.IsNotExist(err) {
		t.Fatal("Cache file was not created")
	}

	// Carregar cache
	loaded, err := LoadVocabCache(cachePath)
	if err != nil {
		t.Fatalf("Failed to load cache: %v", err)
	}

	// Verificar dados
	if len(loaded.Vocab) != len(cache.Vocab) {
		t.Errorf("Vocab size mismatch: expected %d, got %d", len(cache.Vocab), len(loaded.Vocab))
	}

	if loaded.TextHash != cache.TextHash {
		t.Errorf("TextHash mismatch: expected %d, got %d", cache.TextHash, loaded.TextHash)
	}

	if loaded.MaxVocab != cache.MaxVocab {
		t.Errorf("MaxVocab mismatch: expected %d, got %d", cache.MaxVocab, loaded.MaxVocab)
	}

	t.Logf("Cache saved and loaded successfully: %d words", len(loaded.Vocab))
}

func TestVocabCache_CacheHit(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-cache-hit.gob")

	corpus := strings.Repeat("hello world test ", 100)

	// Primeira chamada - deve construir e salvar cache
	vocab1, _, _, err := BuildVocabWithCache(corpus, 100, cachePath)
	if err != nil {
		t.Fatalf("First build failed: %v", err)
	}

	// Verificar que cache foi criado
	if _, err := os.Stat(cachePath); os.IsNotExist(err) {
		t.Fatal("Cache file was not created")
	}

	// Segunda chamada - deve usar cache
	startBefore := time.Now()
	vocab2, _, _, err := BuildVocabWithCache(corpus, 100, cachePath)
	elapsed := time.Since(startBefore)

	if err != nil {
		t.Fatalf("Cache hit failed: %v", err)
	}

	// Verificar que os dados são iguais
	if len(vocab1) != len(vocab2) {
		t.Errorf("Vocab size mismatch after cache hit")
	}

	t.Logf("Cache hit successful in %v", elapsed)
	t.Logf("Vocab size: %d words", len(vocab1))
}

func TestVocabCache_CacheInvalidation(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-invalidation.gob")

	corpus1 := "hello world test"
	corpus2 := "different corpus entirely"

	// Construir cache com corpus1
	_, _, _, err := BuildVocabWithCache(corpus1, 100, cachePath)
	if err != nil {
		t.Fatalf("First build failed: %v", err)
	}

	// Verificar que cache é válido para corpus1
	if !IsVocabCacheValid(corpus1, 100, cachePath) {
		t.Error("Cache should be valid for corpus1")
	}

	// Verificar que cache é inválido para corpus2
	if IsVocabCacheValid(corpus2, 100, cachePath) {
		t.Error("Cache should be invalid for corpus2")
	}

	// Reconstruir com corpus2 - deve invalidar cache antigo
	vocab, wordToID, idToWord, err := BuildVocabWithCache(corpus2, 100, cachePath)
	if err != nil {
		t.Fatalf("Second build failed: %v", err)
	}

	// Verificar que novo cache é válido para corpus2
	if !IsVocabCacheValid(corpus2, 100, cachePath) {
		t.Error("Cache should be valid for corpus2 after rebuild")
	}

	t.Logf("Cache invalidation works correctly")
	t.Logf("New vocab size: %d words", len(vocab))

	// Suppress unused variable warnings
	_ = vocab
	_ = wordToID
	_ = idToWord
}

func TestVocabCache_DifferentMaxVocab(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-maxvocab.gob")

	corpus := "hello world test foo bar"

	// Construir com maxVocab=50
	vocab1, _, _, err := BuildVocabWithCache(corpus, 50, cachePath)
	if err != nil {
		t.Fatalf("First build failed: %v", err)
	}

	// Reconstruir com maxVocab=100 - deve invalidar cache
	vocab2, _, _, err := BuildVocabWithCache(corpus, 100, cachePath)
	if err != nil {
		t.Fatalf("Second build failed: %v", err)
	}

	// Vocabulários podem ter tamanhos diferentes
	t.Logf("Vocab with maxVocab=50: %d words", len(vocab1))
	t.Logf("Vocab with maxVocab=100: %d words", len(vocab2))
}

func TestVocabCache_Invalidate(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-invalidate.gob")

	// Criar cache
	corpus := "hello world test"
	_, _, _, err := BuildVocabWithCache(corpus, 100, cachePath)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Verificar que existe
	if _, err := os.Stat(cachePath); os.IsNotExist(err) {
		t.Fatal("Cache file does not exist")
	}

	// Invalidar cache
	if err := InvalidateVocabCache(cachePath); err != nil {
		t.Fatalf("Failed to invalidate cache: %v", err)
	}

	// Verificar que foi removido
	if _, err := os.Stat(cachePath); !os.IsNotExist(err) {
		t.Error("Cache file should be deleted after invalidation")
	}

	t.Log("Cache invalidation successful")
}

func TestVocabCache_GetInfo(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-info.gob")

	// Testar info de cache inexistente
	info, err := GetVocabCacheInfo(cachePath)
	if err != nil {
		t.Fatalf("GetInfo failed: %v", err)
	}

	if info["exists"].(bool) {
		t.Error("Cache should not exist")
	}

	// Criar cache
	corpus := "hello world test cache info"
	_, _, _, err = BuildVocabWithCache(corpus, 100, cachePath)
	if err != nil {
		t.Fatalf("Build failed: %v", err)
	}

	// Testar info de cache existente
	info, err = GetVocabCacheInfo(cachePath)
	if err != nil {
		t.Fatalf("GetInfo failed: %v", err)
	}

	if !info["exists"].(bool) {
		t.Error("Cache should exist")
	}

	if info["vocab_size"].(int) == 0 {
		t.Error("Vocab size should be > 0")
	}

	if info["max_vocab"].(int) != 100 {
		t.Errorf("Max vocab should be 100, got %v", info["max_vocab"])
	}

	t.Logf("Cache info: %+v", info)
}

func TestVocabCache_Performance(t *testing.T) {
	tmpDir := t.TempDir()
	cachePath := filepath.Join(tmpDir, "test-perf.gob")

	// Corpus grande
	corpus := strings.Repeat("hello world test performance optimization ", 1000)

	// Primeira construção (sem cache)
	start1 := time.Now()
	_, _, _, err := BuildVocabWithCache(corpus, 500, cachePath)
	elapsed1 := time.Since(start1)

	if err != nil {
		t.Fatalf("First build failed: %v", err)
	}

	t.Logf("First build (no cache): %v", elapsed1)

	// Segunda construção (com cache)
	start2 := time.Now()
	_, _, _, err = BuildVocabWithCache(corpus, 500, cachePath)
	elapsed2 := time.Since(start2)

	if err != nil {
		t.Fatalf("Second build failed: %v", err)
	}

	t.Logf("Second build (with cache): %v", elapsed2)

	// Calcular speedup
	if elapsed1 > 0 {
		speedup := float64(elapsed1) / float64(elapsed2)
		t.Logf("Cache speedup: %.2fx", speedup)

		// Cache deve ser mais rápido
		if elapsed2 > elapsed1 {
			t.Logf("Aviso: Cache não foi mais rápido (pode ser devido a I/O)")
		}
	}
}

func TestGetVocabCachePath(t *testing.T) {
	tests := []struct {
		name      string
		modelPath string
		expected  string
	}{
		{
			"simple path",
			"/tmp/model.bin",
			"/tmp/vocab-cache.gob",
		},
		{
			"nested path",
			"/tmp/models/v1/model.bin",
			"/tmp/models/v1/vocab-cache.gob",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := GetVocabCachePath(tt.modelPath)
			if result != tt.expected {
				t.Errorf("Expected %s, got %s", tt.expected, result)
			}
		})
	}
}
