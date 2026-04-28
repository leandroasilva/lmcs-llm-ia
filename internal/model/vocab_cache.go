package model

import (
	"encoding/gob"
	"os"
	"path/filepath"
	"time"
)

// VocabCache armazena vocabulário em cache serializado
type VocabCache struct {
	Vocab       []string          `json:"vocab"`
	WordToID    map[string]int    `json:"word_to_id"`
	IDToWord    map[int]string    `json:"id_to_word"`
	TextHash    uint64            `json:"text_hash"`      // Hash do corpus para invalidação
	MaxVocab    int               `json:"max_vocab"`      // Tamanho máximo usado
	CreatedAt   time.Time         `json:"created_at"`     // Data de criação
	CacheFile   string            `json:"cache_file"`     // Path do arquivo de cache
}

// HashString calcula hash simples de string para detecção de mudanças
func HashString(s string) uint64 {
	var hash uint64 = 5381
	for i := 0; i < len(s); i++ {
		hash = ((hash << 5) + hash) + uint64(s[i])
	}
	return hash
}

// GetVocabCachePath retorna o path do arquivo de cache
func GetVocabCachePath(modelPath string) string {
	dir := filepath.Dir(modelPath)
	return filepath.Join(dir, "vocab-cache.gob")
}

// LoadVocabCache carrega vocabulário do cache
func LoadVocabCache(cachePath string) (*VocabCache, error) {
	file, err := os.Open(cachePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	cache := &VocabCache{
		WordToID: make(map[string]int),
		IDToWord: make(map[int]string),
	}

	if err := decoder.Decode(cache); err != nil {
		return nil, err
	}

	return cache, nil
}

// SaveVocabCache salva vocabulário em cache
func SaveVocabCache(cache *VocabCache) error {
	// Criar diretório se não existir
	dir := filepath.Dir(cache.CacheFile)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	file, err := os.Create(cache.CacheFile)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(cache)
}

// BuildVocabWithCache constrói vocabulário com cache
func BuildVocabWithCache(text string, maxVocab int, cachePath string) ([]string, map[string]int, map[int]string, error) {
	// Calcular hash do corpus
	textHash := HashString(text)

	// Tentar carregar do cache
	if cache, err := LoadVocabCache(cachePath); err == nil {
		// Verificar se cache é válido
		if cache.TextHash == textHash && cache.MaxVocab == maxVocab {
			// Cache hit!
			return cache.Vocab, cache.WordToID, cache.IDToWord, nil
		}
	}

	// Cache miss ou inválido - construir vocabulário
	vocab, wordToID, idToWord := BuildVocabTransformer(text, maxVocab)

	// Salvar em cache
	cache := &VocabCache{
		Vocab:     vocab,
		WordToID:  wordToID,
		IDToWord:  idToWord,
		TextHash:  textHash,
		MaxVocab:  maxVocab,
		CreatedAt: time.Now(),
		CacheFile: cachePath,
	}

	if err := SaveVocabCache(cache); err != nil {
		// Log error but don't fail - cache is optional
		// logger.Warn("Failed to save vocab cache", "error", err)
	}

	return vocab, wordToID, idToWord, nil
}

// InvalidateVocabCache invalida o cache de vocabulário
func InvalidateVocabCache(cachePath string) error {
	return os.Remove(cachePath)
}

// GetVocabCacheInfo retorna informações sobre o cache
func GetVocabCacheInfo(cachePath string) (map[string]interface{}, error) {
	info := make(map[string]interface{})

	cache, err := LoadVocabCache(cachePath)
	if err != nil {
		info["exists"] = false
		return info, nil
	}

	info["exists"] = true
	info["vocab_size"] = len(cache.Vocab)
	info["max_vocab"] = cache.MaxVocab
	info["created_at"] = cache.CreatedAt
	info["cache_file"] = cache.CacheFile
	info["text_hash"] = cache.TextHash

	return info, nil
}

// IsVocabCacheValid verifica se o cache é válido para o corpus atual
func IsVocabCacheValid(text string, maxVocab int, cachePath string) bool {
	cache, err := LoadVocabCache(cachePath)
	if err != nil {
		return false
	}

	textHash := HashString(text)
	return cache.TextHash == textHash && cache.MaxVocab == maxVocab
}
