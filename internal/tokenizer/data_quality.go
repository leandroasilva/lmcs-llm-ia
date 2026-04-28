package tokenizer

import (
	"fmt"
	"strings"
)

// SDGTDataGenerator implements Seed-Driven Growth Technique for synthetic data generation
type SDGTDataGenerator struct {
	Seeds           []string
	Generated       []string
	Templates       []string
	Transformations []func(string) string
}

// NewSDGTDataGenerator creates a new SDGT data generator
func NewSDGTDataGenerator(seeds []string) *SDGTDataGenerator {
	return &SDGTDataGenerator{
		Seeds:     seeds,
		Generated: make([]string, 0),
		Templates: []string{
			"What is %s?",
			"Explain %s in detail.",
			"How does %s work?",
			"Tell me about %s.",
			"Describe %s.",
			"Why is %s important?",
			"What are the benefits of %s?",
			"How to use %s effectively?",
		},
		Transformations: []func(string) string{
			func(s string) string { return strings.ToUpper(s) },
			func(s string) string { return strings.ToLower(s) },
			func(s string) string { return "Please " + s },
			func(s string) string { return "Can you " + s },
			func(s string) string { return s + " Please provide examples." },
			func(s string) string { return "I need help with: " + s },
			func(s string) string { return "Could you explain " + s + "?" },
		},
	}
}

// Generate generates synthetic data from seeds using SDGT
func (sdgt *SDGTDataGenerator) Generate(numSamples int) []string {
	generated := make([]string, 0, numSamples)

	// Add original seeds
	generated = append(generated, sdgt.Seeds...)

	// Generate variations until we reach target
	for len(generated) < numSamples {
		for _, seed := range sdgt.Seeds {
			if len(generated) >= numSamples {
				break
			}

			// Apply templates
			for _, template := range sdgt.Templates {
				if len(generated) >= numSamples {
					break
				}

				variation := fmt.Sprintf(template, seed)
				generated = append(generated, variation)
			}

			// Apply transformations
			for _, transform := range sdgt.Transformations {
				if len(generated) >= numSamples {
					break
				}

				variation := transform(seed)
				generated = append(generated, variation)
			}
		}
	}

	sdgt.Generated = generated
	return generated
}

// AddSeed adds a new seed
func (sdgt *SDGTDataGenerator) AddSeed(seed string) {
	sdgt.Seeds = append(sdgt.Seeds, seed)
}

// AddTemplate adds a new template
func (sdgt *SDGTDataGenerator) AddTemplate(template string) {
	sdgt.Templates = append(sdgt.Templates, template)
}

// GetGenerated returns all generated data
func (sdgt *SDGTDataGenerator) GetGenerated() []string {
	return sdgt.Generated
}

// GetSeedCount returns number of seeds
func (sdgt *SDGTDataGenerator) GetSeedCount() int {
	return len(sdgt.Seeds)
}

// GetGeneratedCount returns number of generated samples
func (sdgt *SDGTDataGenerator) GetGeneratedCount() int {
	return len(sdgt.Generated)
}

// SDGTExpansionRate calculates expansion rate
func (sdgt *SDGTDataGenerator) SDGTExpansionRate() float64 {
	if len(sdgt.Seeds) == 0 {
		return 0.0
	}
	return float64(len(sdgt.Generated)) / float64(len(sdgt.Seeds))
}

// DataCurator curates and filters dataset for quality
type DataCurator struct {
	MinLength          int
	MaxLength          int
	DiversityThreshold float64
}

// NewDataCurator creates a new data curator
func NewDataCurator() *DataCurator {
	return &DataCurator{
		MinLength:          10,
		MaxLength:          1000,
		DiversityThreshold: 0.3,
	}
}

// Curate filters and curates dataset
func (dc *DataCurator) Curate(data []string) []string {
	curated := make([]string, 0)

	for _, item := range data {
		// Filter by length
		if len(item) < dc.MinLength || len(item) > dc.MaxLength {
			continue
		}

		// Filter duplicates
		if dc.isDuplicate(item, curated) {
			continue
		}

		// Check diversity
		if !dc.isDiverse(item, curated) {
			continue
		}

		curated = append(curated, item)
	}

	return curated
}

// isDuplicate checks if item is duplicate
func (dc *DataCurator) isDuplicate(item string, curated []string) bool {
	for _, existing := range curated {
		if item == existing {
			return true
		}
	}
	return false
}

// isDiverse checks if item is diverse enough
func (dc *DataCurator) isDiverse(item string, curated []string) bool {
	if len(curated) == 0 {
		return true
	}

	// Simple diversity check based on character overlap
	minSimilarity := 1.0

	for _, existing := range curated {
		similarity := dc.calculateSimilarity(item, existing)
		if similarity < minSimilarity {
			minSimilarity = similarity
		}
	}

	return minSimilarity < dc.DiversityThreshold
}

// calculateSimilarity calculates Jaccard similarity between two strings
func (dc *DataCurator) calculateSimilarity(s1, s2 string) float64 {
	// Simple Jaccard similarity on character bigrams
	getBigrams := func(s string) map[string]bool {
		bigrams := make(map[string]bool)
		for i := 0; i < len(s)-1; i++ {
			bigram := s[i : i+2]
			bigrams[bigram] = true
		}
		return bigrams
	}

	bigrams1 := getBigrams(s1)
	bigrams2 := getBigrams(s2)

	// Intersection
	intersection := 0
	for bigram := range bigrams1 {
		if bigrams2[bigram] {
			intersection++
		}
	}

	// Union
	union := len(bigrams1) + len(bigrams2) - intersection

	if union == 0 {
		return 0.0
	}

	return float64(intersection) / float64(union)
}

// GetCurateStats returns curation statistics
func (dc *DataCurator) GetCurateStats(original []string, curated []string) map[string]interface{} {
	retentionRate := 0.0
	if len(original) > 0 {
		retentionRate = float64(len(curated)) / float64(len(original))
	}

	return map[string]interface{}{
		"original_count": len(original),
		"curated_count":  len(curated),
		"filtered_out":   len(original) - len(curated),
		"retention_rate": retentionRate,
	}
}

// VocabularyOptimizer optimizes vocabulary size vs coverage
type VocabularyOptimizer struct {
	TargetVocabSize int
	MinCoverage     float64
}

// NewVocabularyOptimizer creates a new vocab optimizer
func NewVocabularyOptimizer(targetSize int, minCoverage float64) *VocabularyOptimizer {
	return &VocabularyOptimizer{
		TargetVocabSize: targetSize,
		MinCoverage:     minCoverage,
	}
}

// Optimize optimizes vocabulary
func (vo *VocabularyOptimizer) Optimize(vocab map[string]int, corpus []string) map[string]int {
	// Calculate coverage for each token
	tokenCoverage := make(map[string]float64)

	for token := range vocab {
		coverage := vo.calculateTokenCoverage(token, corpus)
		tokenCoverage[token] = coverage
	}

	// Sort by coverage
	type tokenScore struct {
		Token    string
		Coverage float64
	}

	sorted := make([]tokenScore, 0, len(tokenCoverage))
	for token, coverage := range tokenCoverage {
		sorted = append(sorted, tokenScore{token, coverage})
	}

	// Keep top tokens that meet min coverage
	optimized := make(map[string]int)
	count := 0

	for _, ts := range sorted {
		if count >= vo.TargetVocabSize {
			break
		}
		if ts.Coverage >= vo.MinCoverage {
			optimized[ts.Token] = count
			count++
		}
	}

	return optimized
}

// calculateTokenCoverage calculates how much corpus a token covers
func (vo *VocabularyOptimizer) calculateTokenCoverage(token string, corpus []string) float64 {
	if len(corpus) == 0 {
		return 0.0
	}

	tokenCount := 0
	totalChars := 0

	for _, text := range corpus {
		totalChars += len(text)
		// Count occurrences
		for i := 0; i <= len(text)-len(token); i++ {
			if text[i:i+len(token)] == token {
				tokenCount++
			}
		}
	}

	if totalChars == 0 {
		return 0.0
	}

	return float64(tokenCount*len(token)) / float64(totalChars)
}

// GetOptimizationStats returns optimization statistics
func (vo *VocabularyOptimizer) GetOptimizationStats(original map[string]int, optimized map[string]int) map[string]interface{} {
	return map[string]interface{}{
		"original_size":  len(original),
		"optimized_size": len(optimized),
		"reduction":      len(original) - len(optimized),
		"reduction_pct":  float64(len(original)-len(optimized)) / float64(len(original)),
	}
}
