package tokenizer

import (
	"sort"
	"strings"
)

// BPETokenizer implementa Byte Pair Encoding
type BPETokenizer struct {
	VocabSize     int
	Merges        []Merge
	Vocab         map[string]int
	TokenToWord   map[int]string
	SpecialTokens map[string]int
}

// Merge representa uma operação de merge BPE
type Merge struct {
	Pair   [2]string
	Result string
}

// NewBPETokenizer cria um novo tokenizador BPE
func NewBPETokenizer() *BPETokenizer {
	return &BPETokenizer{
		Vocab:       make(map[string]int),
		TokenToWord: make(map[int]string),
		SpecialTokens: map[string]int{
			"<PAD>": 0,
			"<UNK>": 1,
			"<BOS>": 2,
			"<EOS>": 3,
		},
	}
}

// Train treina o tokenizador BPE no corpus
func (t *BPETokenizer) Train(corpus string, vocabSize int) {
	t.VocabSize = vocabSize

	// Pré-processar: adicionar marcador de fim de palavra
	words := strings.Fields(corpus)
	wordFreq := make(map[string]int)
	for _, word := range words {
		word = word + "</w>"
		wordFreq[word]++
	}

	// Inicializar vocabulário com caracteres
	vocab := make(map[string]int)
	nextID := len(t.SpecialTokens)

	for token, id := range t.SpecialTokens {
		vocab[token] = id
	}

	// Extrair todos os caracteres únicos
	for word := range wordFreq {
		for _, char := range strings.Split(word, "") {
			if char == "" {
				continue
			}
			if _, exists := vocab[char]; !exists {
				vocab[char] = nextID
				nextID++
			}
		}
	}

	t.Vocab = vocab

	// Aprender merges BPE
	merges := make([]Merge, 0)

	for len(vocab) < vocabSize {
		// Contar pares adjacentes
		pairCounts := make(map[[2]string]int)

		for word, freq := range wordFreq {
			symbols := strings.Split(word, "")
			for i := 0; i < len(symbols)-1; i++ {
				if symbols[i] == "" || symbols[i+1] == "" {
					continue
				}
				pair := [2]string{symbols[i], symbols[i+1]}
				pairCounts[pair] += freq
			}
		}

		if len(pairCounts) == 0 {
			break
		}

		// Encontrar par mais frequente
		var bestPair [2]string
		bestCount := 0
		for pair, count := range pairCounts {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		if bestCount == 0 {
			break
		}

		// Criar novo token
		newToken := bestPair[0] + bestPair[1]
		vocab[newToken] = nextID
		nextID++

		// Adicionar merge
		merge := Merge{
			Pair:   bestPair,
			Result: newToken,
		}
		merges = append(merges, merge)

		// Aplicar merge em todas as palavras
		newWordFreq := make(map[string]int)
		for word, freq := range wordFreq {
			symbols := strings.Split(word, "")
			newSymbols := make([]string, 0)

			i := 0
			for i < len(symbols) {
				if i < len(symbols)-1 && symbols[i] == bestPair[0] && symbols[i+1] == bestPair[1] {
					newSymbols = append(newSymbols, newToken)
					i += 2
				} else {
					newSymbols = append(newSymbols, symbols[i])
					i++
				}
			}

			newWord := strings.Join(newSymbols, "")
			newWordFreq[newWord] += freq
		}
		wordFreq = newWordFreq
	}

	t.Merges = merges
	t.Vocab = vocab

	// Criar mapa inverso
	for token, id := range vocab {
		t.TokenToWord[id] = token
	}
}

// Tokenize tokeniza um texto usando BPE
func (t *BPETokenizer) Tokenize(text string) []int {
	// Pré-processar
	text = strings.ToLower(text)
	words := strings.Fields(text)

	tokens := make([]int, 0)
	tokens = append(tokens, t.SpecialTokens["<BOS>"])

	for _, word := range words {
		word = word + "</w>"
		wordTokens := t.tokenizeWord(word)
		tokens = append(tokens, wordTokens...)
	}

	tokens = append(tokens, t.SpecialTokens["<EOS>"])
	return tokens
}

// tokenizeWord tokeniza uma única palavra
func (t *BPETokenizer) tokenizeWord(word string) []int {
	// Dividir em caracteres
	symbols := strings.Split(word, "")

	// Aplicar merges
	for _, merge := range t.Merges {
		newSymbols := make([]string, 0)
		i := 0
		for i < len(symbols) {
			if i < len(symbols)-1 && symbols[i] == merge.Pair[0] && symbols[i+1] == merge.Pair[1] {
				newSymbols = append(newSymbols, merge.Result)
				i += 2
			} else {
				newSymbols = append(newSymbols, symbols[i])
				i++
			}
		}
		symbols = newSymbols
	}

	// Converter para IDs
	tokens := make([]int, 0)
	for _, symbol := range symbols {
		if symbol == "" {
			continue
		}
		if id, exists := t.Vocab[symbol]; exists {
			tokens = append(tokens, id)
		} else {
			// Token desconhecido
			tokens = append(tokens, t.SpecialTokens["<UNK>"])
		}
	}

	return tokens
}

// Decode converte tokens de volta para texto
func (t *BPETokenizer) Decode(tokens []int) string {
	words := make([]string, 0)
	currentWord := ""

	for _, token := range tokens {
		if token == t.SpecialTokens["<BOS>"] || token == t.SpecialTokens["<PAD>"] {
			continue
		}
		if token == t.SpecialTokens["<EOS>"] {
			if currentWord != "" {
				words = append(words, currentWord)
				currentWord = ""
			}
			continue
		}
		if token == t.SpecialTokens["<UNK>"] {
			currentWord += "[UNK]"
			continue
		}

		if symbol, exists := t.TokenToWord[token]; exists {
			currentWord += symbol
		}
	}

	if currentWord != "" {
		words = append(words, currentWord)
	}

	// Juntar palavras e remover marcador </w>
	text := strings.Join(words, " ")
	text = strings.ReplaceAll(text, "</w>", "")
	return text
}

// GetVocabSize retorna o tamanho do vocabulário
func (t *BPETokenizer) GetVocabSize() int {
	return len(t.Vocab)
}

// GetVocab retorna o vocabulário
func (t *BPETokenizer) GetVocab() map[string]int {
	return t.Vocab
}

// SaveMerges salva os merges em formato texto
func (t *BPETokenizer) SaveMerges() string {
	lines := make([]string, 0)
	for _, merge := range t.Merges {
		line := merge.Pair[0] + " " + merge.Pair[1]
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

// LoadMerges carrega merges de formato texto
func (t *BPETokenizer) LoadMerges(mergesText string) {
	lines := strings.Split(mergesText, "\n")
	t.Merges = make([]Merge, 0)

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, " ", 2)
		if len(parts) != 2 {
			continue
		}

		merge := Merge{
			Pair:   [2]string{parts[0], parts[1]},
			Result: parts[0] + parts[1],
		}
		t.Merges = append(t.Merges, merge)
	}
}

// WordPieceTokenizer implementa tokenização WordPiece simplificada
type WordPieceTokenizer struct {
	Vocab       map[string]int
	TokenToWord map[int]string
	MaxWordLen  int
}

// NewWordPieceTokenizer cria um novo tokenizador WordPiece
func NewWordPieceTokenizer() *WordPieceTokenizer {
	return &WordPieceTokenizer{
		Vocab:       make(map[string]int),
		TokenToWord: make(map[int]string),
		MaxWordLen:  100,
	}
}

// Train treina o tokenizador WordPiece
func (wp *WordPieceTokenizer) Train(corpus string, vocabSize int) {
	words := strings.Fields(corpus)
	wordFreq := make(map[string]int)
	for _, word := range words {
		wordFreq[strings.ToLower(word)]++
	}

	// Inicializar com caracteres e subwords comuns
	wp.Vocab = map[string]int{
		"[PAD]": 0,
		"[UNK]": 1,
		"[CLS]": 2,
		"[SEP]": 3,
	}
	nextID := 4

	// Adicionar subwords frequentes
	subwordFreq := make(map[string]int)
	for word, freq := range wordFreq {
		// Adicionar palavra completa
		subwordFreq[word] += freq

		// Adicionar prefixos
		for i := 1; i < len(word) && i < 10; i++ {
			prefix := word[:i] + "##"
			subwordFreq[prefix] += freq
		}
	}

	// Ordenar por frequência
	type subwordCount struct {
		subword string
		count   int
	}
	subwords := make([]subwordCount, 0)
	for sw, count := range subwordFreq {
		subwords = append(subwords, subwordCount{sw, count})
	}
	sort.Slice(subwords, func(i, j int) bool {
		return subwords[i].count > subwords[j].count
	})

	// Adicionar ao vocabulário até atingir tamanho
	for _, sw := range subwords {
		if len(wp.Vocab) >= vocabSize {
			break
		}
		if _, exists := wp.Vocab[sw.subword]; !exists {
			wp.Vocab[sw.subword] = nextID
			wp.TokenToWord[nextID] = sw.subword
			nextID++
		}
	}
}

// Tokenize tokeniza texto com WordPiece
func (wp *WordPieceTokenizer) Tokenize(text string) []int {
	words := strings.Fields(strings.ToLower(text))
	tokens := []int{wp.Vocab["[CLS]"]}

	for _, word := range words {
		wordTokens := wp.tokenizeWord(word)
		tokens = append(tokens, wordTokens...)
	}

	tokens = append(tokens, wp.Vocab["[SEP]"])
	return tokens
}

// tokenizeWord tokeniza uma palavra com WordPiece
func (wp *WordPieceTokenizer) tokenizeWord(word string) []int {
	// Tentar greedy longest-match-first
	tokens := make([]int, 0)
	start := 0

	for start < len(word) {
		found := false

		// Tentar subword mais longa
		for end := min(start+10, len(word)); end > start; end-- {
			subword := word[start:end]
			if start > 0 {
				subword = "##" + subword
			}

			if id, exists := wp.Vocab[subword]; exists {
				tokens = append(tokens, id)
				start = end
				found = true
				break
			}
		}

		if !found {
			// Caractere desconhecido
			tokens = append(tokens, wp.Vocab["[UNK]"])
			start++
		}
	}

	return tokens
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
