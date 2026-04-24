package main

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"sync"
)

// LmcsLLM armazena a estrutura da rede
type LmcsLLM struct {
	Weights  [][]float64
	Chars    []rune
	CharToID map[rune]int
	IDToChar map[int]rune
	Size     int
}

// --- 1. PERSISTÊNCIA (SALVAR/CARREGAR) ---
func (m *LmcsLLM) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return gob.NewEncoder(file).Encode(m)
}

func LoadModel(path string) (*LmcsLLM, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var m LmcsLLM
	err = gob.NewDecoder(file).Decode(&m)
	return &m, err
}

// --- 2. INICIALIZAÇÃO ---

func NewLmcsLLM(text string) *LmcsLLM {
	uniqueMap := make(map[rune]bool)
	for _, r := range text {
		uniqueMap[r] = true
	}

	var chars []rune
	for r := range uniqueMap {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })

	size := len(chars)
	cToID := make(map[rune]int)
	idToC := make(map[int]rune)
	weights := make([][]float64, size)

	for i, r := range chars {
		cToID[r] = i
		idToC[i] = r
		weights[i] = make([]float64, size)
		for j := range weights[i] {
			weights[i][j] = (rand.Float64() - 0.5) * 0.1
		}
	}

	return &LmcsLLM{weights, chars, cToID, idToC, size}
}

// --- 3. TREINAMENTO MULTI-CORE ---
func (m *LmcsLLM) Train(text string, epochs int, lr float64) {
	runes := []rune(text)
	n := len(runes)

	for e := 1; e <= epochs; e++ {
		var wg sync.WaitGroup
		loss := 0.0

		// Usamos um mutex para proteger a atualização dos pesos entre threads
		var mu sync.Mutex

		// Dividimos o texto em pedaços para processar em paralelo
		for i := 0; i < n-1; i++ {
			wg.Add(1)
			go func(idx int) {
				defer wg.Done()

				curr := m.CharToID[runes[idx]]
				next := m.CharToID[runes[idx+1]]

				// Forward
				logits := m.Weights[curr]
				probs := softmax(logits)

				mu.Lock()
				// Acumular Loss
				loss += -math.Log(probs[next] + 1e-9)

				// Backward (Ajuste de pesos)
				for j := 0; j < m.Size; j++ {
					target := 0.0
					if j == next {
						target = 1.0
					}
					m.Weights[curr][j] -= lr * (probs[j] - target)
				}
				mu.Unlock()
			}(i)
		}
		wg.Wait()
		fmt.Printf("Época %d/%d concluída. Erro: %.4f\n", e, epochs, loss/float64(n))
	}
}

// --- 4. FUNÇÕES AUXILIARES ---
func softmax(x []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range x {
		if v > max {
			max = v
		}
	}
	res := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		res[i] = math.Exp(v - max)
		sum += res[i]
	}
	for i := range res {
		res[i] /= sum
	}
	return res
}

func (m *LmcsLLM) Generate(seed rune, length int) string {
	res := string(seed)
	curr := seed
	for i := 0; i < length; i++ {
		id, ok := m.CharToID[curr]
		if !ok {
			break
		}

		maxIdx := 0
		maxVal := -math.MaxFloat64
		for idx, v := range m.Weights[id] {
			if v > maxVal {
				maxVal = v
				maxIdx = idx
			}
		}
		curr = m.IDToChar[maxIdx]
		res += string(curr)
	}
	return res
}

// --- 5. EXECUÇÃO PRINCIPAL ---
func main() {
	modelPath := "modelo_treinado.bin"
	inputFile := "livro.txt"

	// Carregar texto do livro
	data, _ := os.ReadFile(inputFile)
	content := string(data)
	if content == "" {
		content = "o rato roeu a roupa do rei de roma." // Fallback
	}

	var model *LmcsLLM
	if _, err := os.Stat(modelPath); err == nil {
		fmt.Println(">> Carregando modelo existente...")
		model, _ = LoadModel(modelPath)
	} else {
		fmt.Println(">> Inicializando novo modelo...")
		model = NewLmcsLLM(content)
	}

	// Treinar e salvar
	model.Train(content, 50, 0.1)
	model.Save(modelPath)
	fmt.Println(">> Modelo salvo com sucesso!")

	// Subir API
	http.HandleFunc("/ask", func(w http.ResponseWriter, r *http.Request) {
		char := r.URL.Query().Get("q")
		if char == "" {
			char = "o"
		}
		resp := model.Generate(rune(char[0]), 100)
		fmt.Fprintf(w, "Resultado: %s", resp)
	})

	fmt.Println("API rodando em http://localhost:8080/ask?q=o")
	http.ListenAndServe(":8080", nil)
}
