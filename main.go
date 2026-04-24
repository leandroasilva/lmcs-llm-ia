package main

import (
	"encoding/gob"
	"encoding/json"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Config armazena configurações do modelo
type Config struct {
	Epochs       int
	LearningRate float64
	BatchSize    int
	Temperature  float64
	Port         string
	ModelPath    string
	InputFile    string
}

// LmcsLLM armazena a estrutura da rede
type LmcsLLM struct {
	Weights  [][]float64 `json:"-"`
	Chars    []rune
	CharToID map[rune]int
	IDToChar map[int]rune
	Size     int
	mu       sync.RWMutex // Mutex para proteção de leitura/escrita
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
	log.Println("Inicializando novo modelo...")

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

	// Semente para reproducibilidade
	rand.Seed(time.Now().UnixNano())

	for i, r := range chars {
		cToID[r] = i
		idToC[i] = r
		weights[i] = make([]float64, size)
		for j := range weights[i] {
			// Inicialização Xavier simplificada
			weights[i][j] = (rand.Float64() - 0.5) * 2.0 / float64(size)
		}
	}

	log.Printf("Modelo criado com %d caracteres únicos\n", size)
	return &LmcsLLM{weights, chars, cToID, idToC, size, sync.RWMutex{}}
}

// --- 3. TREINAMENTO COM MINI-BATCHES ---
func (m *LmcsLLM) Train(text string, epochs int, lr float64, batchSize int) {
	runes := []rune(text)
	n := len(runes)

	if n < 2 {
		log.Println("Texto muito curto para treinamento")
		return
	}

	log.Printf("Iniciando treinamento: %d épocas, lr=%.4f, batch=%d\n", epochs, lr, batchSize)

	for e := 1; e <= epochs; e++ {
		startTime := time.Now()
		totalLoss := 0.0
		samples := 0

		// Processar em mini-batches
		for batchStart := 0; batchStart < n-1; batchStart += batchSize {
			batchEnd := batchStart + batchSize
			if batchEnd > n-1 {
				batchEnd = n - 1
			}

			// Acumular gradientes para o batch
			gradients := make([][]float64, m.Size)
			for i := range gradients {
				gradients[i] = make([]float64, m.Size)
			}

			var wg sync.WaitGroup
			var mu sync.Mutex
			batchLoss := 0.0
			batchSamples := 0

			// Processar cada par no batch em paralelo
			for i := batchStart; i < batchEnd; i++ {
				wg.Add(1)
				go func(idx int) {
					defer wg.Done()

					curr, ok1 := m.CharToID[runes[idx]]
					next, ok2 := m.CharToID[runes[idx+1]]
					if !ok1 || !ok2 {
						return
					}

					// Forward
					m.mu.RLock()
					logits := make([]float64, m.Size)
					copy(logits, m.Weights[curr])
					m.mu.RUnlock()

					probs := softmax(logits)

					// Calcular loss
					loss := -math.Log(probs[next] + 1e-9)

					// Calcular gradientes
					localGrad := make([]float64, m.Size)
					for j := 0; j < m.Size; j++ {
						target := 0.0
						if j == next {
							target = 1.0
						}
						localGrad[j] = probs[j] - target
					}

					// Acumular gradientes e loss de forma thread-safe
					mu.Lock()
					for j := 0; j < m.Size; j++ {
						gradients[curr][j] += localGrad[j]
					}
					batchLoss += loss
					batchSamples++
					mu.Unlock()
				}(i)
			}
			wg.Wait()

			// Atualizar pesos com gradientes acumulados (fora do loop paralelo)
			if batchSamples > 0 {
				m.mu.Lock()
				for i := range gradients {
					for j := range gradients[i] {
						if gradients[i][j] != 0 {
							m.Weights[i][j] -= lr * gradients[i][j] / float64(batchSamples)
						}
					}
				}
				m.mu.Unlock()
			}

			totalLoss += batchLoss
			samples += batchSamples
		}

		elapsed := time.Since(startTime)
		avgLoss := totalLoss / float64(samples)
		log.Printf("Época %d/%d - Loss: %.4f | Tempo: %v\n", e, epochs, avgLoss, elapsed)
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

func (m *LmcsLLM) Generate(seed rune, length int, temperature float64) string {
	if temperature <= 0 {
		temperature = 0.8 // Valor padrão
	}

	var result strings.Builder
	result.WriteRune(seed)
	curr := seed

	for i := 0; i < length; i++ {
		id, ok := m.CharToID[curr]
		if !ok {
			break
		}

		m.mu.RLock()
		logits := make([]float64, m.Size)
		copy(logits, m.Weights[id])
		m.mu.RUnlock()

		// Aplicar temperatura
		if temperature != 1.0 {
			for j := range logits {
				logits[j] /= temperature
			}
		}

		probs := softmax(logits)

		// Amostragem categórica
		nextIdx := sample(probs)
		curr = m.IDToChar[nextIdx]
		result.WriteRune(curr)
	}
	return result.String()
}

// sample amostra um índice baseado nas probabilidades
func sample(probs []float64) int {
	r := rand.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// --- 5. HANDLERS DA API ---
type GenerateRequest struct {
	Seed        string  `json:"seed"`
	Length      int     `json:"length"`
	Temperature float64 `json:"temperature"`
}

type GenerateResponse struct {
	Success bool   `json:"success"`
	Result  string `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
}

func handleAsk(model *LmcsLLM) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		var req GenerateRequest

		// Tentar parse JSON primeiro
		if r.Method == http.MethodPost {
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				w.WriteHeader(http.StatusBadRequest)
				json.NewEncoder(w).Encode(GenerateResponse{
					Success: false,
					Error:   "JSON inválido: " + err.Error(),
				})
				return
			}
		} else {
			// Fallback para query params (GET)
			req.Seed = r.URL.Query().Get("q")
			req.Length = 100
			req.Temperature = 0.8
		}

		// Validações
		if req.Seed == "" {
			req.Seed = "o"
		}
		if req.Length <= 0 || req.Length > 1000 {
			req.Length = 100
		}
		if req.Temperature <= 0 || req.Temperature > 2.0 {
			req.Temperature = 0.8
		}

		seed := []rune(req.Seed)[0]
		resp := model.Generate(seed, req.Length, req.Temperature)

		json.NewEncoder(w).Encode(GenerateResponse{
			Success: true,
			Result:  resp,
		})
	}
}

// --- 6. EXECUÇÃO PRINCIPAL ---
func main() {
	// Configuração
	config := Config{
		Epochs:       50,
		LearningRate: 0.01,
		BatchSize:    32,
		Temperature:  0.8,
		Port:         ":8080",
		ModelPath:    "modelo_treinado.bin",
		InputFile:    "livro.txt",
	}

	log.Println("=== LMCS LLM IA ===")
	log.Printf("Configuração: %+v\n", config)

	// Carregar texto do livro
	data, err := os.ReadFile(config.InputFile)
	var content string
	if err != nil {
		log.Printf("Aviso: Não foi possível ler %s, usando texto padrão\n", config.InputFile)
		content = "o rato roeu a roupa do rei de roma. o rei mandou buscar outro rato." // Fallback
	} else {
		content = string(data)
		log.Printf("Texto carregado: %d caracteres\n", len(content))
	}

	var model *LmcsLLM
	if _, err := os.Stat(config.ModelPath); err == nil {
		log.Printf("Carregando modelo existente de %s...\n", config.ModelPath)
		model, err = LoadModel(config.ModelPath)
		if err != nil {
			log.Fatalf("Erro ao carregar modelo: %v\n", err)
		}
		log.Println("Modelo carregado com sucesso!")
	} else {
		model = NewLmcsLLM(content)
	}

	// Treinar
	model.Train(content, config.Epochs, config.LearningRate, config.BatchSize)

	// Salvar modelo
	if err := model.Save(config.ModelPath); err != nil {
		log.Printf("Erro ao salvar modelo: %v\n", err)
	} else {
		log.Printf("Modelo salvo em %s\n", config.ModelPath)
	}

	// Configurar API
	http.HandleFunc("/ask", handleAsk(model))
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status": "ok",
			"model":  "LMCS LLM",
		})
	})

	log.Printf("API rodando em http://localhost%s\n", config.Port)
	log.Printf("Endpoints:")
	log.Printf("  GET  /ask?q=o&length=100&temperature=0.8")
	log.Printf("  POST /ask {\"seed\": \"o\", \"length\": 100, \"temperature\": 0.8}")
	log.Printf("  GET  /health")

	if err := http.ListenAndServe(config.Port, nil); err != nil {
		log.Fatalf("Erro ao iniciar servidor: %v\n", err)
	}
}
