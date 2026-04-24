package main

import (
	"log"
	"net/http"
	"os"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/api"
	"github.com/leandroasilva/lmcs-llm-ia/config"
	"github.com/leandroasilva/lmcs-llm-ia/model"
)

func main() {
	log.Println("=== LMCS LLM IA ===")

	// Carregar configurações
	cfg := config.DefaultConfig()

	// Tentar carregar de arquivo se existir
	if _, err := os.Stat("config.json"); err == nil {
		log.Println("Carregando configurações de config.json...")
		if loadedCfg, err := config.LoadFromFile("config.json"); err != nil {
			log.Printf("Aviso: Erro ao carregar config.json, usando padrões: %v\n", err)
		} else {
			cfg = loadedCfg
		}
	}

	// Validar configurações
	if err := cfg.Validate(); err != nil {
		log.Fatalf("Configurações inválidas: %v\n", err)
	}

	log.Printf("Configuração: %+v\n", cfg)

	// Carregar texto de treinamento
	data, err := os.ReadFile(cfg.Paths.InputFile)
	var content string
	if err != nil {
		log.Printf("Aviso: Não foi possível ler %s, usando texto padrão\n", cfg.Paths.InputFile)
		content = "o rato roeu a roupa do rei de roma. o rei mandou buscar outro rato."
	} else {
		content = string(data)
		log.Printf("Texto carregado: %d caracteres\n", len(content))

		// Pré-processar texto
		log.Println("Pré-processando texto...")
		content = model.PreprocessText(content)
		log.Printf("Texto após pré-processamento: %d caracteres\n", len(content))
	}

	// Carregar ou criar modelo
	var mdl *model.LmcsLLM
	var lstmMdl *model.LstmModel

	if cfg.Training.UseLSTM {
		// Usar modelo LSTM
		if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
			log.Printf("Carregando modelo LSTM existente de %s...\n", cfg.Paths.ModelPath)
			lstmMdl, err = model.LoadLstmModel(cfg.Paths.ModelPath)
			if err != nil {
				log.Printf("Erro ao carregar modelo LSTM, criando novo: %v\n", err)
				lstmMdl = nil
			}
		}

		if lstmMdl == nil {
			log.Println("Inicializando novo modelo LSTM...")
			// Criar mapeamento de caracteres
			charToID, idToChar := model.BuildVocab(content)
			vocabSize := len(charToID)

			lstmMdl = model.NewLstmModel(
				vocabSize,
				cfg.Training.HiddenSize,
				cfg.Training.ContextSize,
				cfg.Training.LearningRate,
				charToID,
				idToChar,
			)
			log.Printf("Modelo LSTM criado: %s\n", lstmMdl.GetModelInfo())
		}

		// Treinar modelo LSTM
		log.Printf("Iniciando treinamento LSTM: %d épocas, lr=%.4f, batch=%d, context=%d, hidden=%d\n",
			cfg.Training.Epochs, cfg.Training.LearningRate, cfg.Training.BatchSize,
			cfg.Training.ContextSize, cfg.Training.HiddenSize)

		trainLstm(lstmMdl, content, cfg)

		// Salvar modelo LSTM
		if err := lstmMdl.SaveModel(cfg.Paths.ModelPath); err != nil {
			log.Printf("Erro ao salvar modelo LSTM: %v\n", err)
		} else {
			log.Printf("Modelo LSTM salvo em %s\n", cfg.Paths.ModelPath)
		}
	} else {
		// Usar modelo antigo (softmax regression)
		if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
			log.Printf("Carregando modelo existente de %s...\n", cfg.Paths.ModelPath)
			mdl, err = model.Load(cfg.Paths.ModelPath)
			if err != nil {
				log.Fatalf("Erro ao carregar modelo: %v\n", err)
			}
			log.Println("Modelo carregado com sucesso!")
		} else {
			mdl = model.New(content, cfg.Training.ContextSize)
		}

		// Treinar modelo
		mdl.Train(
			content,
			cfg.Training.Epochs,
			cfg.Training.LearningRate,
			cfg.Training.BatchSize,
		)

		// Salvar modelo treinado
		if err := mdl.Save(cfg.Paths.ModelPath); err != nil {
			log.Printf("Erro ao salvar modelo: %v\n", err)
		} else {
			log.Printf("Modelo salvo em %s\n", cfg.Paths.ModelPath)
		}
	}

	// Configurar servidor HTTP
	mux := http.NewServeMux()

	if cfg.Training.UseLSTM && lstmMdl != nil {
		handler := api.NewLstmHandler(lstmMdl)
		handler.RegisterRoutes(mux)
	} else {
		handler := api.NewHandler(mdl)
		handler.RegisterRoutes(mux)
	}

	log.Printf("Servidor rodando em http://%s%s\n", cfg.Server.Host, cfg.Server.Port)
	log.Println("Frontend: http://localhost" + cfg.Server.Port)
	log.Println("API Endpoints:")
	log.Println("  GET  /api/health")
	log.Println("  POST /api/ask")

	// Iniciar servidor
	if err := http.ListenAndServe(cfg.Server.Port, mux); err != nil {
		log.Fatalf("Erro ao iniciar servidor: %v\n", err)
	}
}

// trainLstm treina o modelo LSTM
func trainLstm(mdl *model.LstmModel, content string, cfg *config.Config) {
	startTime := time.Now()
	chars := []rune(content)
	totalLoss := 0.0
	reportInterval := 10

	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		epochLoss := 0.0
		samples := 0

		// Treinar em batches
		for i := 0; i < len(chars)-mdl.ContextSize-1; i += cfg.Training.BatchSize {
			// Preparar input e target
			end := i + mdl.ContextSize
			if end >= len(chars)-1 {
				break
			}

			// Converter contexto para índices
			inputs := make([]int, 0, mdl.ContextSize)
			for j := i; j < end; j++ {
				if id, ok := mdl.CharToID[chars[j]]; ok {
					inputs = append(inputs, id)
				}
			}

			if len(inputs) == 0 {
				continue
			}

			// Target
			targetChar := chars[end]
			target, ok := mdl.CharToID[targetChar]
			if !ok {
				continue
			}

			// Treinar
			loss := mdl.Train(inputs, target)
			epochLoss += loss
			samples++
		}

		if samples > 0 {
			avgLoss := epochLoss / float64(samples)
			totalLoss = avgLoss

			// Reportar progresso
			if epoch%reportInterval == 0 {
				elapsed := time.Since(startTime)
				log.Printf("Época %d/%d - Loss: %.4f | Tempo: %s\n",
					epoch, cfg.Training.Epochs, avgLoss, elapsed)
			}
		}
	}

	log.Printf("Treinamento LSTM concluído! Loss final: %.4f\n", totalLoss)
}
