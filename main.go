package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"runtime"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/api"
	"github.com/leandroasilva/lmcs-llm-ia/config"
	"github.com/leandroasilva/lmcs-llm-ia/dataset"
	"github.com/leandroasilva/lmcs-llm-ia/model"
)

func main() {
	log.Println("=== LMCS LLM IA ===")

	// Verificar parâmetros de linha de comando
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "--download-dataset", "-d":
			log.Println("📥 Modo download de dataset")
			if err := dataset.DownloadDataset(); err != nil {
				log.Fatalf("Erro ao baixar dataset: %v\n", err)
			}
			log.Println("✅ Dataset baixado com sucesso!")
			return
		case "--help", "-h":
			printHelp()
			return
		}
	}

	// Configurar número de threads/cores para máxima performance
	numCPUs := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPUs)
	log.Printf("Usando %d threads/cores (GOMAXPROCS)\n", numCPUs)

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
		modelLoaded := false
		if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
			log.Printf("Carregando modelo LSTM existente de %s...\n", cfg.Paths.ModelPath)
			lstmMdl, err = model.LoadLstmModel(cfg.Paths.ModelPath)
			if err != nil {
				log.Printf("Erro ao carregar modelo LSTM, criando novo: %v\n", err)
				lstmMdl = nil
			} else {
				modelLoaded = true
				log.Printf("Modelo LSTM carregado com sucesso! %s\n", lstmMdl.GetModelInfo())
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
				cfg.Training.NumLayers,
				cfg.Training.LearningRate,
				charToID,
				idToChar,
			)
			log.Printf("Modelo LSTM criado: %s\n", lstmMdl.GetModelInfo())
		}

		// Treinar modelo LSTM
		// Verificar se deve treinar (novo ou flag --train)
		shouldTrain := !modelLoaded
		for _, arg := range os.Args {
			if arg == "--train" || arg == "-t" {
				shouldTrain = true
				log.Println("\n🔄 Modo treinamento adicional ativado!")
			}
		}

		if shouldTrain {
			if modelLoaded {
				log.Printf("\nContinuando treinamento: %d épocas já treinadas", lstmMdl.EpochsTrained)
				log.Printf("Adicionando %d novas épocas...\n", cfg.Training.Epochs)
			} else {
				log.Printf("\nIniciando treinamento LSTM: %d épocas, lr=%.4f, batch=%d, context=%d, hidden=%d\n",
					cfg.Training.Epochs, cfg.Training.LearningRate, cfg.Training.BatchSize,
					cfg.Training.ContextSize, cfg.Training.HiddenSize)
			}

			trainLstm(lstmMdl, content, cfg)

			// Salvar modelo LSTM
			if err := lstmMdl.SaveModel(cfg.Paths.ModelPath); err != nil {
				log.Printf("Erro ao salvar modelo LSTM: %v\n", err)
			} else {
				log.Printf("\n✅ Modelo LSTM salvo em %s", cfg.Paths.ModelPath)
				log.Printf("Total de épocas treinadas: %d", lstmMdl.EpochsTrained)
				log.Printf("💡 Use './lmcs-llm' para carregar o modelo ou './lmcs-llm --train' para treinar mais\n")
			}
		} else {
			log.Println("\n✅ Modelo já treinado carregado. Pronto para uso!")
			log.Printf("📊 Status: %d épocas treinadas", lstmMdl.EpochsTrained)
			log.Println("💡 Use './lmcs-llm --train' para adicionar mais épocas de treinamento")
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

// trainLstm treina o modelo LSTM com paralelismo
func trainLstm(mdl *model.LstmModel, content string, cfg *config.Config) {
	startTime := time.Now()
	chars := []rune(content)
	totalLoss := 0.0
	reportInterval := 10
	numWorkers := runtime.NumCPU()

	log.Printf("Iniciando treinamento LSTM: %d épocas, lr=%.4f, batch=%d, context=%d, hidden=%d, workers=%d\n",
		cfg.Training.Epochs, cfg.Training.LearningRate, cfg.Training.BatchSize, mdl.ContextSize, mdl.HiddenSize, numWorkers)

	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		epochLoss := 0.0
		samples := 0

		// Preparar todos os samples para processamento paralelo
		type trainingSample struct {
			inputs []int
			target int
		}

		var samplesList []trainingSample
		for i := 0; i < len(chars)-mdl.ContextSize-1; i += cfg.Training.BatchSize {
			end := i + mdl.ContextSize
			if end >= len(chars)-1 {
				break
			}

			inputs := make([]int, 0, mdl.ContextSize)
			for j := i; j < end; j++ {
				if id, ok := mdl.CharToID[chars[j]]; ok {
					inputs = append(inputs, id)
				}
			}

			if len(inputs) == 0 {
				continue
			}

			targetChar := chars[end]
			target, ok := mdl.CharToID[targetChar]
			if !ok {
				continue
			}

			samplesList = append(samplesList, trainingSample{inputs, target})
		}

		// Processar samples em paralelo usando goroutines
		if numWorkers > 1 && len(samplesList) > 100 {
			// Dividir samples entre workers
			chunkSize := (len(samplesList) + numWorkers - 1) / numWorkers
			workerLoss := make([]float64, numWorkers)
			workerSamples := make([]int, numWorkers)
			done := make(chan bool, numWorkers)

			for w := 0; w < numWorkers; w++ {
				start := w * chunkSize
				end := start + chunkSize
				if end > len(samplesList) {
					end = len(samplesList)
				}

				if start >= len(samplesList) {
					break
				}

				chunk := samplesList[start:end]
				go func(workerID int, chunk []trainingSample) {
					localLoss := 0.0
					localSamples := 0
					for _, sample := range chunk {
						loss := mdl.Train(sample.inputs, sample.target)
						localLoss += loss
						localSamples++
					}
					workerLoss[workerID] = localLoss
					workerSamples[workerID] = localSamples
					done <- true
				}(w, chunk)
			}

			// Esperar todos workers completarem
			for w := 0; w < numWorkers; w++ {
				<-done
			}

			// Agregar resultados
			for w := 0; w < numWorkers; w++ {
				epochLoss += workerLoss[w]
				samples += workerSamples[w]
			}
		} else {
			// Processamento sequencial para datasets pequenos
			for _, sample := range samplesList {
				loss := mdl.Train(sample.inputs, sample.target)
				epochLoss += loss
				samples++
			}
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

	// Incrementar contador de épocas treinadas
	mdl.EpochsTrained += cfg.Training.Epochs
	log.Printf("Total de épocas treinadas (acumulado): %d\n", mdl.EpochsTrained)
}

// printHelp exibe ajuda dos parâmetros
func printHelp() {
	fmt.Println(`
🤖 LMCS LLM IA - Assistente Conversacional

Uso:
  ./lmcs-llm                  Iniciar servidor (treina se não houver modelo)
  ./lmcs-llm --train          Treinar mais épocas (incremental)
  ./lmcs-llm -d               Baixar dataset do HuggingFace
  ./lmcs-llm --help           Mostrar esta ajuda

Exemplos:
  # Baixar dataset conversacional
  ./lmcs-llm --download-dataset

  # Treinar modelo do zero
  ./lmcs-llm

  # Adicionar mais épocas de treinamento
  ./lmcs-llm --train

  # Iniciar servidor com modelo existente
  ./lmcs-llm

Dataset:
  O download baixa conversas de atendimento em português do HuggingFace:
  - Brazilian Customer Service Conversations
  - Formato: Usuário/Assistente
  - Gera: dataset/data/train.txt (e livro.txt para compatibilidade)
`)
}
