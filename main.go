package main

import (
	"fmt"
	"log"
	"math"
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
		case "--download-enriched", "-de":
			log.Println("📥 Modo download de dataset enriquecido com metadados")
			if err := dataset.DownloadEnrichedDataset(); err != nil {
				log.Fatalf("Erro ao baixar dataset enriquecido: %v\n", err)
			}
			log.Println("✅ Dataset enriquecido baixado com sucesso!")
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

	// Carregar ou criar modelo Transformer
	var transformerMdl *model.TransformerModel
	modelLoaded := false

	if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
		log.Printf("Carregando modelo Transformer existente de %s...\n", cfg.Paths.ModelPath)
		transformerMdl, err = model.LoadTransformerModel(cfg.Paths.ModelPath)
		if err != nil {
			log.Printf("Erro ao carregar modelo Transformer, criando novo: %v\n", err)
			transformerMdl = nil
		} else {
			modelLoaded = true
			log.Printf("Modelo Transformer carregado com sucesso!\n")
		}
	}

	if transformerMdl == nil {
		log.Println("Inicializando novo modelo Transformer...")
		// Criar vocabulário word-level
		vocab, wordToID, idToWord := model.BuildVocabTransformer(content, cfg.Training.MaxVocab)
		vocabSize := len(vocab)

		transformerMdl = model.NewTransformerModel(
			vocabSize,
			cfg.Training.DModel,
			cfg.Training.NHeads,
			cfg.Training.NumLayers,
			cfg.Training.MaxSeqLen,
			cfg.Training.FFHidden,
			cfg.Training.LearningRate,
		)
		transformerMdl.Vocab = vocab
		transformerMdl.WordToID = wordToID
		transformerMdl.IDToWord = idToWord
		log.Printf("Modelo Transformer criado: vocab=%d, d_model=%d, heads=%d, layers=%d\n",
			vocabSize, cfg.Training.DModel, cfg.Training.NHeads, cfg.Training.NumLayers)
	}

	// Treinar modelo Transformer
	shouldTrain := !modelLoaded
	for _, arg := range os.Args {
		if arg == "--train" || arg == "-t" {
			shouldTrain = true
			log.Println("\n🔄 Modo treinamento adicional ativado!")
		}
	}

	if shouldTrain {
		if modelLoaded {
			log.Printf("\nContinuando treinamento: %d épocas já treinadas", transformerMdl.EpochsTrained)
			log.Printf("Adicionando %d novas épocas...\n", cfg.Training.Epochs)
		} else {
			log.Printf("\nIniciando treinamento Transformer: %d épocas, lr=%.4f, batch=%d\n",
				cfg.Training.Epochs, cfg.Training.LearningRate, cfg.Training.BatchSize)
		}

		trainTransformer(transformerMdl, content, cfg)

		// Salvar modelo Transformer
		if err := transformerMdl.SaveModel(cfg.Paths.ModelPath); err != nil {
			log.Printf("Erro ao salvar modelo Transformer: %v\n", err)
		} else {
			log.Printf("\n✅ Modelo Transformer salvo em %s", cfg.Paths.ModelPath)
			log.Printf("Total de épocas treinadas: %d", transformerMdl.EpochsTrained)
			log.Println("💡 Use './lmcs-llm' para carregar o modelo ou './lmcs-llm --train' para treinar mais\n")
		}
	} else {
		log.Println("\n✅ Modelo já treinado carregado. Pronto para uso!")
		log.Printf("📊 Status: %d épocas treinadas", transformerMdl.EpochsTrained)
		log.Println("💡 Use './lmcs-llm --train' para adicionar mais épocas de treinamento")
	}

	// Configurar servidor HTTP
	mux := http.NewServeMux()
	handler := api.NewHandler(transformerMdl)
	handler.RegisterRoutes(mux)

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
// trainTransformer treina o modelo Transformer
func trainTransformer(mdl *model.TransformerModel, content string, cfg *config.Config) {
	startTime := time.Now()
	initialLR := cfg.Training.LearningRate

	// Tokenizar todo o conteúdo
	tokens := mdl.Tokenize(content)
	totalTokens := len(tokens)
	log.Printf("Dataset: %d tokens (vocab: %d)\n", totalTokens, len(mdl.Vocab))

	for epoch := 1; epoch <= cfg.Training.Epochs; epoch++ {
		// Learning rate scheduling
		decayFactor := math.Pow(0.95, float64(epoch-1))
		currentLR := initialLR * decayFactor
		mdl.LearningRate = currentLR

		epochLoss := 0.0
		samples := 0

		// Training loop com teacher forcing
		for i := 0; i < totalTokens-cfg.Training.MaxSeqLen-1; i += cfg.Training.BatchSize {
			seqEnd := i + cfg.Training.MaxSeqLen
			if seqEnd >= totalTokens-1 {
				break
			}

			inputSeq := tokens[i:seqEnd]
			targetToken := tokens[seqEnd]

			// Forward pass
			output := mdl.Forward(inputSeq)

			// Calcular loss na última posição
			seqLen := output.RawMatrix().Rows
			lastRow := make([]float64, mdl.DModel)
			for j := 0; j < mdl.DModel; j++ {
				lastRow[j] = output.At(seqLen-1, j)
			}

			// Calcular logits
			logits := make([]float64, mdl.VocabSize)
			for v := 0; v < mdl.VocabSize; v++ {
				logits[v] = mdl.BOut.At(v, 0)
				for j := 0; j < mdl.DModel; j++ {
					logits[v] += mdl.WOut.At(v, j) * lastRow[j]
				}
			}

			// Cross-entropy loss
			probs := model.Softmax(logits)
			loss := model.CrossEntropyLoss(probs, targetToken)
			epochLoss += loss

			// Update pesos (simplified gradient descent)
			for v := 0; v < mdl.VocabSize; v++ {
				grad := probs[v]
				if v == targetToken {
					grad -= 1.0
				}
				// Update output weights
				for j := 0; j < mdl.DModel; j++ {
					mdl.WOut.Set(v, j, mdl.WOut.At(v, j)-currentLR*grad*lastRow[j]*0.01)
				}
				mdl.BOut.Set(v, 0, mdl.BOut.At(v, 0)-currentLR*grad*0.01)
			}

			samples++
		}

		if samples > 0 {
			avgLoss := epochLoss / float64(samples)

			// Reportar progresso
			if epoch%5 == 0 || epoch == 1 {
				elapsed := time.Since(startTime)
				log.Printf("Época %d/%d - Loss: %.4f | LR: %.6f | Samples: %d | Tempo: %s\n",
					epoch, cfg.Training.Epochs, avgLoss, currentLR, samples, elapsed)
			}
		}
	}

	log.Printf("Treinamento Transformer concluído!\n")
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
  ./lmcs-llm -d               Baixar dataset simples do HuggingFace
  ./lmcs-llm -de              Baixar dataset enriquecido COM metadados (RECOMENDADO)
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
