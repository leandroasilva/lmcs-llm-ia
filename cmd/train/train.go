package train

import (
	"log"
	"math"
	"os"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
	"github.com/leandroasilva/lmcs-llm-ia/internal/model"
	"github.com/leandroasilva/lmcs-llm-ia/internal/training"
)

func RunTrain(configPath string) error {
	// Carregar configurações
	cfg := config.DefaultConfig()

	if _, err := os.Stat(configPath); err == nil {
		log.Printf("Carregando configurações de %s...\n", configPath)
		if loadedCfg, err := config.LoadFromFile(configPath); err != nil {
			log.Printf("Aviso: Erro ao carregar %s, usando padrões: %v\n", configPath, err)
		} else {
			cfg = loadedCfg
		}
	}

	// Validar configurações
	if err := cfg.Validate(); err != nil {
		return err
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
	var transformerMdl *model.TransformerModel
	modelLoaded := false

	if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
		log.Printf("Carregando modelo existente de %s...\n", cfg.Paths.ModelPath)
		transformerMdl, err = model.LoadTransformerModel(cfg.Paths.ModelPath)
		if err != nil {
			log.Printf("Erro ao carregar modelo, criando novo: %v\n", err)
			transformerMdl = nil
		} else {
			modelLoaded = true
			log.Printf("Modelo carregado com sucesso!\n")
		}
	}

	if transformerMdl == nil {
		log.Println("Inicializando novo modelo...")
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
			cfg.Training.DropoutRate,
			cfg.Training.WeightDecay,
		)
		transformerMdl.Vocab = vocab
		transformerMdl.WordToID = wordToID
		transformerMdl.IDToWord = idToWord
		log.Printf("Modelo criado: vocab=%d, d_model=%d, heads=%d, layers=%d\n",
			vocabSize, cfg.Training.DModel, cfg.Training.NHeads, cfg.Training.NumLayers)
	}

	// Iniciar métricas de treinamento
	training.GlobalMetrics.StartTraining(cfg.Training.Epochs)
	training.GlobalMetrics.LearningRate = cfg.Training.LearningRate
	training.GlobalMetrics.BatchSize = cfg.Training.BatchSize

	// Treinar modelo
	if modelLoaded {
		log.Printf("\nContinuando treinamento: %d épocas já treinadas", transformerMdl.EpochsTrained)
		log.Printf("Adicionando %d novas épocas...\n", cfg.Training.Epochs)
	} else {
		log.Printf("\nIniciando treinamento: %d épocas, lr=%.4f, batch=%d\n",
			cfg.Training.Epochs, cfg.Training.LearningRate, cfg.Training.BatchSize)
	}

	trainTransformer(transformerMdl, content, cfg, training.GlobalMetrics)

	// Finalizar métricas
	training.GlobalMetrics.EndTraining()

	// Salvar modelo
	if err := transformerMdl.SaveModel(cfg.Paths.ModelPath); err != nil {
		return err
	}

	log.Printf("\n✅ Modelo salvo em %s", cfg.Paths.ModelPath)
	log.Printf("Total de épocas treinadas: %d", transformerMdl.EpochsTrained)
	return nil
}

func trainTransformer(mdl *model.TransformerModel, content string, cfg *config.Config, metrics *training.TrainingMetrics) {
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

			// Backpropagation
			mdl.BackwardPropagation(inputSeq, targetToken, output, currentLR)

			samples++
		}

		if samples > 0 {
			avgLoss := epochLoss / float64(samples)
			metrics.ProcessedSamples += samples

			// Atualizar métricas
			metrics.UpdateEpoch(epoch, avgLoss, metrics.ProcessedSamples)

			// Reportar progresso
			if epoch%5 == 0 || epoch == 1 {
				elapsed := time.Since(startTime)
				perplexity := metrics.Perplexity
				log.Printf("Época %d/%d - Loss: %.4f | Perplexity: %.2f | LR: %.6f | Samples: %d | Tempo: %s\n",
					epoch, cfg.Training.Epochs, avgLoss, perplexity, currentLR, samples, elapsed)
			}
		}
	}

	log.Printf("Treinamento concluído!\n")
	mdl.EpochsTrained += cfg.Training.Epochs
	log.Printf("Total de épocas treinadas (acumulado): %d\n", mdl.EpochsTrained)
}
