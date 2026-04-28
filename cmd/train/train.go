package train

import (
	"math"
	"os"
	"time"

	"github.com/leandroasilva/lmcs-llm-ia/internal/config"
	"github.com/leandroasilva/lmcs-llm-ia/internal/logger"
	"github.com/leandroasilva/lmcs-llm-ia/internal/model"
	"github.com/leandroasilva/lmcs-llm-ia/internal/training"
)

func RunTrain(configPath string) error {
	// Carregar configurações
	cfg := config.DefaultConfig()

	if _, err := os.Stat(configPath); err == nil {
		logger.Info("Loading configuration", "config_path", configPath)
		if loadedCfg, err := config.LoadFromFile(configPath); err != nil {
			logger.Warn("Error loading config, using defaults", "error", err)
		} else {
			cfg = loadedCfg
		}
	}

	// Validar configurações
	if err := cfg.Validate(); err != nil {
		return err
	}

	logger.Info("Configuration loaded", "config", cfg)

	// Carregar texto de treinamento
	data, err := os.ReadFile(cfg.Paths.InputFile)
	var content string
	if err != nil {
		logger.Warn("Could not read input file, using default text", "file", cfg.Paths.InputFile, "error", err)
		content = "o rato roeu a roupa do rei de roma. o rei mandou buscar outro rato."
	} else {
		content = string(data)
		logger.Info("Text loaded", "characters", len(content))

		// Pré-processar texto
		logger.Info("Preprocessing text")
		content = model.PreprocessText(content)
		logger.Info("Text preprocessed", "characters", len(content))
	}

	// Carregar ou criar modelo
	var transformerMdl *model.TransformerModel
	modelLoaded := false

	if _, err := os.Stat(cfg.Paths.ModelPath); err == nil {
		logger.Info("Loading existing model", "path", cfg.Paths.ModelPath)
		transformerMdl, err = model.LoadTransformerModel(cfg.Paths.ModelPath)
		if err != nil {
			logger.Warn("Error loading model, creating new one", "error", err)
			transformerMdl = nil
		} else {
			modelLoaded = true
			logger.Info("Model loaded successfully")
		}
	}

	if transformerMdl == nil {
		logger.Info("Initializing new model")
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
		logger.Info("Model created",
			"vocab", vocabSize,
			"d_model", cfg.Training.DModel,
			"heads", cfg.Training.NHeads,
			"layers", cfg.Training.NumLayers)
	}

	// Iniciar métricas de treinamento
	training.GlobalMetrics.StartTraining(cfg.Training.Epochs)
	training.GlobalMetrics.LearningRate = cfg.Training.LearningRate
	training.GlobalMetrics.BatchSize = cfg.Training.BatchSize

	// Treinar modelo
	if modelLoaded {
		logger.Info("Continuing training", "epochs_trained", transformerMdl.EpochsTrained, "new_epochs", cfg.Training.Epochs)
	} else {
		logger.Info("Starting training",
			"epochs", cfg.Training.Epochs,
			"learning_rate", cfg.Training.LearningRate,
			"batch_size", cfg.Training.BatchSize)
	}

	trainTransformer(transformerMdl, content, cfg, training.GlobalMetrics)

	// Finalizar métricas
	training.GlobalMetrics.EndTraining()

	// Salvar modelo
	if err := transformerMdl.SaveModel(cfg.Paths.ModelPath); err != nil {
		return err
	}

	logger.Info("Model saved", "path", cfg.Paths.ModelPath, "epochs_trained", transformerMdl.EpochsTrained)
	return nil
}

func trainTransformer(mdl *model.TransformerModel, content string, cfg *config.Config, metrics *training.TrainingMetrics) {
	startTime := time.Now()
	initialLR := cfg.Training.LearningRate

	// Tokenizar todo o conteúdo
	tokens := mdl.Tokenize(content)
	totalTokens := len(tokens)
	logger.Info("Dataset prepared", "tokens", totalTokens, "vocab", len(mdl.Vocab))

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
				logger.TrainingLog(epoch, cfg.Training.Epochs, avgLoss, perplexity,
					"learning_rate", currentLR,
					"samples", samples,
					"elapsed", elapsed.String())
			}
		}
	}

	logger.Info("Training completed", "total_epochs", mdl.EpochsTrained)
	mdl.EpochsTrained += cfg.Training.Epochs
	logger.Info("Total accumulated epochs", "epochs", mdl.EpochsTrained)
}
