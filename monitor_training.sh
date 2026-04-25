#!/bin/bash
# Script para monitorar o progresso do treinamento
# Uso: ./monitor_training.sh

LOG_FILE="training.log"

echo "=================================="
echo "📊 Monitor de Treinamento LSTM"
echo "=================================="
echo ""

# Verificar se o binário existe
if [ ! -f "lmcs-llm" ]; then
    echo "❌ Binário não encontrado. Compile primeiro:"
    echo "   go build -o lmcs-llm"
    exit 1
fi

# Verificar se já existe modelo
if [ -f "lmcs-model.bin" ]; then
    MODEL_SIZE=$(du -h lmcs-model.bin | cut -f1)
    MODEL_DATE=$(stat -f "%Sm" lmcs-model.bin)
    echo "✅ Modelo existente: $MODEL_SIZE (atualizado em $MODEL_DATE)"
    echo ""
    read -p "Deseja treinar um novo modelo? (s/N): " RETRAIN
    if [[ ! "$RETRAIN" =~ ^[Ss]$ ]]; then
        echo "Carregando modelo existente..."
        echo "Iniciando servidor..."
        ./lmcs-llm
        exit 0
    fi
    rm -f lmcs-model.bin
    echo "🗑️  Modelo antigo removido."
    echo ""
fi

# Verificar dataset
if [ ! -f "livro.txt" ]; then
    echo "❌ Dataset 'livro.txt' não encontrado!"
    echo "Execute: ./extract_pdfs.sh"
    exit 1
fi

DATASET_SIZE=$(du -h livro.txt | cut -f1)
echo "📚 Dataset: $DATASET_SIZE"
echo ""

# Iniciar treinamento com log
echo "🚀 Iniciando treinamento..."
echo "Logs serão salvos em: $LOG_FILE"
echo ""

# Rodar treinamento e capturar logs
./lmcs-llm 2>&1 | tee "$LOG_FILE" &
TRAINING_PID=$!

echo "Treinamento iniciado (PID: $TRAINING_PID)"
echo ""

# Monitorar progresso
sleep 10
while kill -0 $TRAINING_PID 2>/dev/null; do
    # Extrair última loss dos logs
    LAST_LOSS=$(grep -o "loss=[0-9.]*" "$LOG_FILE" | tail -1)
    LAST_EPOCH=$(grep -o "Época [0-9]*" "$LOG_FILE" | tail -1)
    
    if [ -n "$LAST_LOSS" ]; then
        echo "📈 $LAST_EPOCH - $LAST_LOSS ($(date '+%H:%M:%S'))"
    fi
    
    sleep 30
done

# Verificar resultado
wait $TRAINING_PID
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Treinamento concluído com sucesso!"
    echo ""
    
    if [ -f "lmcs-model.bin" ]; then
        FINAL_SIZE=$(du -h lmcs-model.bin | cut -f1)
        echo "📦 Modelo salvo: $FINAL_SIZE"
        echo ""
        
        # Mostrar últimas métricas
        echo "📊 Últimas métricas:"
        grep "loss=" "$LOG_FILE" | tail -5
        echo ""
        
        echo "🚀 Iniciando servidor..."
        ./lmcs-llm
    fi
else
    echo ""
    echo "❌ Treinamento falhou (exit code: $EXIT_CODE)"
    echo "Verifique os logs: $LOG_FILE"
fi
