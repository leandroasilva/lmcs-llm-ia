#!/bin/bash
# Script para iniciar treinamento conversacional do zero

echo "========================================"
echo "🤖 LMCS LLM - Modo Conversacional GPT"
echo "========================================"
echo ""

# Verificar dataset
if [ ! -f "livro.txt" ]; then
    echo "❌ Dataset não encontrado!"
    echo "Execute: python3 download_dataset.py"
    exit 1
fi

DATASET_SIZE=$(du -h livro.txt | cut -f1)
DIALOGS=$(grep -c "^Usuário:" livro.txt)

echo "📊 Dataset:"
echo "   Tamanho: $DATASET_SIZE"
echo "   Diálogos: $DIALOGS"
echo ""

# Verificar se existe modelo antigo
if [ -f "lmcs-model.bin" ]; then
    echo "⚠️  Modelo existente encontrado!"
    echo ""
    read -p "Deseja treinar um NOVO modelo do zero? (s/N): " RETRAIN
    
    if [[ "$RETRAIN" =~ ^[Ss]$ ]]; then
        rm -f lmcs-model.bin
        echo "🗑️  Modelo antigo removido."
    else
        echo ""
        echo "Iniciando servidor com modelo existente..."
        ./lmcs-llm
        exit 0
    fi
fi

echo ""
echo "🚀 Iniciando treinamento conversacional..."
echo "   Dataset: Brazilian Customer Service (755 conversas)"
echo "   Config: 300 épocas, context=200, hidden=256"
echo ""
echo "⏱️  Tempo estimado: ~30-45 minutos"
echo ""

# Iniciar treinamento
./lmcs-llm
