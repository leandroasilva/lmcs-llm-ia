#!/bin/bash
# Script para treinamento incremental do modelo LSTM
# Uso: ./train_incremental.sh [numero_epocas]

EPOCHS=${1:-40}

echo "========================================"
echo "🎯 Treinamento Incremental LSTM"
echo "========================================"
echo ""

# Verificar binário
if [ ! -f "lmcs-llm" ]; then
    echo "❌ Binário não encontrado. Compile primeiro:"
    echo "   go build -o lmcs-llm"
    exit 1
fi

# Verificar dataset
if [ ! -f "livro.txt" ]; then
    echo "❌ Dataset 'livro.txt' não encontrado!"
    exit 1
fi

# Verificar se já existe modelo
if [ -f "lmcs-model.bin" ]; then
    MODEL_SIZE=$(du -h lmcs-model.bin | cut -f1)
    echo "✅ Modelo existente: $MODEL_SIZE"
    echo ""
    echo "📊 Opções:"
    echo "  1. Testar modelo atual"
    echo "  2. Treinar mais $EPOCHS épocas (incremental)"
    echo "  3. Treinar do zero (deletar modelo atual)"
    echo ""
    read -p "Escolha (1/2/3): " OPTION
    
    case $OPTION in
        1)
            echo ""
            echo "🧪 Testando modelo atual..."
            echo ""
            
            # Iniciar servidor em background
            ./lmcs-llm &
            SERVER_PID=$!
            sleep 3
            
            # Testar geração
            echo "Teste 1: 'era uma vez'"
            curl -s -X POST http://localhost:8080/api/ask \
                -H "Content-Type: application/json" \
                -d "{\"seed\": \"era uma vez\", \"length\": 80, \"temperature\": 0.6}" | jq -r '.result'
            echo ""
            echo ""
            
            echo "Teste 2: 'o gato'"
            curl -s -X POST http://localhost:8080/api/ask \
                -H "Content-Type: application/json" \
                -d "{\"seed\": \"o gato\", \"length\": 80, \"temperature\": 0.7}" | jq -r '.result'
            echo ""
            echo ""
            
            # Parar servidor
            kill $SERVER_PID 2>/dev/null
            echo ""
            echo "✅ Testes concluídos!"
            ;;
        2)
            echo ""
            echo "🔄 Treinando mais $EPOCHS épocas (modo incremental)..."
            echo ""
            
            # Atualizar config.json com número de épocas
            if command -v jq &> /dev/null; then
                jq ".training.epochs = $EPOCHS" config.json > config_temp.json
                mv config_temp.json config.json
            fi
            
            # Treinar com flag --train
            ./lmcs-llm --train
            ;;
        3)
            echo ""
            read -p "⚠️  Tem certeza? Isso deletará o modelo atual (s/N): " CONFIRM
            if [[ "$CONFIRM" =~ ^[Ss]$ ]]; then
                rm -f lmcs-model.bin
                echo "🗑️  Modelo removido."
                echo ""
                echo "🚀 Iniciando treinamento do zero com $EPOCHS épocas..."
                echo ""
                
                if command -v jq &> /dev/null; then
                    jq ".training.epochs = $EPOCHS" config.json > config_temp.json
                    mv config_temp.json config.json
                fi
                
                ./lmcs-llm
            else
                echo "Operação cancelada."
            fi
            ;;
        *)
            echo "Opção inválida!"
            exit 1
            ;;
    esac
else
    echo "🆕 Nenhum modelo encontrado."
    echo ""
    echo "🚀 Iniciando primeiro treinamento com $EPOCHS épocas..."
    echo ""
    
    if command -v jq &> /dev/null; then
        jq ".training.epochs = $EPOCHS" config.json > config_temp.json
        mv config_temp.json config.json
    fi
    
    ./lmcs-llm
fi

echo ""
echo "========================================"
echo "✅ Processo concluído!"
echo "========================================"
