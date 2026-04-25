#!/bin/bash
# Script para verificar status do modelo treinado
# Uso: ./check_model.sh

echo "=================================="
echo "📊 Status do Modelo LSTM"
echo "=================================="
echo ""

# Verificar se modelo existe
if [ ! -f "lmcs-model.bin" ]; then
    echo "❌ Nenhum modelo encontrado (lmcs-model.bin)"
    echo ""
    echo "💡 Treine um modelo primeiro:"
    echo "   ./train_incremental.sh 40"
    exit 1
fi

# Informações do arquivo
MODEL_SIZE=$(du -h lmcs-model.bin | cut -f1)
MODEL_DATE=$(stat -f "%Sm" lmcs-model.bin)
MODEL_EPOCH=$(stat -f "%Sm" lmcs-model.bin)

echo "✅ Modelo encontrado!"
echo ""
echo "📦 Arquivo:"
echo "   Tamanho: $MODEL_SIZE"
echo "   Última modificação: $MODEL_DATE"
echo ""

# Iniciar servidor temporário para verificar informações
echo "🔍 Carregando modelo para verificar detalhes..."
echo ""

# Iniciar em background
./lmcs-llm &> /tmp/lmcs_status.log &
SERVER_PID=$!

# Aguardar inicialização
sleep 3

# Verificar se servidor iniciou
if kill -0 $SERVER_PID 2>/dev/null; then
    # Fazer request para API
    HEALTH=$(curl -s http://localhost:8080/api/health 2>/dev/null)
    
    if [ -n "$HEALTH" ]; then
        echo "📈 Informações do Modelo:"
        echo ""
        
        # Extrair informações do health check
        echo "$HEALTH" | jq -r '.model' 2>/dev/null || echo "$HEALTH"
        echo ""
        
        # Verificar épocas nos logs
        EPOCHS=$(grep -o "épocas treinadas" /tmp/lmcs_status.log | wc -l)
        if [ "$EPOCHS" -gt 0 ]; then
            grep "épocas treinadas" /tmp/lmcs_status.log | tail -1
        fi
    else
        echo "⚠️  Não foi possível conectar ao servidor"
        echo "Verificando logs..."
        tail -20 /tmp/lmcs_status.log
    fi
    
    # Parar servidor
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
else
    echo "❌ Falha ao iniciar servidor"
    echo "Logs:"
    cat /tmp/lmcs_status.log
fi

echo ""
echo "=================================="
echo "💡 Próximos passos:"
echo ""
echo "  Testar modelo:"
echo "    ./train_incremental.sh  (opção 1)"
echo ""
echo "  Treinar mais:"
echo "    ./lmcs-llm --train"
echo ""
echo "  Ver logs completos:"
echo "    cat /tmp/lmcs_status.log"
echo "=================================="

# Limpar
rm -f /tmp/lmcs_status.log
