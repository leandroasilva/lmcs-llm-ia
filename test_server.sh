#!/bin/bash
# Teste rápido de geração com o servidor

echo "========================================================================"
echo "  Teste de Geração - Servidor LMCS LLM"
echo "========================================================================"
echo ""

# Verificar se servidor está rodando
if ! curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
    echo "❌ Servidor não está rodando!"
    echo "Inicie com: ./lmcs-llm serve --config config.trained.server.json"
    exit 1
fi

echo "✅ Servidor está rodando!"
echo ""

# Teste 1 - Health check
echo "📊 Teste 1: Health Check"
echo "------------------------------------------------------------------------"
curl -s http://localhost:8080/api/health | python3 -m json.tool
echo ""

# Teste 2 - Geração simples
echo "💬 Teste 2: Geração de Texto"
echo "------------------------------------------------------------------------"
echo "Prompt: 'ola como voce esta'"
echo ""

# Usar gtimeout ou esperar manualmente
if command -v gtimeout &> /dev/null; then
    TIMEOUT_CMD="gtimeout"
elif command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout"
else
    TIMEOUT_CMD=""  # macOS sem timeout
fi

echo "Aguardando resposta (pode levar 30-60 segundos)..."
echo ""

if [ -n "$TIMEOUT_CMD" ]; then
    $TIMEOUT_CMD 60 curl -s -X POST http://localhost:8080/api/ask \
        -H "Content-Type: application/json" \
        -d '{
            "question": "ola como voce esta",
            "temperature": 0.8,
            "top_k": 30
        }' | python3 -m json.tool
else
    # macOS: rodar em background e monitorar
    curl -s -X POST http://localhost:8080/api/ask \
        -H "Content-Type: application/json" \
        -d '{
            "question": "ola como voce esta",
            "temperature": 0.8,
            "top_k": 30
        }' > /tmp/llm_response.json &
    CURL_PID=$!
    
    # Aguardar até 60 segundos
    WAITED=0
    while kill -0 $CURL_PID 2>/dev/null; do
        if [ $WAITED -ge 60 ]; then
            kill $CURL_PID 2>/dev/null
            echo "⚠️ Timeout após 60 segundos"
            break
        fi
        sleep 5
        WAITED=$((WAITED + 5))
        echo "  Aguardando... ${WAITED}s"
    done
    
    if [ -f /tmp/llm_response.json ] && [ -s /tmp/llm_response.json ]; then
        python3 -m json.tool /tmp/llm_response.json
        rm -f /tmp/llm_response.json
    fi
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Teste concluído com sucesso!"
else
    echo ""
    echo "⚠️ Timeout ou erro na geração"
    echo "O modelo pode estar demorando para responder (modelo grande: 20M parâmetros)"
fi

echo ""
echo "========================================================================"
echo "  Para mais testes:"
echo "========================================================================"
echo ""
echo "  curl -X POST http://localhost:8080/api/ask \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"question\": \"sua pergunta aqui\"}'"
echo ""
