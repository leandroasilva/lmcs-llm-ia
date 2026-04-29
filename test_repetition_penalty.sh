#!/bin/bash
# Script para testar a penalidade de repetição com diferentes configurações

BASE_URL="http://localhost:8080"
TEST_QUESTIONS=(
    "ola como voce esta"
    "qual e o seu nome"
    "me ajude com um problema tecnico"
)

echo "========================================="
echo "Teste de Penalidade de Repetição"
echo "========================================="
echo ""

for question in "${TEST_QUESTIONS[@]}"; do
    echo "-----------------------------------------"
    echo "Pergunta: $question"
    echo "-----------------------------------------"
    
    # Fazer requisição
    START_TIME=$(date +%s%N)
    RESPONSE=$(curl -s -X POST "$BASE_URL/api/ask" \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}")
    END_TIME=$(date +%s%N)
    
    # Extrair resposta e tempo
    ANSWER=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['answer'])")
    ELAPSED=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['elapsed_ms'])")
    
    # Contar repetições (tokens que aparecem mais de 2 vezes)
    REPEAT_COUNT=$(echo "$ANSWER" | tr ' ' '\n' | sort | uniq -c | sort -rn | awk '$1 > 2 {count++} END {print count+0}')
    
    # Contar tokens únicos
    UNIQUE_TOKENS=$(echo "$ANSWER" | tr ' ' '\n' | sort -u | wc -l | tr -d ' ')
    TOTAL_TOKENS=$(echo "$ANSWER" | tr ' ' '\n' | wc -l | tr -d ' ')
    
    # Calcular ratio de unicidade
    if [ "$TOTAL_TOKENS" -gt 0 ]; then
        UNIQUENESS=$(echo "scale=2; $UNIQUE_TOKENS * 100 / $TOTAL_TOKENS" | bc)
    else
        UNIQUENESS="0"
    fi
    
    echo "Resposta: $ANSWER"
    echo "Tempo: ${ELAPSED}ms"
    echo "Tokens: $TOTAL_TOKENS total, $UNIQUE_TOKENS únicos ($UNIQUENESS%)"
    echo "Repetições excessivas: $REPEAT_COUNT tokens"
    echo ""
done

echo "========================================="
echo "Teste concluído!"
echo "========================================="
