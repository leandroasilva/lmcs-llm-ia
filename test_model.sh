#!/bin/bash
# Script para testar o modelo LSTM com diversos prompts em português

BASE_URL="http://localhost:8080/api/ask"

echo "=========================================="
echo "  TESTE DO MODELO LSTM - PORTUGUÊS"
echo "=========================================="
echo ""

# Teste 1: Seed clássico
echo "📝 Teste 1: 'o rato roeu' (temperatura: 0.6)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "o rato roeu", "length": 100, "temperature": 0.6}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 2: Literatura
echo "📚 Teste 2: 'no meio do caminho' (temperatura: 0.5)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "no meio do caminho", "length": 100, "temperature": 0.5}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 3: Narrativa
echo "📖 Teste 3: 'era uma vez' (temperatura: 0.7)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "era uma vez", "length": 120, "temperature": 0.7}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 4: Educação
echo "🎓 Teste 4: 'a educação é importante' (temperatura: 0.6)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "a educação é importante", "length": 100, "temperature": 0.6}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 5: Temperatura baixa
echo "🌡️  Teste 5: 'a linguagem' (temperatura: 0.3 - baixa)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "a linguagem", "length": 80, "temperature": 0.3}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 6: Temperatura alta
echo "🔥 Teste 6: 'o conhecimento' (temperatura: 0.9 - alta)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "o conhecimento", "length": 100, "temperature": 0.9}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 7: Gramática
echo "📝 Teste 7: 'os acentos' (temperatura: 0.5)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "os acentos", "length": 90, "temperature": 0.5}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

# Teste 8: Longo
echo "📜 Teste 8: 'na sociedade moderna' (temperatura: 0.6, longo)"
echo "------------------------------------------"
curl -s -X POST $BASE_URL \
  -H "Content-Type: application/json" \
  -d '{"seed": "na sociedade moderna", "length": 150, "temperature": 0.6}' | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('result', 'Erro'))"
echo ""
echo ""

echo "=========================================="
echo "  TESTES CONCLUÍDOS!"
echo "=========================================="
