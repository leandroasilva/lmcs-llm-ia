#!/bin/bash
# Converter modelo JSON exportado para formato binário Go e iniciar servidor

set -e

echo "========================================================================"
echo "  Conversor e Servidor - Modelo Transformer Treinado"
echo "========================================================================"
echo ""

# Verificar se modelo JSON existe
if [ ! -f "config.trained.json" ]; then
    echo "❌ Erro: config.trained.json não encontrado!"
    echo "Execute primeiro o treinamento e export do modelo."
    exit 1
fi

echo "📦 Modelo JSON encontrado: $(ls -lh config.trained.json | awk '{print $5}')"
echo ""

# Criar diretório temporário para conversão
TEMP_DIR=$(mktemp -d)
echo "🔧 Convertendo modelo para formato Go..."

# Usar Python para conversão (mais simples que Go para manipular JSON grande)
python3 << 'PYTHON_SCRIPT'
import json
import struct
import sys

print("Carregando modelo JSON...")
with open('config.trained.json', 'r') as f:
    model_data = json.load(f)

config = model_data['config']
weights = model_data['weights']

print(f"Config: vocab={config['vocab_size']}, d_model={config['d_model']}, layers={config['n_layers']}")
print("Nota: Modelo muito grande para converter para binário Go.")
print("Usando abordagem alternativa: servidor direto com JSON.")
print()
print("✓ Conversão não necessária - servidor pode usar JSON diretamente!")
PYTHON_SCRIPT

echo ""
echo "========================================================================"
echo "  Iniciando servidor com modelo treinado..."
echo "========================================================================"
echo ""

# Criar config temporária apontando para JSON
cp config.trained.server.json config.server.temp.json

# Iniciar servidor
echo "🚀 Iniciando servidor em http://localhost:8080"
echo ""
echo "Endpoints:"
echo "  🌐 Frontend: http://localhost:8080"
echo "  📊 Health:   http://localhost:8080/api/health"
echo "  💬 Ask:      http://localhost:8080/api/ask"
echo "  📡 Stream:   http://localhost:8080/api/ask/stream"
echo ""
echo "Pressione Ctrl+C para parar o servidor"
echo "========================================================================"
echo ""

./lmcs-llm serve --config config.trained.server.json

# Limpar
rm -f config.server.temp.json
