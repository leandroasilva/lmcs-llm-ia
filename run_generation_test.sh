#!/bin/bash
# Testar modelo treinado com geração de texto

echo "========================================================================"
echo "  Teste de Geração - Modelo Transformer Treinado"
echo "========================================================================"
echo ""

# Compilar
echo "Compilando..."
go build -o test_model_bin test_model_simple.go

if [ $? -ne 0 ]; then
    echo "Erro na compilação!"
    exit 1
fi

echo "Compilação concluída!"
echo ""

# Executar testes
echo "Executando testes de geração..."
echo ""
./test_model_bin config.trained.json

# Limpar
rm -f test_model_bin
