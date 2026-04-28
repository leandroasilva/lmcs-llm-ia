#!/bin/bash
# LMCS LLM IA - Script helper para diferentes ambientes
# Uso: ./run.sh [environment] [command]

set -e

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Ambiente padrão
ENV=${1:-development}
COMMAND=${2:-help}

# Converter para uppercase (compatível com macOS)
ENV_UPPER=$(echo "$ENV" | tr '[:lower:]' '[:upper:]')

# Configurar variáveis de ambiente baseado no ambiente
case $ENV in
    prod|production)
        export LMCS_ENV=production
        export LMCS_EPOCHS=500
        export LMCS_D_MODEL=512
        export LMCS_N_HEADS=8
        export LMCS_NUM_LAYERS=6
        export LMCS_BATCH_SIZE=32
        export LMCS_FF_HIDDEN=1024
        export LMCS_HOST=0.0.0.0
        CONFIG_FILE="config.production.json"
        ;;
    staging|stage)
        export LMCS_ENV=staging
        export LMCS_EPOCHS=200
        export LMCS_D_MODEL=256
        export LMCS_N_HEADS=4
        export LMCS_NUM_LAYERS=4
        export LMCS_BATCH_SIZE=16
        export LMCS_FF_HIDDEN=512
        export LMCS_DROPOUT_RATE=0.15
        CONFIG_FILE="config.staging.json"
        ;;
    test|testing)
        export LMCS_ENV=test
        export LMCS_EPOCHS=10
        export LMCS_D_MODEL=64
        export LMCS_N_HEADS=4
        export LMCS_NUM_LAYERS=2
        export LMCS_MAX_SEQ_LEN=128
        export LMCS_FF_HIDDEN=128
        export LMCS_MAX_VOCAB=1000
        export LMCS_BATCH_SIZE=8
        CONFIG_FILE="config.test.json"
        ;;
    dev|development|*)
        export LMCS_ENV=development
        export LMCS_EPOCHS=300
        export LMCS_D_MODEL=512
        export LMCS_N_HEADS=8
        export LMCS_NUM_LAYERS=6
        export LMCS_BATCH_SIZE=16
        export LMCS_FF_HIDDEN=1024
        CONFIG_FILE="config.json"
        ;;
esac

# Compilar se necessário
if [ ! -f "./lmcs-llm" ] || [ "./cmd/lmcs-llm/main.go" -nt "./lmcs-llm" ]; then
    echo -e "${BLUE}🔨 Compilando...${NC}"
    go build -o lmcs-llm ./cmd/lmcs-llm/
    echo -e "${GREEN}✅ Compilação concluída${NC}"
fi

echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${BLUE}  LMCS LLM IA - Ambiente: ${YELLOW}${ENV_UPPER}${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo -e "${GREEN}📊 Configuração:${NC}"
echo -e "   Environment: ${YELLOW}$LMCS_ENV${NC}"
echo -e "   Épocas: ${YELLOW}$LMCS_EPOCHS${NC}"
echo -e "   D-Model: ${YELLOW}$LMCS_D_MODEL${NC}"
echo -e "   Layers: ${YELLOW}$LMCS_NUM_LAYERS${NC}"
echo -e "   Heads: ${YELLOW}$LMCS_N_HEADS${NC}"
echo -e "   Batch: ${YELLOW}$LMCS_BATCH_SIZE${NC}"
echo -e "   Config: ${YELLOW}$CONFIG_FILE${NC}"
echo -e "${BLUE}═══════════════════════════════════════${NC}"
echo ""

# Executar comando
case $COMMAND in
    train|t)
        echo -e "${GREEN}🚀 Iniciando treinamento...${NC}"
        ./lmcs-llm train --config $CONFIG_FILE
        ;;
    serve|s)
        echo -e "${GREEN}🌐 Iniciando servidor...${NC}"
        ./lmcs-llm serve --config $CONFIG_FILE
        ;;
    download|d)
        echo -e "${GREEN}📥 Baixando dataset enriquecido...${NC}"
        ./lmcs-llm download --enriched
        ;;
    help|h|*)
        echo -e "${YELLOW}Uso:${NC}"
        echo -e "  ./run.sh [environment] [command]"
        echo ""
        echo -e "${YELLOW}Ambientes:${NC}"
        echo -e "  development (padrão)  - Configuração de desenvolvimento"
        echo -e "  production            - Configuração de produção"
        echo -e "  staging               - Configuração de homologação"
        echo -e "  test                  - Configuração de testes rápidos"
        echo ""
        echo -e "${YELLOW}Comandos:${NC}"
        echo -e "  train, t              - Treinar modelo"
        echo -e "  serve, s              - Iniciar servidor"
        echo -e "  download, d           - Baixar dataset"
        echo -e "  help, h               - Mostrar esta ajuda"
        echo ""
        echo -e "${YELLOW}Exemplos:${NC}"
        echo -e "  ./run.sh production train     # Treinar em produção"
        echo -e "  ./run.sh staging serve        # Servir em staging"
        echo -e "  ./run.sh test train           # Teste rápido"
        echo -e "  ./run.sh                      # Mostrar ajuda"
        ;;
esac
