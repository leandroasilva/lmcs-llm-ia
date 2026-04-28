#!/bin/bash

# Script para rodar todos os testes do LMCS LLM IA

echo "================================================"
echo "  LMCS LLM IA - Test Suite"
echo "================================================"
echo ""

# Cores
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PASS=0
FAIL=0

# Função para rodar testes
run_tests() {
    local package=$1
    local description=$2
    
    echo -e "${CYAN}Testing: ${description}${NC}"
    echo "Package: $package"
    echo "---"
    
    if go test "$package" -v -timeout 30s 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# 1. Tokenizer Tests
run_tests "./internal/tokenizer/" "BPE & WordPiece Tokenizer"

# 2. Model Tests
run_tests "./internal/model/" "Transformer Model"

# 3. Evaluation Tests
run_tests "./internal/evaluation/" "Evaluation & Cross-Validation"

# Resumo
echo "================================================"
echo "  Test Summary"
echo "================================================"
echo ""
echo -e "${GREEN}Passed: $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed: $FAIL${NC}"
else
    echo -e "Failed: 0"
fi
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
