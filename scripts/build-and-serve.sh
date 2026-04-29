#!/usr/bin/env bash
set -euo pipefail

# LMCS LLM Build & Serve Script
# Builds the React frontend, compiles the Go server, and starts serving

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FRONTEND_DIR="${PROJECT_ROOT}/frontend"

echo "=========================================="
echo "  LMCS LLM - Build & Serve"
echo "=========================================="
echo ""

# Determine config path
CONFIG_PATH="${1:-config.trained.json}"
if [[ ! "${CONFIG_PATH}" = /* ]]; then
    CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "ERROR: Model/config file not found: ${CONFIG_PATH}"
    echo ""
    echo "Please train a model first:"
    echo "  ./scripts/train.sh"
    exit 1
fi

# Build frontend
echo "Building frontend..."
cd "${FRONTEND_DIR}"
if [[ ! -d "node_modules" ]]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run build

# Build Go binary
echo ""
echo "Building Go server..."
cd "${PROJECT_ROOT}"
go build -o lmcs-llm ./cmd/lmcs-llm
if [[ ! -f "${PROJECT_ROOT}/lmcs-llm" ]]; then
    echo "ERROR: Go build failed."
    exit 1
fi

# Start server
echo ""
echo "=========================================="
echo "  Starting LMCS LLM Server"
echo "=========================================="
echo "  Config: ${CONFIG_PATH}"
echo "  URL:    http://localhost:8080"
echo "=========================================="
echo ""

exec "${PROJECT_ROOT}/lmcs-llm" serve --config "${CONFIG_PATH}"
