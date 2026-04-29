#!/usr/bin/env bash
set -euo pipefail

# LMCS LLM Training Pipeline Script
# Runs the full Python pipeline: download, merge, train, export

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAINING_DIR="${PROJECT_ROOT}/training_gpu"

echo "=========================================="
echo "  LMCS LLM - Training Pipeline"
echo "=========================================="
echo ""
echo "Project root: ${PROJECT_ROOT}"
echo "Training dir: ${TRAINING_DIR}"
echo ""

# Check Python virtual environment
if [[ -d "${TRAINING_DIR}/.venv" ]]; then
    echo "Activating virtual environment..."
    source "${TRAINING_DIR}/.venv/bin/activate"
else
    echo "WARNING: No .venv found in training_gpu/. Using system Python."
fi

# Run pipeline
cd "${TRAINING_DIR}"
echo "Starting pipeline..."
python pipeline.py "$@"

echo ""
echo "=========================================="
echo "  Training pipeline finished!"
echo "=========================================="
echo ""
echo "To start the server with the trained model:"
echo "  ./scripts/build-and-serve.sh"
