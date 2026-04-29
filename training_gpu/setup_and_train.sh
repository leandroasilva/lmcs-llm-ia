#!/bin/bash
# Setup and Train - GPU Training for LMCS LLM
# Usage: ./setup_and_train.sh

set -e

echo "🚀 GPU Training Setup for LMCS LLM"
echo "=================================="
echo

# Check if in correct directory
if [ ! -f "train_gpu.py" ]; then
    echo "❌ Error: Run this script from the training_gpu/ directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
if [ ! -f ".venv/.installed" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    touch .venv/.installed
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

echo
echo "🎯 Starting GPU Training"
echo "========================"
echo

# Run training with passed arguments or defaults
python train_gpu.py "$@"

echo
echo "✅ Training completed!"
echo "💡 To use the trained model in Go:"
echo "   python train_gpu.py --export-go ./checkpoints/model_best.pt --output-path ../config.trained.json"
echo "   cd .."
echo "   ./lmcs-llm serve --config config.trained.json"
