#!/bin/bash
# Complete Training & Export Pipeline
# ====================================
# This script:
# 1. Checks if training is already running
# 2. Monitors training progress
# 3. Auto-exports to Go when complete
# 4. Generates usage documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🚀 LMCS LLM - Complete Training & Export Pipeline"
echo "=================================================="
echo

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}📦 Setting up Python environment...${NC}"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo -e "${GREEN}✓ Python environment ready${NC}"
echo

# Check if training is already running
if pgrep -f "train_gpu.py" > /dev/null; then
    echo -e "${YELLOW}⚠️  Training is already running!${NC}"
    echo "   PID: $(pgrep -f 'train_gpu.py')"
    echo "   Monitoring for completion..."
    echo
else
    echo -e "${GREEN}🎯 Starting training...${NC}"
    echo "   Device: MPS (Apple Silicon GPU)"
    echo "   Epochs: 300"
    echo "   Batch size: 16"
    echo "   Max seq length: 128"
    echo
    
    # Start training in background
    python train_gpu.py \
        --device auto \
        --epochs 300 \
        --batch-size 16 \
        --learning-rate 1e-4 \
        --max-seq-len 128 \
        &
    
    TRAIN_PID=$!
    echo -e "${GREEN}✓ Training started (PID: $TRAIN_PID)${NC}"
    echo
fi

echo "📊 Monitoring training progress..."
echo "=================================================="
echo

# Monitor training
LAST_UPDATE=$(date +%s)
START_TIME=$(date +%s)

while true; do
    # Check if training complete marker exists
    if [ -f "checkpoints/training_complete.txt" ]; then
        echo
        echo -e "${GREEN}==================================================${NC}"
        echo -e "${GREEN}✅ Training Completed!${NC}"
        echo -e "${GREEN}==================================================${NC}"
        echo
        cat checkpoints/training_complete.txt
        echo
        break
    fi
    
    # Check if training process is still running
    if pgrep -f "train_gpu.py" > /dev/null; then
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))
        
        # Update every 60 seconds
        if [ $((CURRENT_TIME - LAST_UPDATE)) -ge 60 ]; then
            echo -e "${YELLOW}[$(date '+%H:%M:%S')] Training running... (${ELAPSED}s elapsed)${NC}"
            
            # Show latest checkpoint info
            if [ -f "checkpoints/model_best.pt" ]; then
                CHECKPOINT_AGE=$((CURRENT_TIME - $(stat -f %m checkpoints/model_best.pt 2>/dev/null || echo $CURRENT_TIME)))
                echo "   Last checkpoint: ${CHECKPOINT_AGE}s ago"
            fi
            
            LAST_UPDATE=$CURRENT_TIME
        fi
        
        sleep 10
    else
        echo
        echo -e "${RED}❌ Training process stopped unexpectedly${NC}"
        echo "   Checking if model was saved..."
        
        if [ -f "checkpoints/model_best.pt" ]; then
            echo "   ✓ Checkpoint found, attempting export..."
        else
            echo "   ✗ No checkpoint found"
            exit 1
        fi
        break
    fi
done

echo
echo "📦 Exporting model to Go format..."
echo "=================================================="

# Export to Go
python -c "
import sys
sys.path.insert(0, '.')
from monitor_and_export import export_to_go, generate_go_usage_guide

try:
    export_path = export_to_go(
        checkpoint_path='./checkpoints/model_best.pt',
        output_path='../config.trained.json'
    )
    generate_go_usage_guide(export_path)
    print()
    print('✅ Export successful!')
except Exception as e:
    print(f'❌ Export failed: {e}')
    sys.exit(1)
"

echo
echo "📝 Testing exported model..."
echo "=================================================="

# Verify JSON is valid
if python -m json.tool ../config.trained.json > /dev/null 2>&1; then
    echo -e "${GREEN}✓ JSON is valid${NC}"
    
    # Show file info
    FILE_SIZE=$(du -h ../config.trained.json | cut -f1)
    echo "  File: ../config.trained.json"
    echo "  Size: $FILE_SIZE"
    
    # Show config summary
    python -c "
import json
with open('../config.trained.json') as f:
    data = json.load(f)
    
print()
print('📊 Model Summary:')
print('  Parameters:', data['metadata']['total_parameters'])
print('  Final epoch:', data['metadata']['final_epoch'])
print('  Final loss:', data['metadata']['final_loss'])
print('  Framework:', data['metadata']['framework'])
"
else
    echo -e "${RED}✗ JSON validation failed${NC}"
    exit 1
fi

echo
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}🎉 Pipeline Complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo
echo "✅ Model trained and exported successfully"
echo
echo "📁 Generated files:"
echo "   • checkpoints/model_best.pt  - PyTorch checkpoint"
echo "   • config.trained.json        - Go-compatible model"
echo "   • docs/MODEL_EXPORT_GUIDE.md - Usage guide"
echo
echo "🚀 Next steps:"
echo "   cd .."
echo "   ./lmcs-llm serve --config config.trained.json"
echo
echo "📖 Documentation:"
echo "   cat docs/MODEL_EXPORT_GUIDE.md"
echo
