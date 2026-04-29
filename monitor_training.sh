#!/bin/bash
# Quick Training Monitor
# Shows real-time training progress

echo "📊 Training Progress Monitor"
echo "============================"
echo

while true; do
    # Clear screen
    clear
    
    echo "📊 Training Progress Monitor - $(date '+%H:%M:%S')"
    echo "============================"
    echo
    
    # Check if training is running
    if pgrep -f "train_gpu.py" > /dev/null; then
        echo "Status: ✅ Training running"
        echo "PID: $(pgrep -f 'train_gpu.py')"
    else
        echo "Status: ❌ Training not running"
    fi
    
    echo
    
    # Show checkpoint info
    if [ -f "training_gpu/checkpoints/model_best.pt" ]; then
        LAST_MOD=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" training_gpu/checkpoints/model_best.pt)
        SIZE=$(du -h training_gpu/checkpoints/model_best.pt | cut -f1)
        echo "Latest checkpoint:"
        echo "  File: training_gpu/checkpoints/model_best.pt"
        echo "  Size: $SIZE"
        echo "  Modified: $LAST_MOD"
    else
        echo "Latest checkpoint: Not found"
    fi
    
    echo
    
    # Check if training complete
    if [ -f "training_gpu/checkpoints/training_complete.txt" ]; then
        echo "✅ Training COMPLETE!"
        echo
        cat training_gpu/checkpoints/training_complete.txt
        echo
        echo "Next step:"
        echo "  ./training_gpu/auto_train_and_export.sh"
        exit 0
    fi
    
    echo "Monitoring... (Ctrl+C to stop)"
    sleep 10
done
