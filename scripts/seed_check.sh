#!/bin/bash
set -e

for SEED in 0 1 2; do
    echo ""
    echo "======================================================"
    echo "TRAINING WITH SEED $SEED"
    echo "======================================================"
    
    # Train baseline
    echo "Training baseline seed $SEED..."
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 src/train_baseline.py \
        --output_dir models/baseline_track1_seed$SEED \
        --seed $SEED 2>&1 | tail -5
    
    # Train internal token
    echo "Training internal token seed $SEED..."
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python3 src/train_internal_token.py \
        --output_dir models/internal_token_track1_seed$SEED \
        --seed $SEED 2>&1 | tail -5
done

echo ""
echo "======================================================"
echo "ALL TRAINING COMPLETE - RUNNING PROBING"
echo "======================================================"
