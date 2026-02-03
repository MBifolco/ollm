#!/bin/bash
# Retrain Triad Experiment - uses existing training scripts unchanged
# Train-O, Train-R, Train-M × Baseline, Token × Seeds 0,1,2

export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "========================================"
echo "RETRAIN TRIAD EXPERIMENT"
echo "========================================"

# Training configurations
TRAIN_SETS=("O" "R" "M")
SEEDS=(0 1 2)

for TRAIN in "${TRAIN_SETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training on data_${TRAIN}/"
    echo "========================================"

    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- Seed ${SEED} ---"

        # Train baseline
        BASELINE_DIR="models/triad/baseline_${TRAIN}_seed${SEED}"
        if [ ! -d "$BASELINE_DIR" ]; then
            echo "Training baseline -> ${BASELINE_DIR}"
            python src/train_baseline.py \
                --data_dir "data_${TRAIN}" \
                --output_dir "$BASELINE_DIR" \
                --seed "$SEED" 2>&1 | tail -5
        else
            echo "Baseline exists: ${BASELINE_DIR}"
        fi

        # Train token model
        TOKEN_DIR="models/triad/token_${TRAIN}_seed${SEED}"
        if [ ! -d "$TOKEN_DIR" ]; then
            echo "Training token -> ${TOKEN_DIR}"
            python src/train_internal_token.py \
                --data_dir "data_${TRAIN}" \
                --output_dir "$TOKEN_DIR" \
                --seed "$SEED" 2>&1 | tail -5
        else
            echo "Token exists: ${TOKEN_DIR}"
        fi
    done
done

echo ""
echo "========================================"
echo "TRAINING COMPLETE"
echo "========================================"
echo "Models saved to models/triad/"
ls -la models/triad/
