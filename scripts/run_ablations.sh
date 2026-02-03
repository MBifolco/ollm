#!/bin/bash
# Run ablation experiments (E2)

export HSA_OVERRIDE_GFX_VERSION=10.3.0

echo "========================================"
echo "ABLATION EXPERIMENTS (E2)"
echo "========================================"

SEED=42

echo ""
echo "--- Ablation 1: Baseline with 10 epochs ---"
python src/train_ablations.py --ablation baseline-10ep --seed $SEED 2>&1 | tail -10

echo ""
echo "--- Ablation 2: Random tokens ---"
python src/train_ablations.py --ablation random-token --seed $SEED 2>&1 | tail -10

echo ""
echo "--- Ablation 3: Single token ---"
python src/train_ablations.py --ablation single-token --seed $SEED 2>&1 | tail -10

echo ""
echo "========================================"
echo "EVALUATING ABLATIONS"
echo "========================================"
python src/eval_ablations.py --seed $SEED

echo ""
echo "Done!"
