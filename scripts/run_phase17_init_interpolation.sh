#!/bin/bash
# Phase 17: Init Interpolation Sweep
#
# Train models with interpolated initialization between random and semantic.
# Measure calibration to find where the phase transition occurs.
#
# α = 0.00 → pure random init
# α = 0.25 → 25% semantic + 75% random
# α = 0.50 → 50% semantic + 50% random
# α = 0.75 → 75% semantic + 25% random
# α = 1.00 → pure semantic init

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

# Output directories
MODEL_DIR="models/unified"
EVAL_DIR="results/phase17_eval"
CALIB_DIR="results/phase17_calibration"
mkdir -p "$EVAL_DIR" "$CALIB_DIR"

echo "=== Phase 17: Init Interpolation Sweep ==="
echo ""

# Alpha values to test
ALPHAS=(0.00 0.25 0.50 0.75 1.00)
SEED=42

# ============================================================================
# Step 1: Train interpolated models
# ============================================================================
echo "=== Step 1: Training Interpolated Models ==="

for alpha in "${ALPHAS[@]}"; do
    alpha_int=$(echo "$alpha * 100" | bc | cut -d'.' -f1)
    alpha_str=$(printf "alpha%03d" "$alpha_int")
    model_path="$MODEL_DIR/semantic_${alpha_str}_seed${SEED}"

    if [ -d "$model_path" ]; then
        echo "SKIP: $alpha_str (model exists)"
        continue
    fi

    echo "TRAIN: α=$alpha ($alpha_str)"
    python src/train_k2.py \
        --channel dedicated \
        --token_string semantic \
        --init interpolated \
        --init_alpha "$alpha" \
        --seed "$SEED" \
        --output_dir "$model_path" \
        2>&1 | tail -20
    echo ""
done

# ============================================================================
# Step 2: Run early-exit evaluation with per-example margins
# ============================================================================
echo ""
echo "=== Step 2: Per-Example Evaluation ==="

for alpha in "${ALPHAS[@]}"; do
    alpha_int=$(echo "$alpha * 100" | bc | cut -d'.' -f1)
    alpha_str=$(printf "alpha%03d" "$alpha_int")
    model_path="$MODEL_DIR/semantic_${alpha_str}_seed${SEED}"
    output_file="$EVAL_DIR/${alpha_str}_seed${SEED}_per_example.json"

    if [ ! -d "$model_path" ]; then
        echo "SKIP: $alpha_str (no model)"
        continue
    fi

    if [ -f "$output_file" ]; then
        echo "SKIP: $alpha_str evaluation (exists)"
        continue
    fi

    echo "EVAL: α=$alpha ($alpha_str)"
    python src/early_exit.py \
        --model_path "$model_path" \
        --test_data data/test_rewritten.jsonl \
        --output "$output_file" \
        --save_per_example \
        2>&1 | tail -15
    echo ""
done

# ============================================================================
# Step 3: Run calibration analysis
# ============================================================================
echo ""
echo "=== Step 3: Calibration Analysis ==="

for alpha in "${ALPHAS[@]}"; do
    alpha_int=$(echo "$alpha * 100" | bc | cut -d'.' -f1)
    alpha_str=$(printf "alpha%03d" "$alpha_int")
    input_file="$EVAL_DIR/${alpha_str}_seed${SEED}_per_example.json"
    output_file="$CALIB_DIR/${alpha_str}_seed${SEED}_calibration.json"

    if [ ! -f "$input_file" ]; then
        echo "SKIP: $alpha_str calibration (no eval data)"
        continue
    fi

    if [ -f "$output_file" ]; then
        echo "SKIP: $alpha_str calibration (exists)"
        continue
    fi

    echo "CALIB: α=$alpha ($alpha_str)"
    python src/calibration_analysis.py \
        --results "$input_file" \
        --output "$output_file" \
        --layers "14,16,18,23"
done

# ============================================================================
# Step 4: Generate summary table
# ============================================================================
echo ""
echo "=== Step 4: Summary ==="

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

eval_dir = Path("results/phase17_eval")
calib_dir = Path("results/phase17_calibration")

alphas = [0.00, 0.25, 0.50, 0.75, 1.00]

print("=" * 90)
print("PHASE 17: INIT INTERPOLATION RESULTS")
print("=" * 90)
print(f"\n{'Alpha':<8} {'L14 ECE':<10} {'L16 ECE':<10} {'L18 ECE':<10} {'L23 ECE':<10} {'L14 AUC':<10} {'Full AUC':<10}")
print("-" * 78)

for alpha in alphas:
    alpha_str = f"alpha{int(alpha * 100):03d}"

    # Read calibration data
    calib_file = calib_dir / f"{alpha_str}_seed42_calibration.json"
    eval_file = eval_dir / f"{alpha_str}_seed42_per_example.json"

    if not calib_file.exists():
        print(f"{alpha:<8.2f} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        continue

    with open(calib_file) as f:
        calib = json.load(f)

    with open(eval_file) as f:
        evl = json.load(f)

    pl = calib.get("per_layer", {})
    ece_14 = pl.get("14", {}).get("ece", None)
    ece_16 = pl.get("16", {}).get("ece", None)
    ece_18 = pl.get("18", {}).get("ece", None)
    ece_23 = pl.get("23", {}).get("ece", None)

    auc_14 = evl.get("per_layer", {}).get("14", {}).get("auc", None)
    full_auc = evl.get("full_forward", {}).get("auc", None)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"

    print(f"{alpha:<8.2f} {fmt(ece_14):<10} {fmt(ece_16):<10} {fmt(ece_18):<10} {fmt(ece_23):<10} {fmt(auc_14):<10} {fmt(full_auc):<10}")

print("\n" + "=" * 90)
print("Interpretation:")
print("- Lower ECE = better calibration")
print("- High L14 AUC with low L14 ECE = good early-exit candidate")
print("- Look for phase transition where ECE jumps")
print("=" * 90)
PYTHON_SCRIPT

echo ""
echo "=== Phase 17 Complete ==="
echo "Models: $MODEL_DIR"
echo "Evaluations: $EVAL_DIR"
echo "Calibration: $CALIB_DIR"
