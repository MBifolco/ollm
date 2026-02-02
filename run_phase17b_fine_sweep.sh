#!/bin/bash
# Phase 17b: Fine α sweep near the phase boundary
# Locate the exact transition point between α=0.50 and α=0.75

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

MODEL_DIR="models/unified"
EVAL_DIR="results/phase17_eval"
CALIB_DIR="results/phase17_calibration"
ADAPTIVE_DIR="results/phase17_adaptive"
mkdir -p "$EVAL_DIR" "$CALIB_DIR" "$ADAPTIVE_DIR"

echo "=== Phase 17b: Fine α Sweep (0.55-0.70) ==="
echo ""

# Fine-grained alpha values near the transition
ALPHAS=(0.55 0.60 0.65 0.70)
SEED=42

# Train, evaluate, and analyze each model
for alpha in "${ALPHAS[@]}"; do
    alpha_int=$(echo "$alpha * 100" | bc | cut -d'.' -f1)
    alpha_str=$(printf "alpha%03d" "$alpha_int")
    model_path="$MODEL_DIR/semantic_${alpha_str}_seed${SEED}"

    # Step 1: Train
    if [ ! -d "$model_path" ]; then
        echo "TRAIN: α=$alpha ($alpha_str)"
        python src/train_unified.py \
            --channel dedicated \
            --token_string semantic \
            --init interpolated \
            --init_alpha "$alpha" \
            --seed "$SEED" \
            --output_dir "$model_path" \
            2>&1 | tail -10
        echo ""
    else
        echo "SKIP TRAIN: $alpha_str (exists)"
    fi

    # Step 2: Evaluate with per-example margins
    eval_file="$EVAL_DIR/${alpha_str}_seed${SEED}_per_example.json"
    if [ ! -f "$eval_file" ]; then
        echo "EVAL: α=$alpha"
        python src/early_exit.py \
            --model_path "$model_path" \
            --test_data data/test_rewritten.jsonl \
            --output "$eval_file" \
            --save_per_example \
            2>&1 | tail -10
        echo ""
    else
        echo "SKIP EVAL: $alpha_str (exists)"
    fi

    # Step 3: Calibration analysis
    calib_file="$CALIB_DIR/${alpha_str}_seed${SEED}_calibration.json"
    if [ ! -f "$calib_file" ]; then
        echo "CALIB: α=$alpha"
        python src/calibration_analysis.py \
            --results "$eval_file" \
            --output "$calib_file" \
            --layers "14,16,18,23" \
            2>&1 | grep -E "ECE|Layer [0-9]"
        echo ""
    else
        echo "SKIP CALIB: $alpha_str (exists)"
    fi

    # Step 4: Adaptive exit analysis
    adaptive_file="$ADAPTIVE_DIR/${alpha_str}_adaptive.json"
    if [ ! -f "$adaptive_file" ]; then
        echo "ADAPTIVE: α=$alpha"
        python src/analyze_adaptive_exit.py \
            --results "$eval_file" \
            --output "$adaptive_file" \
            --candidate_layers "14,16,18,23" \
            2>&1 | grep -E "0.80|0.90|Pareto"
        echo ""
    else
        echo "SKIP ADAPTIVE: $alpha_str (exists)"
    fi
done

# Summary table
echo ""
echo "=== Phase 17b Summary ==="
python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

calib_dir = Path("results/phase17_calibration")
eval_dir = Path("results/phase17_eval")
adaptive_dir = Path("results/phase17_adaptive")

# All alphas including original sweep
alphas = [0.00, 0.25, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.00]

print("=" * 100)
print("PHASE 17b: FINE-GRAINED TRANSITION ANALYSIS")
print("=" * 100)
print(f"\n{'Alpha':<8} {'L14 ECE':<10} {'L14 AUC':<10} {'Full AUC':<10} {'τ=0.80 AUC':<12} {'τ=0.80 Speed':<12}")
print("-" * 72)

for alpha in alphas:
    alpha_str = f"alpha{int(alpha * 100):03d}"

    calib_file = calib_dir / f"{alpha_str}_seed42_calibration.json"
    eval_file = eval_dir / f"{alpha_str}_seed42_per_example.json"
    adaptive_file = adaptive_dir / f"{alpha_str}_adaptive.json"

    row = [f"{alpha:.2f}"]

    # Calibration ECE
    if calib_file.exists():
        with open(calib_file) as f:
            c = json.load(f)
        ece = c.get("per_layer", {}).get("14", {}).get("ece")
        row.append(f"{ece:.4f}" if ece else "N/A")
    else:
        row.append("N/A")

    # Eval AUC
    if eval_file.exists():
        with open(eval_file) as f:
            e = json.load(f)
        l14_auc = e.get("per_layer", {}).get("14", {}).get("auc")
        full_auc = e.get("full_forward", {}).get("auc")
        row.append(f"{l14_auc:.4f}" if l14_auc else "N/A")
        row.append(f"{full_auc:.4f}" if full_auc else "N/A")
    else:
        row.extend(["N/A", "N/A"])

    # Adaptive exit at τ=0.80
    if adaptive_file.exists():
        with open(adaptive_file) as f:
            a = json.load(f)
        for kt in a.get("key_thresholds", []):
            if abs(kt.get("tau", 0) - 0.80) < 0.02:
                row.append(f"{kt['auc']:.4f}" if kt.get('auc') else "N/A")
                row.append(f"{kt['speedup']:.2f}x")
                break
        else:
            row.extend(["N/A", "N/A"])
    else:
        row.extend(["N/A", "N/A"])

    print(f"{row[0]:<8} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<12} {row[5]:<12}")

print("\n" + "=" * 100)
print("Look for the transition point where L14 AUC and τ=0.80 AUC jump together")
print("=" * 100)
PYTHON_SCRIPT

echo ""
echo "=== Phase 17b Complete ==="
