#!/bin/bash
# Phase 16b: Robustness Checks for Adaptive Early-Exit
#
# Runs calibration analysis and adaptive exit with normalization
# across all seeds to verify calibration claims survive peer review.

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

# Output directories
CALIB_DIR="results/calibration"
ADAPTIVE_DIR="results/adaptive_analysis"
mkdir -p "$CALIB_DIR" "$ADAPTIVE_DIR"

echo "=== Phase 16b: Robustness Checks ==="
echo ""

# Models and seeds to analyze
MODELS=("semantic" "random")
SEEDS=(42 123 456)
CANDIDATE_LAYERS="14,16,18,23"
EVAL_DIR="results/adaptive_eval"

# ============================================================================
# Step 0: Generate per-example data for all seeds (if missing)
# ============================================================================
echo "=== Step 0: Generate Per-Example Margins ==="

for seed in "${SEEDS[@]}"; do
    # Semantic model
    SEMANTIC_OUT="$EVAL_DIR/semantic_seed${seed}_per_example.json"
    if [ ! -f "$SEMANTIC_OUT" ]; then
        echo "EVAL: semantic_seed${seed}"
        python src/early_exit.py \
            --model_path models/unified/semantic_seed${seed} \
            --test_data data/test_rewritten.jsonl \
            --output "$SEMANTIC_OUT" \
            --save_per_example \
            2>&1 | tail -15
    else
        echo "SKIP: semantic_seed${seed} (already exists)"
    fi

    # Random model
    RANDOM_OUT="$EVAL_DIR/random_seed${seed}_per_example.json"
    if [ ! -f "$RANDOM_OUT" ]; then
        echo "EVAL: random_seed${seed}"
        python src/early_exit_ablations.py \
            --model_path models/unified/random_seed${seed} \
            --model_type random \
            --test_data data/test_rewritten.jsonl \
            --output "$RANDOM_OUT" \
            --save_per_example \
            2>&1 | tail -15
    else
        echo "SKIP: random_seed${seed} (already exists)"
    fi
done

echo ""

# ============================================================================
# Step 1: Run calibration analysis for each model/seed
# ============================================================================
echo "=== Step 1: Calibration Analysis (ECE, Brier, Reliability Curves) ==="

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        input_file="$EVAL_DIR/${model}_seed${seed}_per_example.json"
        output_file="$CALIB_DIR/${model}_seed${seed}_calibration.json"

        if [ ! -f "$input_file" ]; then
            echo "SKIP: $model seed$seed (no per-example data at $input_file)"
            continue
        fi

        if [ -f "$output_file" ]; then
            echo "SKIP: $model seed$seed calibration (already exists)"
            continue
        fi

        echo "CALIB: $model seed$seed"
        python src/calibration_analysis.py \
            --results "$input_file" \
            --output "$output_file" \
            --layers "$CANDIDATE_LAYERS" \
            --n_bins 10
    done
done

echo ""

# ============================================================================
# Step 2: Adaptive exit analysis with different normalization methods
# ============================================================================
echo "=== Step 2: Adaptive Exit with Normalization ==="

NORMALIZE_METHODS=("none" "zscore" "temperature")
CALIB_SPLIT=0.3

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        input_file="$EVAL_DIR/${model}_seed${seed}_per_example.json"

        if [ ! -f "$input_file" ]; then
            echo "SKIP: $model seed$seed (no per-example data)"
            continue
        fi

        for norm in "${NORMALIZE_METHODS[@]}"; do
            if [ "$norm" == "none" ]; then
                output_file="$ADAPTIVE_DIR/${model}_seed${seed}_adaptive.json"
            else
                output_file="$ADAPTIVE_DIR/${model}_seed${seed}_adaptive_${norm}.json"
            fi

            if [ -f "$output_file" ]; then
                echo "SKIP: $model seed$seed norm=$norm (already exists)"
                continue
            fi

            echo "ADAPTIVE: $model seed$seed norm=$norm"
            python src/analyze_adaptive_exit.py \
                --results "$input_file" \
                --output "$output_file" \
                --candidate_layers "$CANDIDATE_LAYERS" \
                --min_exit_layer 14 \
                --normalize "$norm" \
                --calib_split $CALIB_SPLIT \
                --seed "$seed"
        done
    done
done

echo ""

# ============================================================================
# Step 3: Generate summary table
# ============================================================================
echo "=== Step 3: Summary Table ==="
echo ""

python3 << 'PYTHON_SCRIPT'
import json
from pathlib import Path

calib_dir = Path("results/calibration")
adaptive_dir = Path("results/adaptive_analysis")

# Calibration summary
print("=" * 80)
print("CALIBRATION SUMMARY (ECE per layer)")
print("=" * 80)
print(f"{'Model':<12} {'Seed':<6} {'L14 ECE':<10} {'L16 ECE':<10} {'L18 ECE':<10} {'L23 ECE':<10}")
print("-" * 60)

for model in ["semantic", "random"]:
    for seed in [42, 123, 456]:
        fpath = calib_dir / f"{model}_seed{seed}_calibration.json"
        if not fpath.exists():
            print(f"{model:<12} {seed:<6} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        with open(fpath) as f:
            data = json.load(f)

        pl = data.get("per_layer", {})
        eces = []
        for layer in ["14", "16", "18", "23"]:
            if layer in pl and "ece" in pl[layer]:
                eces.append(f"{pl[layer]['ece']:.4f}")
            else:
                eces.append("N/A")

        print(f"{model:<12} {seed:<6} {eces[0]:<10} {eces[1]:<10} {eces[2]:<10} {eces[3]:<10}")

# Adaptive exit summary
print("\n" + "=" * 80)
print("ADAPTIVE EXIT SUMMARY (tau=0.90)")
print("=" * 80)
print(f"{'Model':<12} {'Seed':<6} {'Norm':<12} {'AUC':<10} {'Speedup':<10} {'Mean Layer':<12}")
print("-" * 70)

for model in ["semantic", "random"]:
    for seed in [42, 123, 456]:
        for norm in ["none", "zscore", "temperature"]:
            if norm == "none":
                fpath = adaptive_dir / f"{model}_seed{seed}_adaptive.json"
            else:
                fpath = adaptive_dir / f"{model}_seed{seed}_adaptive_{norm}.json"

            if not fpath.exists():
                print(f"{model:<12} {seed:<6} {norm:<12} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
                continue

            with open(fpath) as f:
                data = json.load(f)

            # Find tau=0.90 threshold
            kt = None
            for k in data.get("key_thresholds", []):
                if abs(k.get("tau", 0) - 0.90) < 0.01:
                    kt = k
                    break

            if kt:
                auc = f"{kt['auc']:.4f}" if kt.get('auc') else "N/A"
                speedup = f"{kt['speedup']:.2f}x"
                mean_layer = f"{kt['mean_exit_layer']:.1f}"
            else:
                auc = speedup = mean_layer = "N/A"

            print(f"{model:<12} {seed:<6} {norm:<12} {auc:<10} {speedup:<10} {mean_layer:<12}")

print("\n" + "=" * 80)
PYTHON_SCRIPT

echo ""
echo "=== Phase 16b Complete ==="
echo "Calibration results: $CALIB_DIR"
echo "Adaptive results: $ADAPTIVE_DIR"
