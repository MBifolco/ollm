#!/bin/bash
# Phase 15: Unified Training Experiment Matrix
# Run all DDC variants with standardized hyperparameters

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

# Base output directory
OUT_DIR="models/unified"
mkdir -p "$OUT_DIR"

echo "=== Phase 15: Unified Training Experiment Matrix ==="
echo "Output directory: $OUT_DIR"
echo ""

# Seeds for replication
SEEDS=(42 123 456)

# Run a single experiment
run_experiment() {
    local name=$1
    local channel=$2
    local token_string=$3
    local init=$4
    local init_lm_head=$5
    local seed=$6

    local model_name="${name}_seed${seed}"
    local output_dir="${OUT_DIR}/${model_name}"

    if [ -d "$output_dir" ]; then
        echo "SKIP: $model_name already exists"
        return 0
    fi

    echo "TRAIN: $model_name (channel=$channel, token=$token_string, init=$init, lm_head=$init_lm_head)"

    python src/train_k2.py \
        --channel "$channel" \
        --token_string "$token_string" \
        --init "$init" \
        --init_lm_head "$init_lm_head" \
        --train_set M \
        --seed "$seed" \
        --output_dir "$output_dir" \
        2>&1 | tail -20

    echo "DONE: $model_name"
    echo ""
}

# Experiment matrix:
# 1. dedicated + semantic string + semantic init (original semantic)
# 2. dedicated + semantic string + random init (semantic-randinit)
# 3. dedicated + random string + random init (original random)
# 4. dedicated + random string + semantic init (random-seminit)
# 5. rn_vocab baseline
# 6. single baseline (negative control)

for seed in "${SEEDS[@]}"; do
    echo "=== Seed $seed ==="

    # Core dedicated channel experiments
    run_experiment "semantic" dedicated semantic semantic 1 "$seed"
    run_experiment "semantic-randinit" dedicated semantic random 0 "$seed"
    run_experiment "random" dedicated random random 0 "$seed"
    run_experiment "random-seminit" dedicated random semantic 1 "$seed"

    # Baselines
    run_experiment "rn_vocab" rn_vocab semantic semantic 0 "$seed"
    run_experiment "single" single semantic semantic 0 "$seed"
done

echo "=== All experiments complete ==="
echo "Models saved to: $OUT_DIR"
