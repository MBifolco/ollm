#!/bin/bash
# Phase 16: Adaptive Early-Exit Analysis
# Run per-example evaluation and generate AUC-latency curves

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

# Output directories
EVAL_DIR="results/adaptive_eval"
ANALYSIS_DIR="results/adaptive_analysis"
mkdir -p "$EVAL_DIR" "$ANALYSIS_DIR"

echo "=== Phase 16: Adaptive Early-Exit Analysis ==="
echo ""

# Step 1: Run evaluation with per-example margins for each model type
echo "=== Step 1: Evaluating with per-example margins ==="

# Semantic model (uses early_exit.py)
SEMANTIC_OUT="$EVAL_DIR/semantic_seed42_per_example.json"
if [ ! -f "$SEMANTIC_OUT" ]; then
    echo "EVAL: semantic_seed42"
    python src/early_exit.py \
        --model_path models/unified/semantic_seed42 \
        --test_data data/test_rewritten.jsonl \
        --output "$SEMANTIC_OUT" \
        --save_per_example \
        2>&1 | tail -20
    echo ""
else
    echo "SKIP: semantic_seed42 (already exists)"
fi

# Random model (uses early_exit_ablations.py with model_type=random)
RANDOM_OUT="$EVAL_DIR/random_seed42_per_example.json"
if [ ! -f "$RANDOM_OUT" ]; then
    echo "EVAL: random_seed42"
    python src/early_exit_ablations.py \
        --model_path models/unified/random_seed42 \
        --model_type random \
        --test_data data/test_rewritten.jsonl \
        --output "$RANDOM_OUT" \
        --save_per_example \
        2>&1 | tail -20
    echo ""
else
    echo "SKIP: random_seed42 (already exists)"
fi

# R/N vocab model (uses early_exit_rn.py - needs per_example support added)
RN_OUT="$EVAL_DIR/rn_vocab_seed42_per_example.json"
if [ ! -f "$RN_OUT" ]; then
    echo "EVAL: rn_vocab_seed42"
    # For now, use semantic script since rn_vocab format is similar
    # Note: This may need custom handling later
    python src/early_exit_rn.py \
        --model_path models/unified/rn_vocab_seed42 \
        --test_data data/test_rewritten.jsonl \
        --output "$RN_OUT" \
        2>&1 | tail -20
    echo "  Note: rn_vocab per-example not yet supported, skipping adaptive analysis"
    echo ""
else
    echo "SKIP: rn_vocab_seed42 (already exists)"
fi

# Step 2: Run adaptive exit analysis
echo ""
echo "=== Step 2: Analyzing adaptive exit policies ==="

for model_name in semantic random; do
    input_file="$EVAL_DIR/${model_name}_seed42_per_example.json"
    output_file="$ANALYSIS_DIR/${model_name}_seed42_adaptive.json"

    if [ ! -f "$input_file" ]; then
        echo "SKIP: $model_name (no per-example data)"
        continue
    fi

    # Check if per_example key exists
    has_per_example=$(python3 -c "import json; d=json.load(open('$input_file')); print('yes' if 'per_example' in d else 'no')")
    if [ "$has_per_example" != "yes" ]; then
        echo "SKIP: $model_name (no per_example in JSON)"
        continue
    fi

    echo "ANALYZE: $model_name"
    python src/analyze_adaptive_exit.py \
        --results "$input_file" \
        --output "$output_file" \
        --candidate_layers "14,16,18,23" \
        --min_exit_layer 14

    echo ""
done

echo "=== Analysis Complete ==="
echo "Per-example results: $EVAL_DIR"
echo "Adaptive analysis: $ANALYSIS_DIR"
