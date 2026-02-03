#!/bin/bash
# Evaluate all unified models with early-exit on Test-R

set -e

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTHONPATH=/home/biff/eng/ollm:$PYTHONPATH

MODEL_DIR="models/unified"
OUTPUT_DIR="results/unified_early_exit"
mkdir -p "$OUTPUT_DIR"

echo "=== Evaluating Unified Models ==="
echo "Model directory: $MODEL_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# List all models
for model_path in "$MODEL_DIR"/*_seed*; do
    if [ ! -d "$model_path" ]; then
        continue
    fi

    model_name=$(basename "$model_path")
    output_file="$OUTPUT_DIR/${model_name}_testR.json"

    if [ -f "$output_file" ]; then
        echo "SKIP: $model_name (already evaluated)"
        continue
    fi

    echo "EVAL: $model_name"

    # Read the training config to determine model type
    config_file="$model_path/training_config.json"
    if [ -f "$config_file" ]; then
        channel=$(python -c "import json; print(json.load(open('$config_file'))['channel'])")
        token_string=$(python -c "import json; print(json.load(open('$config_file'))['token_string'])")
    else
        echo "  WARNING: No config found, assuming semantic tokens"
        channel="dedicated"
        token_string="semantic"
    fi

    # Choose evaluation script based on channel/token type
    if [ "$channel" = "single" ]; then
        # Single token baseline
        python src/early_exit_ablations.py \
            --model_path "$model_path" \
            --model_type single \
            --test_data data/test_rewritten.jsonl \
            --output "$output_file" 2>&1 | tail -10
    elif [ "$channel" = "rn_vocab" ]; then
        # R/N baseline using existing vocab tokens
        python src/early_exit_rn.py \
            --model_path "$model_path" \
            --test_data data/test_rewritten.jsonl \
            --output "$output_file" 2>&1 | tail -10
    elif [ "$token_string" = "random" ]; then
        # Random tokens (RAND_A/RAND_B)
        python src/early_exit_ablations.py \
            --model_path "$model_path" \
            --model_type random \
            --test_data data/test_rewritten.jsonl \
            --output "$output_file" 2>&1 | tail -10
    else
        # Semantic tokens (LOVE_ROM/LOVE_NONROM)
        python src/early_exit.py \
            --model_path "$model_path" \
            --test_data data/test_rewritten.jsonl \
            --output "$output_file" 2>&1 | tail -10
    fi

    echo "  Saved: $output_file"
    echo ""
done

echo "=== Evaluation Complete ==="
echo ""

# Generate summary table
echo "=== Summary Table ==="
echo "Model | Full AUC | L16 AUC | L20 AUC | Crystallization"
echo "----- | -------- | ------- | ------- | ---------------"

for output_file in "$OUTPUT_DIR"/*_testR.json; do
    if [ ! -f "$output_file" ]; then
        continue
    fi

    model_name=$(basename "$output_file" | sed 's/_testR.json//')

    # Extract metrics using Python
    python3 -c "
import json
import sys
with open('$output_file') as f:
    data = json.load(f)

full_auc = data['full_forward']['auc']
per_layer = data['per_layer']

# Get L16 and L20 if available
l16_auc = per_layer.get('16', {}).get('auc', '-')
l20_auc = per_layer.get('20', {}).get('auc', '-')

# Find crystallization layer (first layer where AUC >= 95% of full)
crystal_layer = 'Never'
for layer in sorted(per_layer.keys(), key=int):
    if per_layer[layer]['auc'] >= 0.95 * full_auc:
        crystal_layer = f'L{layer}'
        break

l16_str = f'{l16_auc:.3f}' if isinstance(l16_auc, float) else l16_auc
l20_str = f'{l20_auc:.3f}' if isinstance(l20_auc, float) else l20_auc

print(f'$model_name | {full_auc:.3f} | {l16_str} | {l20_str} | {crystal_layer}')
"
done
