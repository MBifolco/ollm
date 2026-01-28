#!/bin/bash
# Full experiment pipeline for love disambiguation MVP

set -e

cd "$(dirname "$0")/.."

# Use CPU for training (AMD ROCm kernel issues workaround)
export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false

echo "=============================================="
echo "Love Disambiguation MVP - Full Experiment"
echo "=============================================="

# Step 1: Generate dataset (if not exists)
if [ ! -f "data/train.jsonl" ]; then
    echo ""
    echo "[Step 1/4] Generating dataset..."
    python src/data_generation.py --total 1000 --output data
else
    echo ""
    echo "[Step 1/4] Dataset already exists, skipping generation"
fi

# Step 2: Train baseline model
if [ ! -d "models/baseline" ]; then
    echo ""
    echo "[Step 2/4] Training baseline model..."
    python src/train_baseline.py \
        --data_dir data \
        --output_dir models/baseline \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --max_seq_length 512 \
        --lora_r 8
else
    echo ""
    echo "[Step 2/4] Baseline model already exists, skipping training"
fi

# Step 3: Train internal token model
if [ ! -d "models/internal_token" ]; then
    echo ""
    echo "[Step 3/4] Training internal token model..."
    python src/train_internal_token.py \
        --data_dir data \
        --output_dir models/internal_token \
        --model_name Qwen/Qwen2.5-0.5B-Instruct \
        --max_seq_length 512 \
        --lora_r 8
else
    echo ""
    echo "[Step 3/4] Internal token model already exists, skipping training"
fi

# Step 4: Evaluate and compare
echo ""
echo "[Step 4/4] Running evaluation..."
python src/evaluate.py \
    --baseline_path models/baseline \
    --internal_token_path models/internal_token \
    --test_data data/test.jsonl \
    --base_model Qwen/Qwen2.5-0.5B-Instruct \
    --output evaluation_results.json

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "Results saved to evaluation_results.json"
echo "=============================================="
