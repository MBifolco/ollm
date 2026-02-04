#!/bin/bash
# =============================================================================
# run_kn.sh - Complete KN Experiment Runner
# =============================================================================
# Trains all model variants and runs all evaluations for the DDC research paper.
#
# Variants:
#   - ddc (alpha=0.65): DDC with semantic-leaning initialization
#   - ddc (alpha=0.0): DDC with random initialization
#   - vocab_flat: Vocab baseline with minimal prior bias tokens
#   - vocab_peaky: Vocab baseline with high prior bias tokens (stress test)
#   - dedicated_baseline: New tokens with random initialization
#
# Tasks:
#   - k2_love: Binary love disambiguation (romantic vs non-romantic)
#   - k4_support: 4-way support classification
#
# Seeds: 42, 123, 456 for variance estimates
#
# Usage:
#   ./run_kn.sh           # Run everything
#   ./run_kn.sh train     # Only train models
#   ./run_kn.sh eval      # Only evaluate (assumes models exist)
#   ./run_kn.sh summary   # Only generate summary
# =============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
MODELS_DIR="${SCRIPT_DIR}/models"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOG_DIR="${SCRIPT_DIR}/logs"

SEEDS=(42 123 456)
TASKS=("k2_love" "k4_support")

# Create directories
mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="${LOG_DIR}/run_kn_${TIMESTAMP}.log"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "$1" | tee -a "${MASTER_LOG}"
}

log_header() {
    log "\n${BLUE}============================================================${NC}"
    log "${BLUE}$1${NC}"
    log "${BLUE}============================================================${NC}"
}

log_success() {
    log "${GREEN}✓ $1${NC}"
}

log_error() {
    log "${RED}✗ $1${NC}"
}

log_warning() {
    log "${YELLOW}⚠ $1${NC}"
}

# =============================================================================
# Training Functions
# =============================================================================

train_model() {
    local task=$1
    local variant=$2
    local seed=$3
    local extra_args=$4

    local model_name="${task}_${variant}_seed${seed}"
    local output_dir="${MODELS_DIR}/${model_name}"
    local log_file="${LOG_DIR}/train_${model_name}_${TIMESTAMP}.log"

    # Check if model already exists
    if [ -f "${output_dir}/run_config.json" ]; then
        log_warning "Model ${model_name} already exists, skipping..."
        return 0
    fi

    log "\n${YELLOW}Training: ${model_name}${NC}"
    log "  Output: ${output_dir}"
    log "  Log: ${log_file}"

    # Determine data variant for K2
    local data_variant="default"
    if [ "$task" == "k2_love" ]; then
        data_variant="M"  # Mixed data for training
    fi

    # Build command
    local cmd="python ${SRC_DIR}/train_kn.py \
        --task ${task} \
        --variant ${variant} \
        --data_variant ${data_variant} \
        --output_dir ${output_dir} \
        --seed ${seed} \
        ${extra_args}"

    # Run training
    if eval "${cmd}" > "${log_file}" 2>&1; then
        log_success "Completed: ${model_name}"
        return 0
    else
        log_error "Failed: ${model_name}"
        log "  See log: ${log_file}"
        return 1
    fi
}

train_all_models() {
    log_header "TRAINING PHASE"

    local total=0
    local success=0
    local failed=0

    for task in "${TASKS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            # DDC with alpha=0.65 (semantic leaning)
            total=$((total + 1))
            if train_model "${task}" "ddc" "${seed}" "--alpha 0.65"; then
                # Rename to indicate alpha
                local src="${MODELS_DIR}/${task}_ddc_seed${seed}"
                local dst="${MODELS_DIR}/${task}_ddc_a065_seed${seed}"
                if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
                    mv "${src}" "${dst}"
                fi
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi

            # DDC with alpha=0.0 (random init)
            total=$((total + 1))
            if train_model "${task}" "ddc" "${seed}" "--alpha 0.0"; then
                local src="${MODELS_DIR}/${task}_ddc_seed${seed}"
                local dst="${MODELS_DIR}/${task}_ddc_a000_seed${seed}"
                if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
                    mv "${src}" "${dst}"
                fi
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi

            # Vocab baseline - flat
            total=$((total + 1))
            if train_model "${task}" "vocab_baseline" "${seed}" "--vocab_mode flat"; then
                local src="${MODELS_DIR}/${task}_vocab_baseline_seed${seed}"
                local dst="${MODELS_DIR}/${task}_vocab_flat_seed${seed}"
                if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
                    mv "${src}" "${dst}"
                fi
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi

            # Vocab baseline - peaky
            total=$((total + 1))
            if train_model "${task}" "vocab_baseline" "${seed}" "--vocab_mode peaky"; then
                local src="${MODELS_DIR}/${task}_vocab_baseline_seed${seed}"
                local dst="${MODELS_DIR}/${task}_vocab_peaky_seed${seed}"
                if [ -d "${src}" ] && [ ! -d "${dst}" ]; then
                    mv "${src}" "${dst}"
                fi
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi

            # Dedicated baseline
            total=$((total + 1))
            if train_model "${task}" "dedicated_baseline" "${seed}" ""; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done
    done

    log_header "TRAINING SUMMARY"
    log "Total: ${total}, Success: ${success}, Failed: ${failed}"

    return $failed
}

# =============================================================================
# Evaluation Functions
# =============================================================================

eval_model() {
    local model_path=$1
    local test_data=$2
    local test_name=$3
    local output_file=$4

    local model_name=$(basename "${model_path}")
    local log_file="${LOG_DIR}/eval_${model_name}_${test_name}_${TIMESTAMP}.log"

    # Check if model exists
    if [ ! -f "${model_path}/run_config.json" ]; then
        log_warning "Model not found: ${model_path}, skipping..."
        return 1
    fi

    # Check if output already exists
    if [ -f "${output_file}" ]; then
        log_warning "Results exist: ${output_file}, skipping..."
        return 0
    fi

    log "\n${YELLOW}Evaluating: ${model_name} on ${test_name}${NC}"
    log "  Test data: ${test_data}"
    log "  Output: ${output_file}"

    local cmd="python ${SRC_DIR}/eval_kn.py \
        --model_path ${model_path} \
        --test_data ${test_data} \
        --mode all \
        --output ${output_file}"

    if eval "${cmd}" > "${log_file}" 2>&1; then
        log_success "Completed: ${model_name} on ${test_name}"
        return 0
    else
        log_error "Failed: ${model_name} on ${test_name}"
        log "  See log: ${log_file}"
        return 1
    fi
}

eval_all_models() {
    log_header "EVALUATION PHASE"

    local total=0
    local success=0
    local failed=0

    # Define all variants (must match training output names)
    local variants=("ddc_a065" "ddc_a000" "vocab_flat" "vocab_peaky" "dedicated_baseline")

    for task in "${TASKS[@]}"; do
        # Determine test data paths
        local test_r=""
        local test_o=""

        if [ "$task" == "k2_love" ]; then
            # K2 test files in O/R subdirectories (230 examples each)
            test_r="data/k2_love/R/test.jsonl"
            test_o="data/k2_love/O/test.jsonl"
        else
            # K4 support - use test.jsonl for both (only one test set)
            test_r="data/k4_support/test.jsonl"
            test_o="data/k4_support/val.jsonl"  # Use val as secondary
        fi

        for variant in "${variants[@]}"; do
            for seed in "${SEEDS[@]}"; do
                local model_name="${task}_${variant}_seed${seed}"
                local model_path="${MODELS_DIR}/${model_name}"

                # Eval on Test-R (primary)
                total=$((total + 1))
                local output_r="${RESULTS_DIR}/${model_name}_test_r.json"
                if eval_model "${model_path}" "${test_r}" "test_r" "${output_r}"; then
                    success=$((success + 1))
                else
                    failed=$((failed + 1))
                fi

                # Eval on Test-O (secondary)
                total=$((total + 1))
                local output_o="${RESULTS_DIR}/${model_name}_test_o.json"
                if eval_model "${model_path}" "${test_o}" "test_o" "${output_o}"; then
                    success=$((success + 1))
                else
                    failed=$((failed + 1))
                fi
            done
        done
    done

    log_header "EVALUATION SUMMARY"
    log "Total: ${total}, Success: ${success}, Failed: ${failed}"

    return $failed
}

# =============================================================================
# Summary Generation
# =============================================================================

generate_summary() {
    log_header "GENERATING SUMMARY"

    local summary_file="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

    {
        echo "============================================================"
        echo "KN EXPERIMENT RESULTS SUMMARY"
        echo "Generated: $(date)"
        echo "============================================================"
        echo ""

        for task in "${TASKS[@]}"; do
            echo ""
            echo "============================================================"
            echo "TASK: ${task}"
            echo "============================================================"

            local variants=("ddc_a065" "ddc_a000" "vocab_flat" "vocab_peaky" "dedicated_baseline")

            for variant in "${variants[@]}"; do
                echo ""
                echo "--- ${variant} ---"

                # Collect accuracies across seeds for Test-R
                local accs=""
                for seed in "${SEEDS[@]}"; do
                    local result_file="${RESULTS_DIR}/${task}_${variant}_seed${seed}_test_r.json"
                    if [ -f "${result_file}" ]; then
                        local acc=$(python3 -c "import json; d=json.load(open('${result_file}')); print(d.get('evaluations', {}).get('basic', {}).get('accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
                        accs="${accs}${seed}:${acc} "
                    fi
                done
                echo "Test-R accuracies: ${accs}"

                # Collect accuracies for Test-O
                accs=""
                for seed in "${SEEDS[@]}"; do
                    local result_file="${RESULTS_DIR}/${task}_${variant}_seed${seed}_test_o.json"
                    if [ -f "${result_file}" ]; then
                        local acc=$(python3 -c "import json; d=json.load(open('${result_file}')); print(d.get('evaluations', {}).get('basic', {}).get('accuracy', 'N/A'))" 2>/dev/null || echo "N/A")
                        accs="${accs}${seed}:${acc} "
                    fi
                done
                echo "Test-O accuracies: ${accs}"
            done
        done

        echo ""
        echo "============================================================"
        echo "RESULT FILES"
        echo "============================================================"
        ls -la "${RESULTS_DIR}"/*.json 2>/dev/null || echo "No result files found"

    } > "${summary_file}"

    log "Summary written to: ${summary_file}"
    cat "${summary_file}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log_header "KN EXPERIMENT RUNNER"
    log "Started at: $(date)"
    log "Timestamp: ${TIMESTAMP}"
    log "Master log: ${MASTER_LOG}"
    log ""
    log "Tasks: ${TASKS[*]}"
    log "Seeds: ${SEEDS[*]}"
    log "Variants: ddc_a065, ddc_a000, vocab_flat, vocab_peaky, dedicated_baseline"
    log ""
    log "Total training runs: $((${#TASKS[@]} * 5 * ${#SEEDS[@]}))"
    log "Total evaluation runs: $((${#TASKS[@]} * 5 * ${#SEEDS[@]} * 2))"

    local mode="${1:-all}"

    case "$mode" in
        train)
            train_all_models
            ;;
        eval)
            eval_all_models
            ;;
        summary)
            generate_summary
            ;;
        all)
            train_all_models
            eval_all_models
            generate_summary
            ;;
        *)
            echo "Usage: $0 [train|eval|summary|all]"
            exit 1
            ;;
    esac

    log_header "COMPLETED"
    log "Finished at: $(date)"
    log "Master log: ${MASTER_LOG}"
}

# Run main
main "$@"
