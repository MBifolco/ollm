#!/usr/bin/env python3
"""
Unified K=N Evaluation Script

Evaluates trained DDC models using run_config.json as the source of truth.
Supports multiple evaluation modes:
- basic: Accuracy and per-class metrics
- calibration: ECE, Brier score, confidence analysis
- layerwise: Crystallization analysis (at which layer does decision emerge)

Usage:
    # Basic evaluation
    python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42

    # Full evaluation with all modes
    python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --mode all

    # Layerwise crystallization analysis
    python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --mode layerwise

    # Evaluate multiple models
    python src/eval_kn.py --model_paths models/k2_love/ddc_* --mode basic
"""
from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sklearn.metrics import roc_auc_score, confusion_matrix

# Add src directory to path for kn module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kn import (
    format_input,
    load_jsonl,
    load_run_config,
    RunConfig,
    TASK_CONFIGS,
)


# =============================================================================
# Model Loading
# =============================================================================

def load_trained_model(model_path: str):
    """
    Load a trained model with its run_config.

    Uses model_path tokenizer as source of truth (it has the added tokens).
    """
    run_config = load_run_config(model_path)

    # Store model_path in config for reference
    run_config.model_path = model_path

    # Load tokenizer from model_path (source of truth - has added tokens)
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify token IDs match run_config
    for label, info in run_config.token_info.items():
        expected_id = info.token_id
        actual_ids = tokenizer.encode(info.matched_variant, add_special_tokens=False)
        if len(actual_ids) != 1 or actual_ids[0] != expected_id:
            raise ValueError(
                f"Token ID mismatch for '{label}': "
                f"expected {expected_id}, got {actual_ids}. "
                "Tokenizer may be corrupted."
            )

    # Load base model
    print(f"Loading base model: {run_config.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        run_config.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Resize embeddings to match tokenizer (for DDC/dedicated variants)
    # The saved adapter has embeddings sized to the tokenizer used during training,
    # so we need to match that size before loading the adapter
    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Load LoRA adapter (this also loads saved embedding layers if present)
    print(f"Loading LoRA adapter from {model_path}...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    return model, tokenizer, run_config


# =============================================================================
# Inference
# =============================================================================

@torch.no_grad()
def get_decision_logits(
    model, tokenizer, example: Dict, run_config: RunConfig
) -> Dict[str, float]:
    """
    Get logits for all candidate tokens at the decision position.

    Returns dict mapping label -> logit value.
    """
    # Get task config for formatting
    task_config = TASK_CONFIGS[run_config.task]

    # Format input using run_config's decision_prefix_rendered (contract enforcement)
    input_text = format_input(
        example, task_config, run_config.tokens,
        decision_prefix_rendered=run_config.decision_prefix_rendered
    )

    messages = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=model.device)

    outputs = model(input_tensor)
    logits = outputs.logits[0, -1, :]  # Last position

    # Get logits for each label's token
    label_logits = {}
    for label in run_config.label_order:
        token_id = run_config.token_info[label].token_id
        label_logits[label] = logits[token_id].item()

    return label_logits


def _get_lm_head_and_norm(model):
    """
    Get LM head and final norm from model, with architecture-aware fallbacks.

    Supports: Qwen, LLaMA, Mistral, GPT-2 style models.
    """
    base_model = model.get_base_model()

    # Get lm_head
    lm_head = getattr(base_model, "lm_head", None)
    if lm_head is None:
        raise ValueError(
            f"Could not find lm_head on model {type(base_model).__name__}. "
            "Layerwise analysis requires access to the language model head."
        )

    # Get final norm - try common locations
    final_norm = None
    if hasattr(base_model, "model") and hasattr(base_model.model, "norm"):
        # Qwen, LLaMA style
        final_norm = base_model.model.norm
    elif hasattr(base_model, "model") and hasattr(base_model.model, "final_layernorm"):
        # Some other architectures
        final_norm = base_model.model.final_layernorm
    elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "ln_f"):
        # GPT-2 style
        final_norm = base_model.transformer.ln_f

    if final_norm is None:
        raise ValueError(
            f"Could not find final layer norm on model {type(base_model).__name__}. "
            "Layerwise analysis requires access to the final normalization layer."
        )

    return lm_head, final_norm


@torch.no_grad()
def get_layerwise_logits(
    model, tokenizer, example: Dict, run_config: RunConfig
) -> Dict[int, Dict[str, float]]:
    """
    Get logits at each transformer layer for crystallization analysis.

    Returns dict mapping layer_idx -> {label: logit}.
    """
    task_config = TASK_CONFIGS[run_config.task]

    # Use run_config's decision_prefix_rendered (contract enforcement)
    input_text = format_input(
        example, task_config, run_config.tokens,
        decision_prefix_rendered=run_config.decision_prefix_rendered
    )

    messages = [{"role": "user", "content": input_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], device=model.device)

    # Forward pass with hidden states
    outputs = model(
        input_tensor,
        output_hidden_states=True,
        return_dict=True
    )

    hidden_states = outputs.hidden_states

    # Get LM head and final norm (architecture-aware)
    lm_head, final_norm = _get_lm_head_and_norm(model)

    n_layers = len(hidden_states) - 1  # hidden_states[0] is embeddings

    layer_logits = {}
    for layer_idx in range(n_layers + 1):
        h = hidden_states[layer_idx][0, -1, :]  # Last position
        h_normed = final_norm(h.unsqueeze(0)).squeeze(0)
        logits = lm_head(h_normed)

        label_logits = {}
        for label in run_config.label_order:
            token_id = run_config.token_info[label].token_id
            label_logits[label] = logits[token_id].item()

        layer_logits[layer_idx] = label_logits

    return layer_logits


# =============================================================================
# Metrics
# =============================================================================

def compute_accuracy(predictions: List[str], labels: List[str]) -> Dict[str, Any]:
    """Compute overall and per-class accuracy."""
    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    total = len(labels)

    # Per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class = {
        label: class_correct[label] / class_total[label] if class_total[label] > 0 else 0
        for label in class_total
    }

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class,
    }


def compute_calibration(
    probabilities: np.ndarray,  # [n_samples, n_classes]
    labels: List[str],
    label_order: List[str],
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute calibration metrics.

    Returns ECE, MCE, Brier score, and per-bin calibration data.
    """
    n_samples = len(labels)
    n_classes = len(label_order)

    # Get predicted class and confidence
    pred_indices = np.argmax(probabilities, axis=1)
    confidences = np.max(probabilities, axis=1)

    # Convert labels to indices
    label_to_idx = {l: i for i, l in enumerate(label_order)}
    true_indices = np.array([label_to_idx[l] for l in labels])

    # Correctness
    correct = (pred_indices == true_indices).astype(float)

    # ECE computation
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = correct[in_bin].mean()
            bin_error = abs(avg_accuracy - avg_confidence)

            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)

            bin_data.append({
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "count": int(in_bin.sum()),
                "avg_confidence": float(avg_confidence),
                "avg_accuracy": float(avg_accuracy),
                "error": float(bin_error),
            })

    # Brier score (multi-class)
    one_hot = np.zeros((n_samples, n_classes))
    for i, idx in enumerate(true_indices):
        one_hot[i, idx] = 1
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))

    return {
        "ece": float(ece),
        "mce": float(mce),
        "brier_score": float(brier),
        "n_bins": n_bins,
        "bin_data": bin_data,
        "mean_confidence": float(confidences.mean()),
        "mean_accuracy": float(correct.mean()),
    }


def compute_auc(
    probabilities: np.ndarray,
    labels: List[str],
    label_order: List[str]
) -> Dict[str, Any]:
    """Compute AUC metrics (binary or macro for multi-class)."""
    n_classes = len(label_order)
    label_to_idx = {l: i for i, l in enumerate(label_order)}
    true_indices = np.array([label_to_idx[l] for l in labels])

    if n_classes == 2:
        # Binary AUC: first label in label_order is treated as positive class
        positive_label = label_order[0]
        try:
            auc = roc_auc_score(true_indices == 0, probabilities[:, 0])
            return {
                "auc": float(auc),
                "auc_type": "binary",
                "positive_class": positive_label,
            }
        except ValueError:
            return {
                "auc": None,
                "auc_type": "binary",
                "positive_class": positive_label,
                "error": "single class in test set",
            }
    else:
        # Macro AUC (one-vs-rest)
        per_class_auc = {}
        for i, label in enumerate(label_order):
            binary_labels = (true_indices == i).astype(int)
            try:
                auc = roc_auc_score(binary_labels, probabilities[:, i])
                per_class_auc[label] = float(auc)
            except ValueError:
                per_class_auc[label] = None

        valid_aucs = [v for v in per_class_auc.values() if v is not None]
        macro_auc = np.mean(valid_aucs) if valid_aucs else None

        return {
            "macro_auc": float(macro_auc) if macro_auc else None,
            "per_class_auc": per_class_auc,
            "auc_type": "macro_ovr",
        }


# =============================================================================
# Evaluation Modes
# =============================================================================

def evaluate_basic(
    model, tokenizer, test_data: List[Dict], run_config: RunConfig
) -> Dict[str, Any]:
    """Basic evaluation: accuracy, AUC, confusion matrix."""
    print(f"\nRunning basic evaluation on {len(test_data)} examples...")

    all_logits = []
    labels = []
    predictions = []

    for example in tqdm(test_data, desc="Evaluating"):
        label_logits = get_decision_logits(model, tokenizer, example, run_config)
        all_logits.append(label_logits)
        labels.append(example["label"])

        # Prediction is argmax over candidate tokens
        pred_label = max(label_logits, key=label_logits.get)
        predictions.append(pred_label)

    # Convert logits to probabilities
    logit_matrix = np.array([
        [logits[label] for label in run_config.label_order]
        for logits in all_logits
    ])
    probabilities = np.exp(logit_matrix - logit_matrix.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    # Compute metrics
    accuracy_metrics = compute_accuracy(predictions, labels)
    auc_metrics = compute_auc(probabilities, labels, run_config.label_order)

    # Confusion matrix
    label_to_idx = {l: i for i, l in enumerate(run_config.label_order)}
    true_indices = [label_to_idx[l] for l in labels]
    pred_indices = [label_to_idx[p] for p in predictions]
    cm = confusion_matrix(true_indices, pred_indices)

    return {
        "mode": "basic",
        **accuracy_metrics,
        **auc_metrics,
        "confusion_matrix": cm.tolist(),
        "label_order": run_config.label_order,
    }


def evaluate_calibration(
    model, tokenizer, test_data: List[Dict], run_config: RunConfig
) -> Dict[str, Any]:
    """Calibration evaluation: ECE, Brier score, confidence analysis."""
    print(f"\nRunning calibration evaluation on {len(test_data)} examples...")

    all_logits = []
    labels = []

    for example in tqdm(test_data, desc="Evaluating"):
        label_logits = get_decision_logits(model, tokenizer, example, run_config)
        all_logits.append(label_logits)
        labels.append(example["label"])

    # Convert logits to probabilities
    logit_matrix = np.array([
        [logits[label] for label in run_config.label_order]
        for logits in all_logits
    ])
    probabilities = np.exp(logit_matrix - logit_matrix.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    calibration_metrics = compute_calibration(
        probabilities, labels, run_config.label_order
    )

    return {
        "mode": "calibration",
        **calibration_metrics,
    }


def evaluate_layerwise(
    model, tokenizer, test_data: List[Dict], run_config: RunConfig
) -> Dict[str, Any]:
    """Layerwise crystallization analysis."""
    print(f"\nRunning layerwise evaluation on {len(test_data)} examples...")

    # Collect logits per layer
    layer_logits_all = defaultdict(list)
    labels = []

    for example in tqdm(test_data, desc="Probing layers"):
        layer_logits = get_layerwise_logits(model, tokenizer, example, run_config)
        labels.append(example["label"])

        for layer_idx, label_logits in layer_logits.items():
            layer_logits_all[layer_idx].append(label_logits)

    n_layers = max(layer_logits_all.keys())
    n_classes = len(run_config.label_order)

    # Compute metrics per layer
    layer_metrics = {}
    for layer_idx in sorted(layer_logits_all.keys()):
        layer_data = layer_logits_all[layer_idx]

        # Convert to probability matrix
        logit_matrix = np.array([
            [logits[label] for label in run_config.label_order]
            for logits in layer_data
        ])
        probabilities = np.exp(logit_matrix - logit_matrix.max(axis=1, keepdims=True))
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        # Predictions and accuracy
        predictions = [run_config.label_order[np.argmax(p)] for p in probabilities]
        accuracy = sum(1 for p, l in zip(predictions, labels) if p == l) / len(labels)

        # AUC
        auc_metrics = compute_auc(probabilities, labels, run_config.label_order)

        layer_metrics[layer_idx] = {
            "accuracy": float(accuracy),
            **auc_metrics,
        }

    # Find crystallization layer (earliest layer achieving threshold)
    def find_crystallization(metrics_key: str, threshold: float) -> Optional[int]:
        for layer_idx in sorted(layer_metrics.keys()):
            value = layer_metrics[layer_idx].get(metrics_key)
            if value is not None and value >= threshold:
                return layer_idx
        return None

    # Use appropriate metric based on n_classes
    if n_classes == 2:
        auc_key = "auc"
    else:
        auc_key = "macro_auc"

    crystallization = {
        "auc_0.90": find_crystallization(auc_key, 0.90),
        "auc_0.95": find_crystallization(auc_key, 0.95),
        "auc_0.98": find_crystallization(auc_key, 0.98),
        "accuracy_0.90": find_crystallization("accuracy", 0.90),
        "accuracy_0.95": find_crystallization("accuracy", 0.95),
    }

    return {
        "mode": "layerwise",
        "n_layers": n_layers,
        "layer_metrics": layer_metrics,
        "crystallization": crystallization,
        "final_accuracy": layer_metrics[n_layers]["accuracy"],
        "final_auc": layer_metrics[n_layers].get(auc_key),
    }


# =============================================================================
# Main
# =============================================================================

def evaluate_model(
    model_path: str,
    test_data_path: Optional[str] = None,
    modes: List[str] = ["basic"],
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single model."""

    # Load model
    model, tokenizer, run_config = load_trained_model(model_path)

    # Determine test data path
    if test_data_path is None:
        test_data_path = os.path.join(run_config.data_dir, "test.jsonl")

    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    # Load test data
    test_data = load_jsonl(test_data_path)
    print(f"Loaded {len(test_data)} test examples from {test_data_path}")

    # Print model info
    print(f"\n{'='*60}")
    print(f"Model: {model_path}")
    print(f"Task: {run_config.task}")
    print(f"Variant: {run_config.variant}", end="")
    if run_config.alpha is not None:
        print(f", Alpha: {run_config.alpha}", end="")
    if run_config.vocab_mode is not None:
        print(f", Vocab mode: {run_config.vocab_mode}", end="")
    print(f"\nSeed: {run_config.seed}")
    print(f"{'='*60}")

    # Run evaluations
    results = {
        "model_path": model_path,
        "test_data_path": test_data_path,
        "n_test": len(test_data),
        "run_config": {
            "task": run_config.task,
            "variant": run_config.variant,
            "alpha": run_config.alpha,
            "vocab_mode": run_config.vocab_mode,
            "seed": run_config.seed,
            "n_classes": run_config.n_classes,
            "labels": run_config.label_order,
        },
        "timestamp": datetime.now().isoformat(),
        "evaluations": {},
    }

    if "basic" in modes or "all" in modes:
        results["evaluations"]["basic"] = evaluate_basic(
            model, tokenizer, test_data, run_config
        )

        # Print summary
        basic = results["evaluations"]["basic"]
        print(f"\n--- Basic Results ---")
        print(f"Accuracy: {basic['accuracy']:.4f} ({basic['correct']}/{basic['total']})")
        if "auc" in basic:
            print(f"AUC: {basic['auc']:.4f}")
        elif "macro_auc" in basic:
            print(f"Macro AUC: {basic['macro_auc']:.4f}")
        print(f"Per-class accuracy: {basic['per_class_accuracy']}")

    if "calibration" in modes or "all" in modes:
        results["evaluations"]["calibration"] = evaluate_calibration(
            model, tokenizer, test_data, run_config
        )

        cal = results["evaluations"]["calibration"]
        print(f"\n--- Calibration Results ---")
        print(f"ECE: {cal['ece']:.4f}")
        print(f"MCE: {cal['mce']:.4f}")
        print(f"Brier Score: {cal['brier_score']:.4f}")
        print(f"Mean Confidence: {cal['mean_confidence']:.4f}")

    if "layerwise" in modes or "all" in modes:
        results["evaluations"]["layerwise"] = evaluate_layerwise(
            model, tokenizer, test_data, run_config
        )

        lw = results["evaluations"]["layerwise"]
        print(f"\n--- Layerwise Results ---")
        print(f"Final Accuracy (layer {lw['n_layers']}): {lw['final_accuracy']:.4f}")
        print(f"Final AUC: {lw['final_auc']:.4f}" if lw['final_auc'] else "Final AUC: N/A")
        print(f"Crystallization (AUC >= 0.95): layer {lw['crystallization']['auc_0.95']}")

    # Save results
    if output_file is None:
        output_file = os.path.join(model_path, "eval_results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Unified K=N evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42

  # Full evaluation
  python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --mode all

  # Layerwise only
  python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --mode layerwise

  # Custom test data
  python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --test_data data/k2_love/test.jsonl
        """
    )

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model directory")
    parser.add_argument("--test_data", type=str, default=None,
                       help="Path to test data (default: uses data_dir from run_config)")
    parser.add_argument("--mode", type=str, nargs="+", default=["basic"],
                       choices=["basic", "calibration", "layerwise", "all"],
                       help="Evaluation modes to run")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file (default: model_path/eval_results.json)")

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        test_data_path=args.test_data,
        modes=args.mode,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
