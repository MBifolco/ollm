"""
KN Metrics - Evaluation metrics and prior probing.
"""
from __future__ import annotations

import os
from typing import Dict, List, Any

import numpy as np
import torch
from transformers import EvalPrediction

from .config import TaskConfig
from .prompt import format_input
from .io import load_jsonl


def make_compute_metrics(token_info: Dict[str, Dict], task_config: TaskConfig):
    """
    Create compute_metrics function for accuracy at the decision token position.

    Accuracy definition:
    - At the token position immediately after DECISION:
    - Restrict logits to candidate token IDs
    - Argmax among candidates
    - Compare to gold label
    """
    # Build label -> token_id mapping
    label_to_id = {label: info["token_id"] for label, info in token_info.items()}
    candidate_ids = list(label_to_id.values())
    labels = task_config.labels

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, label_ids = eval_pred

        # logits shape: [batch, seq_len, vocab_size]
        # We want the last non-padding position's logits
        # For decision-only, this is the decision token position

        correct = 0
        total = 0

        for i in range(len(logits)):
            # Find the last position (decision position)
            # In our setup, this is typically at a fixed offset from end
            seq_logits = logits[i]

            # Get logits at decision position (first non-masked position)
            # Note: label_ids has -100 for masked positions
            # We only supervise the decision token, so find the first unmasked position
            decision_pos = -1
            for pos in range(len(label_ids[i])):
                if label_ids[i][pos] != -100:
                    decision_pos = pos
                    break

            if decision_pos == -1:
                continue

            # Get logits at decision position
            pos_logits = seq_logits[decision_pos]

            # Restrict to candidate tokens
            candidate_logits = [pos_logits[tid] for tid in candidate_ids]
            pred_idx = np.argmax(candidate_logits)
            pred_label = labels[pred_idx]

            # Get gold label from the token id
            gold_token_id = label_ids[i][decision_pos]
            gold_label = None
            for label, tid in label_to_id.items():
                if tid == gold_token_id:
                    gold_label = label
                    break

            if gold_label is not None:
                total += 1
                if pred_label == gold_label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}

    return compute_metrics


def probe_decision_priors(
    model, tokenizer, token_info: Dict[str, Dict],
    task_config: TaskConfig, tokens: Dict[str, str],
    data_dir: str = None,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Probe the model's prior preferences at the DECISION: locus.

    Returns entropy and maxp statistics for the candidate token set.
    Uses task-specific sample scenarios (from val set if available).
    """
    # Get candidate token IDs
    candidate_ids = [info["token_id"] for info in token_info.values()]
    labels = list(token_info.keys())

    # Try to load scenarios from val set for task-specific probing
    sample_scenarios = []
    if data_dir:
        val_path = os.path.join(data_dir, "val.jsonl")
        if os.path.exists(val_path):
            val_data = load_jsonl(val_path)
            sample_scenarios = [ex["scenario"] for ex in val_data[:n_samples]]

    # Fallback to task-specific generic scenarios if val not available
    if not sample_scenarios:
        if task_config.name == "k2_love":
            sample_scenarios = [
                "A person expresses feelings to another.",
                "Someone shows care for their friend.",
                "Two people share a meaningful moment.",
                "A gesture of affection is made.",
                "Someone demonstrates their devotion.",
            ][:n_samples]
        elif task_config.name == "k4_support":
            sample_scenarios = [
                "Someone offers help during a difficult time.",
                "A person provides resources to another.",
                "Two people discuss their beliefs together.",
                "Someone builds something for another person.",
                "A listener offers comfort and understanding.",
            ][:n_samples]
        else:
            sample_scenarios = ["A scenario unfolds."] * n_samples

    entropies = []
    maxps = []
    argmax_labels = []

    model.eval()
    with torch.no_grad():
        for scenario in sample_scenarios:
            example = {"scenario": scenario}
            input_text = format_input(example, task_config, tokens)

            messages = [{"role": "user", "content": input_text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], device=model.device)

            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Last position

            # Restrict to candidate tokens
            candidate_logits = logits[candidate_ids]
            probs = torch.softmax(candidate_logits, dim=0).cpu().numpy()

            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

            # Compute maxp
            maxp = probs.max()
            maxps.append(maxp)

            # Track argmax
            argmax_idx = probs.argmax()
            argmax_labels.append(labels[argmax_idx])

    return {
        "worst_case_maxp": float(max(maxps)),
        "mean_maxp": float(np.mean(maxps)),
        "worst_case_entropy": float(min(entropies)),
        "mean_entropy": float(np.mean(entropies)),
        "uniform_entropy": float(np.log(len(candidate_ids))),
        "argmax_distribution": {l: argmax_labels.count(l) for l in labels},
        "n_samples": n_samples,
    }
