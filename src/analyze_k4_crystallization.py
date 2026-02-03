#!/usr/bin/env python3
"""
K=4 Crystallization Analysis

Measures "representational depth" - at what layer does the decision become accessible
for the K=4 support classification task.

Uses macro-AUC (one-vs-rest averaged) for multi-class evaluation.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# Token configurations
DDC_TOKENS = {
    "E": "⟦SUPPORT_E⟧",
    "P": "⟦SUPPORT_P⟧",
    "I": "⟦SUPPORT_I⟧",
    "S": "⟦SUPPORT_S⟧",
}

BASELINE_DEDICATED_TOKENS = {
    "E": "⟦BASE_E⟧",
    "P": "⟦BASE_P⟧",
    "I": "⟦BASE_I⟧",
    "S": "⟦BASE_S⟧",
}

BASELINE_VOCAB_TOKENS = {
    "E": "E",
    "P": "P",
    "I": "I",
    "S": "S",
}

CATEGORY_ORDER = ["E", "P", "I", "S"]


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def get_tokens_for_model(model_type: str) -> dict:
    """Get the token dictionary for a model type."""
    if model_type in ["ddc_semantic", "ddc_random"]:
        return DDC_TOKENS
    elif model_type == "baseline_dedicated":
        return BASELINE_DEDICATED_TOKENS
    elif model_type == "baseline_vocab":
        return BASELINE_VOCAB_TOKENS
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def format_ddc_input(example: dict, tokens: dict) -> str:
    """Format input for DDC/dedicated models."""
    token_list = " or ".join(tokens.values())
    return f"""Classify the type of support in this scenario:

{example['scenario']}

Respond with your classification token: {token_list}"""


def format_baseline_vocab_input(example: dict) -> str:
    """Format input for baseline vocab model."""
    return f"""Classify the type of support in this scenario:

{example['scenario']}

Respond with one letter: E (Emotional), P (Practical), I (Ideological), or S (Structural)"""


class K4LayerwiseProber:
    """Probes K=4 decision accessibility at each transformer layer."""

    def __init__(
        self,
        model_path: str,
        model_type: str,  # ddc_semantic, ddc_random, baseline_dedicated, baseline_vocab
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.base_model_name = base_model_name
        self.tokens = get_tokens_for_model(model_type)

        print(f"Loading {model_type} model from {model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Resize embeddings if needed
        if model_type != "baseline_vocab":
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Get number of layers
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.n_layers} layers")

        # Get token IDs for all categories
        self.token_ids = {}
        for cat, token in self.tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            self.token_ids[cat] = token_id
            print(f"  {cat}: '{token}' -> {token_id}")

    def get_lm_head(self):
        """Get the language model head."""
        base_model = self.model.get_base_model()
        return base_model.lm_head

    def get_final_norm(self):
        """Get the final layer norm."""
        base_model = self.model.get_base_model()
        return base_model.model.norm

    @torch.no_grad()
    def probe_example(self, example: dict) -> dict:
        """
        Probe model at each layer.
        Returns logits for each category at each layer.
        """
        if self.model_type == "baseline_vocab":
            user_content = format_baseline_vocab_input(example)
        else:
            user_content = format_ddc_input(example, self.tokens)

        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Forward pass with hidden states
        outputs = self.model(
            input_tensor,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = outputs.hidden_states
        lm_head = self.get_lm_head()
        final_norm = self.get_final_norm()

        layer_logits = {}
        decision_pos = -1  # Last position

        for layer_idx in range(self.n_layers + 1):
            h = hidden_states[layer_idx][0, decision_pos, :]
            h_normed = final_norm(h.unsqueeze(0)).squeeze(0)
            logits = lm_head(h_normed)

            # Get logits for all categories
            cat_logits = {}
            for cat in CATEGORY_ORDER:
                cat_logits[cat] = logits[self.token_ids[cat]].item()

            layer_logits[layer_idx] = cat_logits

        return layer_logits

    def cleanup(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def compute_macro_auc(logits_by_layer: dict, labels: list) -> dict:
    """
    Compute macro-AUC (one-vs-rest averaged) at each layer.

    logits_by_layer: {layer_idx: [{E: float, P: float, I: float, S: float}, ...]}
    labels: list of category labels (E, P, I, S)
    """
    metrics = {}
    n_examples = len(labels)

    for layer_idx in sorted(logits_by_layer.keys()):
        layer_data = logits_by_layer[layer_idx]

        # Compute softmax probabilities for this layer
        probs = []
        for example_logits in layer_data:
            logit_vec = [example_logits[cat] for cat in CATEGORY_ORDER]
            # Softmax
            exp_logits = np.exp(logit_vec - np.max(logit_vec))
            prob_vec = exp_logits / exp_logits.sum()
            probs.append(prob_vec)

        probs = np.array(probs)  # [n_examples, 4]

        # One-vs-rest AUC for each category
        category_aucs = []
        for cat_idx, cat in enumerate(CATEGORY_ORDER):
            # Binary labels: 1 if this category, 0 otherwise
            binary_labels = [1 if l == cat else 0 for l in labels]

            # Probabilities for this category
            cat_probs = probs[:, cat_idx]

            try:
                auc = roc_auc_score(binary_labels, cat_probs)
                category_aucs.append(auc)
            except ValueError:
                # All one class, skip
                pass

        # Macro-AUC is average of per-category AUCs
        macro_auc = np.mean(category_aucs) if category_aucs else None

        # Also compute accuracy
        preds = [CATEGORY_ORDER[np.argmax(p)] for p in probs]
        accuracy = sum(1 for p, l in zip(preds, labels) if p == l) / n_examples

        metrics[layer_idx] = {
            "macro_auc": macro_auc,
            "accuracy": accuracy,
            "per_category_auc": {cat: auc for cat, auc in zip(CATEGORY_ORDER, category_aucs)} if category_aucs else None
        }

    return metrics


def find_crystallization_layer(metrics: dict, threshold: float = 0.95) -> Optional[int]:
    """Find earliest layer achieving macro-AUC >= threshold."""
    for layer_idx in sorted(metrics.keys()):
        auc = metrics[layer_idx]["macro_auc"]
        if auc and auc >= threshold:
            return layer_idx
    return None


def run_k4_crystallization(
    model_paths: dict,  # {model_type: path}
    test_data_path: str,
    output_file: str = "k4_crystallization_results.json"
):
    """Run K=4 crystallization analysis on multiple models."""

    # Load test data
    print("Loading test data...")
    test_data = load_jsonl(test_data_path)
    labels = [ex["label"] for ex in test_data]
    print(f"Test set: {len(test_data)} examples")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    all_metrics = {}
    all_logits = {}

    for model_type, model_path in model_paths.items():
        print(f"\n{'='*60}")
        print(f"Probing {model_type}")
        print(f"{'='*60}")

        prober = K4LayerwiseProber(model_path, model_type)

        # Collect logits per layer
        logits_by_layer = defaultdict(list)

        for example in tqdm(test_data, desc=model_type):
            layer_logits = prober.probe_example(example)
            for layer_idx, cat_logits in layer_logits.items():
                logits_by_layer[layer_idx].append(cat_logits)

        prober.cleanup()

        # Compute metrics
        metrics = compute_macro_auc(logits_by_layer, labels)
        all_metrics[model_type] = metrics
        all_logits[model_type] = {str(k): v for k, v in logits_by_layer.items()}

    # Print summary table
    print("\n" + "="*80)
    print("K=4 CRYSTALLIZATION ANALYSIS (MACRO-AUC)")
    print("="*80)

    n_layers = max(int(k) for k in all_metrics[list(model_paths.keys())[0]].keys())

    # Header
    header = f"{'Layer':<6}"
    for model_type in model_paths.keys():
        header += f" {model_type:<20}"
    print(header)
    print("-"*len(header))

    # Per-layer AUC
    for layer_idx in range(n_layers + 1):
        row = f"{layer_idx:<6}"
        for model_type in model_paths.keys():
            auc = all_metrics[model_type][layer_idx]["macro_auc"]
            row += f" {auc:.4f}" if auc else " N/A"
            row += " " * (20 - 7)  # Padding
        print(row)

    # Crystallization layer summary
    print("\n" + "-"*60)
    print("Earliest layer achieving macro-AUC >= 0.95:")
    for model_type in model_paths.keys():
        crystal_layer = find_crystallization_layer(all_metrics[model_type], 0.95)
        print(f"  {model_type}: {crystal_layer if crystal_layer is not None else 'never'}")

    # Final layer accuracy
    print("\nFinal layer accuracy:")
    for model_type in model_paths.keys():
        acc = all_metrics[model_type][n_layers]["accuracy"]
        print(f"  {model_type}: {acc:.4f}")

    # Save results
    output = {
        "metadata": {
            "test_data_path": test_data_path,
            "n_examples": len(test_data),
            "n_layers": n_layers,
            "models": dict(model_paths)
        },
        "metrics": all_metrics,
        "labels": labels
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K=4 Crystallization analysis")
    parser.add_argument("--test_data", type=str, default="data/k4_support/test.jsonl")
    parser.add_argument("--output", type=str, default="k4_crystallization_results.json")
    args = parser.parse_args()

    # Default model paths
    model_paths = {
        "ddc_semantic": "models/k4/ddc_semantic_alpha065_seed42",
        "ddc_random": "models/k4/ddc_random_alpha000_seed42",
        "baseline_dedicated": "models/k4/baseline_dedicated_seed42",
        "baseline_vocab": "models/k4/baseline_vocab_seed42",
    }

    # Check which models exist
    existing_models = {}
    for model_type, path in model_paths.items():
        if Path(path).exists():
            existing_models[model_type] = path
        else:
            print(f"Warning: {model_type} not found at {path}")

    if not existing_models:
        print("No models found!")
        exit(1)

    run_k4_crystallization(
        model_paths=existing_models,
        test_data_path=args.test_data,
        output_file=args.output
    )
