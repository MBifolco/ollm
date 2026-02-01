"""
Early-Exit Evaluation for Ablation Models

Evaluates random-token and single-token ablations on Test-R
to verify findings hold under rewrite shift.
"""
from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# Token definitions
RANDOM_A = "⟦RAND_A⟧"  # = romantic
RANDOM_B = "⟦RAND_B⟧"  # = non-romantic
SEMANTIC_ROM = "⟦LOVE_ROM⟧"
SEMANTIC_NONROM = "⟦LOVE_NONROM⟧"


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_random_token_input(example: dict) -> str:
    """Format input for random-token ablation."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦RAND_A⟧ or ⟦RAND_B⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_single_token_input(example: dict) -> str:
    """Format input for single-token ablation."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
If non-romantic, emit ⟦LOVE_NONROM⟧ before the label.

Output format:
DECISION: <token or empty>
ANSWER: <label>"""


def format_semantic_token_input(example: dict) -> str:
    """Format input for semantic token model."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


class EarlyExitAblationModel:
    """Early-exit wrapper for ablation models."""

    def __init__(
        self,
        model_path: str,
        model_type: str,  # "random", "single", or "semantic"
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path
        self.model_type = model_type

        print(f"Loading {model_type} model from {model_path}...")

        # Load tokenizer from saved model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Resize embeddings to match tokenizer
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        self.n_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.n_layers} layers")

        # Set up token IDs based on model type
        self._setup_tokens()
        self._cache_components()

    def _setup_tokens(self):
        """Set up decision token IDs based on model type."""
        if self.model_type == "random":
            self.rom_token_id = self.tokenizer.convert_tokens_to_ids(RANDOM_A)
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(RANDOM_B)
            self.format_fn = format_random_token_input
            print(f"Random tokens: A={self.rom_token_id}, B={self.nonrom_token_id}")
        elif self.model_type == "single":
            # For single-token, we compare NONROM token vs absence
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(SEMANTIC_NONROM)
            self.rom_token_id = None  # No romantic token
            self.format_fn = format_single_token_input
            print(f"Single token: NONROM={self.nonrom_token_id}")
        else:  # semantic
            self.rom_token_id = self.tokenizer.convert_tokens_to_ids(SEMANTIC_ROM)
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(SEMANTIC_NONROM)
            self.format_fn = format_semantic_token_input
            print(f"Semantic tokens: ROM={self.rom_token_id}, NONROM={self.nonrom_token_id}")

    def _cache_components(self):
        """Cache model components for efficient early exit."""
        base = self.model.get_base_model()
        self.embed_tokens = base.model.embed_tokens
        self.layers = base.model.layers
        self.final_norm = base.model.norm
        self.lm_head = base.lm_head
        self.rotary_emb = base.model.rotary_emb

    @torch.no_grad()
    def forward_until_layer(self, input_ids: torch.Tensor, exit_layer: int) -> torch.Tensor:
        """Run forward pass up to exit_layer."""
        hidden_states = self.embed_tokens(input_ids)

        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=1
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        attention_mask = torch.where(attention_mask, float('-inf'), 0.0)
        attention_mask = attention_mask.to(hidden_states.dtype)

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx in range(exit_layer + 1):
            layer = self.layers[layer_idx]
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        return hidden_states

    @torch.no_grad()
    def predict_at_layer(self, input_ids: torch.Tensor, exit_layer: int) -> Tuple[int, float, float]:
        """Make prediction by exiting at specified layer."""
        hidden_states = self.forward_until_layer(input_ids, exit_layer)
        last_hidden = hidden_states[0, -1, :]
        normed = self.final_norm(last_hidden.unsqueeze(0)).squeeze(0)
        logits = self.lm_head(normed)

        if self.model_type == "single":
            # For single-token: compare nonrom token logit vs next most likely
            # Use logit of NONROM token as the "non-romantic score"
            nonrom_logit = logits[self.nonrom_token_id].item()
            # Margin: negative = more likely non-romantic
            # We'll use -nonrom_logit so positive = romantic
            margin = -nonrom_logit

            # Get top token to determine prediction
            top_token = logits.argmax().item()
            if top_token == self.nonrom_token_id:
                predicted = self.nonrom_token_id
                confidence = torch.softmax(logits, dim=0)[self.nonrom_token_id].item()
            else:
                predicted = top_token  # Anything else = romantic
                confidence = 1.0 - torch.softmax(logits, dim=0)[self.nonrom_token_id].item()
        else:
            # Standard two-token comparison
            rom_logit = logits[self.rom_token_id].item()
            nonrom_logit = logits[self.nonrom_token_id].item()
            margin = rom_logit - nonrom_logit

            decision_logits = torch.tensor([rom_logit, nonrom_logit])
            probs = torch.softmax(decision_logits, dim=0)

            if margin > 0:
                predicted = self.rom_token_id
                confidence = probs[0].item()
            else:
                predicted = self.nonrom_token_id
                confidence = probs[1].item()

        return predicted, margin, confidence

    def prepare_input(self, example: dict) -> torch.Tensor:
        """Prepare input tensor for an example."""
        user_content = self.format_fn(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix = prompt + "DECISION:"

        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        return torch.tensor([input_ids], device=self.device)

    def is_romantic_prediction(self, pred_token: int) -> bool:
        """Check if prediction is romantic."""
        if self.model_type == "single":
            return pred_token != self.nonrom_token_id
        else:
            return pred_token == self.rom_token_id


def run_evaluation(
    model_path: str,
    model_type: str,
    test_data_path: str,
    exit_layers: list[int],
    output_file: str,
    n_warmup: int = 5,
    n_timing_runs: int = 3,
    save_per_example: bool = False
):
    """Evaluate early-exit at various layers.

    Args:
        save_per_example: If True, save per-example margins for adaptive exit simulation.
    """
    print("="*60)
    print(f"EARLY-EXIT EVALUATION: {model_type.upper()}")
    print("="*60)

    model = EarlyExitAblationModel(model_path, model_type)

    print("\nLoading test data...")
    test_data = load_jsonl(test_data_path)
    print(f"Test set: {len(test_data)} examples")

    results = {
        "metadata": {
            "model_path": model_path,
            "model_type": model_type,
            "test_data_path": test_data_path,
            "n_examples": len(test_data),
            "n_layers": model.n_layers,
            "exit_layers": exit_layers
        },
        "per_layer": {}
    }

    # Per-example storage for adaptive exit simulation
    per_example_data = None
    if save_per_example:
        per_example_data = [
            {
                "id": i,
                "label": 1 if ex["label"] == "romantic" else 0,
                "margins": {}
            }
            for i, ex in enumerate(test_data)
        ]

    # Warmup
    print(f"\nWarming up with {n_warmup} examples...")
    for i in range(min(n_warmup, len(test_data))):
        input_ids = model.prepare_input(test_data[i])
        _ = model.predict_at_layer(input_ids, model.n_layers - 1)

    # Evaluate at each exit layer
    for exit_layer in exit_layers:
        print(f"\n{'='*40}")
        print(f"Evaluating exit at layer {exit_layer}/{model.n_layers}")
        print(f"{'='*40}")

        predictions = []
        margins = []
        confidences = []
        labels = []
        latencies = []

        for idx, example in enumerate(tqdm(test_data, desc=f"Layer {exit_layer}")):
            input_ids = model.prepare_input(example)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            for _ in range(n_timing_runs):
                pred_token, margin, conf = model.predict_at_layer(input_ids, exit_layer)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.perf_counter() - start_time) / n_timing_runs

            pred_label = "romantic" if model.is_romantic_prediction(pred_token) else "non-romantic"
            gt_label = example["label"]

            predictions.append(pred_label)
            margins.append(margin)
            confidences.append(conf)
            labels.append(1 if gt_label == "romantic" else 0)
            latencies.append(elapsed * 1000)

            # Save per-example margin for adaptive exit simulation
            if save_per_example:
                per_example_data[idx]["margins"][str(exit_layer)] = float(margin)

        correct = sum(1 for p, e in zip(predictions, [d["label"] for d in test_data]) if p == e)
        accuracy = correct / len(test_data)

        try:
            auc = roc_auc_score(labels, margins)
        except ValueError:
            auc = None

        mean_latency = np.mean(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        mean_confidence = np.mean(confidences)
        depth_pct = (exit_layer + 1) / model.n_layers * 100

        results["per_layer"][exit_layer] = {
            "accuracy": accuracy,
            "auc": auc,
            "mean_latency_ms": mean_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "mean_confidence": mean_confidence,
            "depth_pct": depth_pct,
            "layers_used": exit_layer + 1,
            "layers_skipped": model.n_layers - exit_layer - 1
        }

        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  AUC: {auc:.4f}" if auc else "  AUC: N/A")
        print(f"  Mean latency: {mean_latency:.2f} ms")
        print(f"  Depth: {depth_pct:.1f}%")

    # Full forward pass
    print(f"\n{'='*40}")
    print(f"Full forward pass (layer {model.n_layers - 1})")
    print(f"{'='*40}")

    full_predictions = []
    full_margins = []
    full_latencies = []

    for example in tqdm(test_data, desc="Full forward"):
        input_ids = model.prepare_input(example)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()

        for _ in range(n_timing_runs):
            pred_token, margin, conf = model.predict_at_layer(input_ids, model.n_layers - 1)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start_time) / n_timing_runs

        pred_label = "romantic" if model.is_romantic_prediction(pred_token) else "non-romantic"
        full_predictions.append(pred_label)
        full_margins.append(margin)
        full_latencies.append(elapsed * 1000)

    full_correct = sum(1 for p, e in zip(full_predictions, [d["label"] for d in test_data]) if p == e)
    full_accuracy = full_correct / len(test_data)
    full_labels = [1 if d["label"] == "romantic" else 0 for d in test_data]
    full_auc = roc_auc_score(full_labels, full_margins)

    results["full_forward"] = {
        "accuracy": full_accuracy,
        "auc": full_auc,
        "mean_latency_ms": np.mean(full_latencies),
        "p50_latency_ms": np.percentile(full_latencies, 50),
        "p95_latency_ms": np.percentile(full_latencies, 95)
    }

    print(f"  Accuracy: {full_accuracy:.2%}")
    print(f"  AUC: {full_auc:.4f}")
    print(f"  Mean latency: {np.mean(full_latencies):.2f} ms")

    # Summary
    print("\n" + "="*80)
    print(f"{model_type.upper()} EARLY-EXIT SUMMARY")
    print("="*80)

    print(f"\n{'Layer':<8} {'Depth %':<10} {'AUC':<10} {'Accuracy':<12} {'Latency (ms)':<15}")
    print("-"*55)

    for exit_layer in sorted(results["per_layer"].keys()):
        r = results["per_layer"][exit_layer]
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{exit_layer:<8} {r['depth_pct']:<10.1f} {auc_str:<10} {r['accuracy']:<12.2%} {r['mean_latency_ms']:<15.2f}")

    print("-"*55)
    print(f"{'Full':<8} {'100.0':<10} {results['full_forward']['auc']:<10.4f} {results['full_forward']['accuracy']:<12.2%} {results['full_forward']['mean_latency_ms']:<15.2f}")

    # Add per-example data if requested
    if save_per_example:
        results["per_example"] = per_example_data
        print(f"\nPer-example margins saved for {len(per_example_data)} examples")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Ablation early-exit evaluation")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True,
                       choices=["random", "single", "semantic"])
    parser.add_argument("--test_data", type=str, default="data/test_rewritten.jsonl")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--exit_layers", type=str, default="14,15,16,17,18,20,23")
    parser.add_argument("--save_per_example", action="store_true",
                       help="Save per-example margins for adaptive exit simulation")
    args = parser.parse_args()

    exit_layers = [int(x.strip()) for x in args.exit_layers.split(",")]

    run_evaluation(
        model_path=args.model_path,
        model_type=args.model_type,
        test_data_path=args.test_data,
        exit_layers=exit_layers,
        output_file=args.output,
        save_per_example=args.save_per_example
    )


if __name__ == "__main__":
    main()
