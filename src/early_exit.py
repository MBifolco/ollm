"""
Actual Early-Exit Implementation

This implements REAL early exit - stopping computation at layer L,
not just probing hidden states after a full forward pass.

Measures:
- AUC/accuracy at each exit layer
- Real wall-clock latency
- Actual compute savings (layers skipped)
"""
from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Optional, Tuple
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


ROM_TOKEN = "⟦LOVE_ROM⟧"
NONROM_TOKEN = "⟦LOVE_NONROM⟧"


def load_jsonl(path: str) -> list[dict]:
    """Load examples from a JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def format_internal_token_input(example: dict) -> str:
    """Format input for the internal-token model."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


class EarlyExitModel:
    """
    Wrapper that enables actual early-exit inference.

    Instead of running all 24 layers, we can stop at layer L and
    apply final_norm + lm_head to get predictions.
    """

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path

        print(f"Loading model from {model_path}...")

        # Load tokenizer
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

        # Resize embeddings for token model
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Get model components for early exit
        self.n_layers = self.model.config.num_hidden_layers
        print(f"Model has {self.n_layers} layers")

        # Get decision token IDs
        self.rom_token_id = self.tokenizer.convert_tokens_to_ids(ROM_TOKEN)
        self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(NONROM_TOKEN)
        print(f"Decision token IDs: ROM={self.rom_token_id}, NONROM={self.nonrom_token_id}")

        # Cache model components
        self._cache_components()

    def _cache_components(self):
        """Cache model components for efficient early exit."""
        base = self.model.get_base_model()

        # Get the transformer layers
        self.embed_tokens = base.model.embed_tokens
        self.layers = base.model.layers
        self.final_norm = base.model.norm
        self.lm_head = base.lm_head
        self.rotary_emb = base.model.rotary_emb

    @torch.no_grad()
    def forward_until_layer(
        self,
        input_ids: torch.Tensor,
        exit_layer: int
    ) -> torch.Tensor:
        """
        Run forward pass up to (and including) exit_layer.

        Args:
            input_ids: [batch, seq_len] input token IDs
            exit_layer: Stop after this layer (0-indexed, 0 to n_layers-1)

        Returns:
            hidden_states: [batch, seq_len, hidden_dim] after exit_layer
        """
        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Get position IDs and compute rotary embeddings
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        # Create causal attention mask
        # For Qwen2, we need a 4D mask [batch, 1, seq, seq]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=self.device, dtype=torch.bool),
            diagonal=1
        )
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        # Convert to float mask (0 for attend, -inf for mask)
        attention_mask = torch.where(attention_mask, float('-inf'), 0.0)
        attention_mask = attention_mask.to(hidden_states.dtype)

        # Get rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through layers up to exit_layer
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
    def predict_at_layer(
        self,
        input_ids: torch.Tensor,
        exit_layer: int
    ) -> Tuple[int, float, float]:
        """
        Make prediction by exiting at specified layer.

        Returns:
            predicted_token_id: The predicted decision token
            margin: P(ROM) - P(NONROM) in logit space
            confidence: softmax probability of predicted token
        """
        # Forward pass up to exit_layer
        hidden_states = self.forward_until_layer(input_ids, exit_layer)

        # Get last position hidden state
        last_hidden = hidden_states[0, -1, :]  # [hidden_dim]

        # Apply final norm and lm_head
        normed = self.final_norm(last_hidden.unsqueeze(0)).squeeze(0)
        logits = self.lm_head(normed)  # [vocab_size]

        # Get decision token logits
        rom_logit = logits[self.rom_token_id].item()
        nonrom_logit = logits[self.nonrom_token_id].item()

        margin = rom_logit - nonrom_logit

        # Compute confidence (softmax over just the two decision tokens)
        decision_logits = torch.tensor([rom_logit, nonrom_logit])
        probs = torch.softmax(decision_logits, dim=0)

        if margin > 0:
            predicted = self.rom_token_id
            confidence = probs[0].item()
        else:
            predicted = self.nonrom_token_id
            confidence = probs[1].item()

        return predicted, margin, confidence

    @torch.no_grad()
    def full_forward_predict(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[int, float, float]:
        """
        Standard full forward pass prediction (for comparison).
        """
        return self.predict_at_layer(input_ids, self.n_layers - 1)

    def prepare_input(self, example: dict) -> torch.Tensor:
        """Prepare input tensor for an example."""
        user_content = format_internal_token_input(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prefix = prompt + "DECISION: "

        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        return torch.tensor([input_ids], device=self.device)


def run_early_exit_evaluation(
    model_path: str,
    test_data_path: str,
    exit_layers: list[int],
    output_file: str = "early_exit_actual.json",
    n_warmup: int = 5,
    n_timing_runs: int = 3,
    save_per_example: bool = False
):
    """
    Evaluate early-exit at various layers with real latency measurements.

    Args:
        save_per_example: If True, save per-example margins at each layer
                          for adaptive exit simulation.
    """
    print("="*60)
    print("ACTUAL EARLY-EXIT EVALUATION")
    print("="*60)

    # Load model
    model = EarlyExitModel(model_path)

    # Load test data
    print("\nLoading test data...")
    test_data = load_jsonl(test_data_path)
    print(f"Test set: {len(test_data)} examples")

    # Results storage
    results = {
        "metadata": {
            "model_path": model_path,
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
        _ = model.full_forward_predict(input_ids)

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

            # Time the prediction
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.perf_counter()

            for _ in range(n_timing_runs):
                pred_token, margin, conf = model.predict_at_layer(input_ids, exit_layer)

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = (time.perf_counter() - start_time) / n_timing_runs

            # Record results
            pred_label = "romantic" if pred_token == model.rom_token_id else "non-romantic"
            gt_label = example["label"]

            predictions.append(pred_label)
            margins.append(margin)
            confidences.append(conf)
            labels.append(1 if gt_label == "romantic" else 0)
            latencies.append(elapsed * 1000)  # Convert to ms

            # Save per-example margin for adaptive exit simulation
            if save_per_example:
                per_example_data[idx]["margins"][str(exit_layer)] = float(margin)

        # Compute metrics
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
        print(f"  P95 latency: {p95_latency:.2f} ms")
        print(f"  Mean confidence: {mean_confidence:.4f}")
        print(f"  Depth: {depth_pct:.1f}% ({exit_layer + 1}/{model.n_layers} layers)")

    # Compare with full forward pass
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
            pred_token, margin, conf = model.full_forward_predict(input_ids)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = (time.perf_counter() - start_time) / n_timing_runs

        pred_label = "romantic" if pred_token == model.rom_token_id else "non-romantic"
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

    # Summary table
    print("\n" + "="*80)
    print("EARLY-EXIT SUMMARY: Compute-Quality Tradeoff")
    print("="*80)

    print(f"\n{'Layer':<8} {'Depth %':<10} {'AUC':<10} {'Accuracy':<12} {'Latency (ms)':<15} {'Speedup':<10}")
    print("-"*70)

    full_latency = results["full_forward"]["mean_latency_ms"]

    for exit_layer in sorted(results["per_layer"].keys()):
        r = results["per_layer"][exit_layer]
        speedup = full_latency / r["mean_latency_ms"] if r["mean_latency_ms"] > 0 else 0
        auc_str = f"{r['auc']:.4f}" if r['auc'] else "N/A"
        print(f"{exit_layer:<8} {r['depth_pct']:<10.1f} {auc_str:<10} {r['accuracy']:<12.2%} {r['mean_latency_ms']:<15.2f} {speedup:<10.2f}x")

    print("-"*70)
    print(f"{'Full':<8} {'100.0':<10} {results['full_forward']['auc']:<10.4f} {results['full_forward']['accuracy']:<12.2%} {full_latency:<15.2f} {'1.00':<10}x")

    # Find best early-exit layer (highest AUC >= 95% of full)
    threshold = 0.95 * results["full_forward"]["auc"]
    best_early = None
    for exit_layer in sorted(results["per_layer"].keys()):
        if results["per_layer"][exit_layer]["auc"] and results["per_layer"][exit_layer]["auc"] >= threshold:
            best_early = exit_layer
            break

    if best_early is not None:
        r = results["per_layer"][best_early]
        speedup = full_latency / r["mean_latency_ms"]
        print(f"\n** Best early-exit: Layer {best_early} achieves {r['auc']:.4f} AUC ({r['auc']/results['full_forward']['auc']*100:.1f}% of full)")
        print(f"   Speedup: {speedup:.2f}x, Depth: {r['depth_pct']:.1f}%")

    # Add per-example data if requested
    if save_per_example:
        results["per_example"] = per_example_data
        print(f"\nPer-example margins saved for {len(per_example_data)} examples")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Actual early-exit evaluation")
    parser.add_argument("--model_path", type=str, default="models/internal_token_track1")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--output", type=str, default="early_exit_actual.json")
    parser.add_argument("--exit_layers", type=str, default="11,14,15,16,17,18,20,23",
                       help="Comma-separated list of layers to test")
    parser.add_argument("--save_per_example", action="store_true",
                       help="Save per-example margins for adaptive exit simulation")
    args = parser.parse_args()

    exit_layers = [int(x.strip()) for x in args.exit_layers.split(",")]

    run_early_exit_evaluation(
        model_path=args.model_path,
        test_data_path=args.test_data,
        exit_layers=exit_layers,
        output_file=args.output,
        save_per_example=args.save_per_example
    )


if __name__ == "__main__":
    main()
