"""
Experiment 3A: Evaluate models on rewritten test set.

Uses the same logit probing methodology as Track 1 (probe_track1.py) but on rewritten data.
Compares degradation between baseline and token model to determine if token learned
semantics rather than surface style.

Usage:
    python src/exp3a_evaluate.py --test_original data/test.jsonl --test_rewritten data/test_rewritten.jsonl
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


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
    """Format input for the internal-token model - Track 1 version."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_baseline_input(example: dict) -> str:
    """Format input for the baseline model - Track 1 version."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""


@dataclass
class Exp3AResult:
    """Result for a single example in Experiment 3A."""
    example_id: int
    ground_truth: str
    bucket: str
    is_rewrite: bool

    # Token model results
    token_predicted: str
    token_correct: bool
    token_margin: float  # P(ROM) - P(NONROM)

    # Baseline results
    baseline_predicted: str
    baseline_correct: bool
    baseline_margin: float  # logP(romantic) - logP(non-romantic)


class Exp3AEvaluator:
    """Evaluator for Experiment 3A."""

    def __init__(
        self,
        baseline_path: str,
        token_model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.base_model_name = base_model_name

        # Load baseline model
        print("Loading baseline model...")
        self.baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_path, trust_remote_code=True)
        if self.baseline_tokenizer.pad_token is None:
            self.baseline_tokenizer.pad_token = self.baseline_tokenizer.eos_token

        self.baseline_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.baseline_model = PeftModel.from_pretrained(self.baseline_model, baseline_path)
        self.baseline_model.eval()

        # Load token model
        print("Loading token model...")
        self.token_tokenizer = AutoTokenizer.from_pretrained(token_model_path, trust_remote_code=True)
        if self.token_tokenizer.pad_token is None:
            self.token_tokenizer.pad_token = self.token_tokenizer.eos_token

        self.token_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )
        self.token_model.resize_token_embeddings(len(self.token_tokenizer))
        self.token_model = PeftModel.from_pretrained(self.token_model, token_model_path)
        self.token_model.eval()

        self.device = next(self.token_model.parameters()).device

        # Get decision token IDs
        self.rom_token_id = self.token_tokenizer.convert_tokens_to_ids(ROM_TOKEN)
        self.nonrom_token_id = self.token_tokenizer.convert_tokens_to_ids(NONROM_TOKEN)
        print(f"Decision token IDs: ROM={self.rom_token_id}, NONROM={self.nonrom_token_id}")

    def build_token_model_prefix(self, example: dict) -> str:
        """Build prefix for token model ending with 'DECISION: '."""
        user_content = format_internal_token_input(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.token_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt + "DECISION: "

    def build_baseline_prefix(self, example: dict) -> str:
        """Build prefix for baseline ending with 'ANSWER: '."""
        user_content = format_baseline_input(example)
        messages = [{"role": "user", "content": user_content}]
        prompt = self.baseline_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt + "ANSWER: "

    @torch.no_grad()
    def evaluate_token_model(self, example: dict) -> dict:
        """Evaluate token model via argmax over decision tokens."""
        prefix = self.build_token_model_prefix(example)
        input_ids = self.token_tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        outputs = self.token_model(input_tensor)
        logits = outputs.logits[0, -1, :]

        p_rom = logits[self.rom_token_id].item()
        p_nonrom = logits[self.nonrom_token_id].item()

        predicted = "romantic" if p_rom > p_nonrom else "non-romantic"
        margin = p_rom - p_nonrom

        return {"predicted": predicted, "margin": margin}

    @torch.no_grad()
    def compute_sequence_logprob(self, model, tokenizer, prefix_text: str, completion: str) -> float:
        """Compute log P(completion | prefix)."""
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        full_ids = prefix_ids + completion_ids
        input_tensor = torch.tensor([full_ids], device=self.device)

        logits = model(input_tensor).logits
        logprobs = torch.log_softmax(logits, dim=-1)

        start = len(prefix_ids) - 1
        end = start + len(completion_ids)

        target = torch.tensor(completion_ids, device=self.device)
        span = logprobs[0, start:end, :]
        token_logps = span.gather(1, target[:, None]).squeeze(1)

        return float(token_logps.sum().item())

    def evaluate_baseline(self, example: dict) -> dict:
        """Evaluate baseline via forced-choice sequence logprob."""
        prefix = self.build_baseline_prefix(example)

        logp_romantic = self.compute_sequence_logprob(
            self.baseline_model, self.baseline_tokenizer, prefix, "romantic"
        )
        logp_nonromantic = self.compute_sequence_logprob(
            self.baseline_model, self.baseline_tokenizer, prefix, "non-romantic"
        )

        predicted = "romantic" if logp_romantic > logp_nonromantic else "non-romantic"
        margin = logp_romantic - logp_nonromantic

        return {"predicted": predicted, "margin": margin}

    def evaluate_example(self, example: dict, example_id: int) -> Exp3AResult:
        """Evaluate a single example on both models."""
        token_result = self.evaluate_token_model(example)
        baseline_result = self.evaluate_baseline(example)

        ground_truth = example.get("label", "unknown")

        return Exp3AResult(
            example_id=example_id,
            ground_truth=ground_truth,
            bucket=example.get("bucket", "unknown"),
            is_rewrite=example.get("is_rewrite", False),
            token_predicted=token_result["predicted"],
            token_correct=token_result["predicted"] == ground_truth,
            token_margin=token_result["margin"],
            baseline_predicted=baseline_result["predicted"],
            baseline_correct=baseline_result["predicted"] == ground_truth,
            baseline_margin=baseline_result["margin"]
        )


def compute_metrics(results: list[Exp3AResult]) -> dict:
    """Compute aggregate metrics from results."""
    n = len(results)
    if n == 0:
        return {}

    token_correct = sum(1 for r in results if r.token_correct)
    baseline_correct = sum(1 for r in results if r.baseline_correct)

    # AUC for token model (using margins)
    y_true = [1 if r.ground_truth == "romantic" else 0 for r in results]
    token_scores = [r.token_margin for r in results]
    baseline_scores = [r.baseline_margin for r in results]

    try:
        token_auc = roc_auc_score(y_true, token_scores)
    except ValueError:
        token_auc = None

    try:
        baseline_auc = roc_auc_score(y_true, baseline_scores)
    except ValueError:
        baseline_auc = None

    # Recall by class
    rom_results = [r for r in results if r.ground_truth == "romantic"]
    nonrom_results = [r for r in results if r.ground_truth == "non-romantic"]

    token_rom_recall = sum(1 for r in rom_results if r.token_correct) / len(rom_results) if rom_results else 0
    token_nonrom_recall = sum(1 for r in nonrom_results if r.token_correct) / len(nonrom_results) if nonrom_results else 0
    baseline_rom_recall = sum(1 for r in rom_results if r.baseline_correct) / len(rom_results) if rom_results else 0
    baseline_nonrom_recall = sum(1 for r in nonrom_results if r.baseline_correct) / len(nonrom_results) if nonrom_results else 0

    return {
        "n": n,
        "token_accuracy": token_correct / n,
        "token_auc": token_auc,
        "token_rom_recall": token_rom_recall,
        "token_nonrom_recall": token_nonrom_recall,
        "baseline_accuracy": baseline_correct / n,
        "baseline_auc": baseline_auc,
        "baseline_rom_recall": baseline_rom_recall,
        "baseline_nonrom_recall": baseline_nonrom_recall
    }


def run_exp3a_evaluation(
    baseline_path: str,
    token_model_path: str,
    test_original_path: str,
    test_rewritten_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_file: str = "exp3a_results.json"
):
    """Run full Experiment 3A evaluation."""

    evaluator = Exp3AEvaluator(
        baseline_path=baseline_path,
        token_model_path=token_model_path,
        base_model_name=base_model_name
    )

    # Load data
    original_data = load_jsonl(test_original_path)
    rewritten_data = load_jsonl(test_rewritten_path)

    print(f"\nOriginal test: {len(original_data)} examples")
    print(f"Rewritten test: {len(rewritten_data)} examples")

    # Evaluate original
    print("\n>>> Evaluating on ORIGINAL test set...")
    original_results = []
    for i, example in enumerate(tqdm(original_data)):
        result = evaluator.evaluate_example(example, i)
        original_results.append(result)

    # Evaluate rewritten
    print("\n>>> Evaluating on REWRITTEN test set...")
    rewritten_results = []
    for i, example in enumerate(tqdm(rewritten_data)):
        result = evaluator.evaluate_example(example, i)
        rewritten_results.append(result)

    # Compute metrics
    original_metrics = compute_metrics(original_results)
    rewritten_metrics = compute_metrics(rewritten_results)

    # Print results
    print("\n" + "="*80)
    print("EXPERIMENT 3A: CONTEXT GENERALIZATION RESULTS")
    print("="*80)

    print(f"\n{'Metric':<25} {'Original':>12} {'Rewritten':>12} {'Delta':>12}")
    print("-"*65)

    # Token model
    print("\nTOKEN MODEL:")
    token_acc_delta = rewritten_metrics["token_accuracy"] - original_metrics["token_accuracy"]
    print(f"{'  Accuracy':<25} {original_metrics['token_accuracy']:>12.2%} {rewritten_metrics['token_accuracy']:>12.2%} {token_acc_delta:>+12.2%}")

    if original_metrics["token_auc"] and rewritten_metrics["token_auc"]:
        token_auc_delta = rewritten_metrics["token_auc"] - original_metrics["token_auc"]
        print(f"{'  AUC':<25} {original_metrics['token_auc']:>12.4f} {rewritten_metrics['token_auc']:>12.4f} {token_auc_delta:>+12.4f}")

    token_rom_delta = rewritten_metrics["token_rom_recall"] - original_metrics["token_rom_recall"]
    token_nonrom_delta = rewritten_metrics["token_nonrom_recall"] - original_metrics["token_nonrom_recall"]
    print(f"{'  ROM Recall':<25} {original_metrics['token_rom_recall']:>12.2%} {rewritten_metrics['token_rom_recall']:>12.2%} {token_rom_delta:>+12.2%}")
    print(f"{'  NONROM Recall':<25} {original_metrics['token_nonrom_recall']:>12.2%} {rewritten_metrics['token_nonrom_recall']:>12.2%} {token_nonrom_delta:>+12.2%}")

    # Baseline
    print("\nBASELINE:")
    baseline_acc_delta = rewritten_metrics["baseline_accuracy"] - original_metrics["baseline_accuracy"]
    print(f"{'  Accuracy':<25} {original_metrics['baseline_accuracy']:>12.2%} {rewritten_metrics['baseline_accuracy']:>12.2%} {baseline_acc_delta:>+12.2%}")

    if original_metrics["baseline_auc"] and rewritten_metrics["baseline_auc"]:
        baseline_auc_delta = rewritten_metrics["baseline_auc"] - original_metrics["baseline_auc"]
        print(f"{'  AUC':<25} {original_metrics['baseline_auc']:>12.4f} {rewritten_metrics['baseline_auc']:>12.4f} {baseline_auc_delta:>+12.4f}")

    baseline_rom_delta = rewritten_metrics["baseline_rom_recall"] - original_metrics["baseline_rom_recall"]
    baseline_nonrom_delta = rewritten_metrics["baseline_nonrom_recall"] - original_metrics["baseline_nonrom_recall"]
    print(f"{'  ROM Recall':<25} {original_metrics['baseline_rom_recall']:>12.2%} {rewritten_metrics['baseline_rom_recall']:>12.2%} {baseline_rom_delta:>+12.2%}")
    print(f"{'  NONROM Recall':<25} {original_metrics['baseline_nonrom_recall']:>12.2%} {rewritten_metrics['baseline_nonrom_recall']:>12.2%} {baseline_nonrom_delta:>+12.2%}")

    # Key comparison
    print("\n" + "="*80)
    print("KEY COMPARISON: Degradation under rewrite")
    print("="*80)
    print(f"\nToken model accuracy drop:    {token_acc_delta:+.2%}")
    print(f"Baseline accuracy drop:       {baseline_acc_delta:+.2%}")
    print(f"Difference (Token - Baseline): {token_acc_delta - baseline_acc_delta:+.2%}")

    if token_acc_delta > baseline_acc_delta:
        print("\n⚠️  Token model degraded MORE than baseline on rewrites.")
        print("   This suggests the token may have learned style, not semantics.")
    elif token_acc_delta - baseline_acc_delta > -0.05:
        print("\n⚡ Token model degraded similarly to baseline.")
        print("   Results are inconclusive about style vs semantics learning.")
    else:
        print("\n✓  Token model degraded LESS than baseline on rewrites.")
        print("   This supports the hypothesis that the token learned semantics.")

    # Save results
    output = {
        "original": {
            "n_examples": len(original_data),
            "metrics": original_metrics,
            "results": [asdict(r) for r in original_results]
        },
        "rewritten": {
            "n_examples": len(rewritten_data),
            "metrics": rewritten_metrics,
            "results": [asdict(r) for r in rewritten_results]
        },
        "comparison": {
            "token_accuracy_delta": token_acc_delta,
            "baseline_accuracy_delta": baseline_acc_delta,
            "degradation_difference": token_acc_delta - baseline_acc_delta
        }
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3A: Context generalization evaluation")
    parser.add_argument("--baseline_path", type=str, default="models/baseline_track1")
    parser.add_argument("--token_model_path", type=str, default="models/internal_token_track1")
    parser.add_argument("--test_original", type=str, default="data/test.jsonl")
    parser.add_argument("--test_rewritten", type=str, default="data/test_rewritten.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="exp3a_results.json")
    args = parser.parse_args()

    run_exp3a_evaluation(
        baseline_path=args.baseline_path,
        token_model_path=args.token_model_path,
        test_original_path=args.test_original,
        test_rewritten_path=args.test_rewritten,
        base_model_name=args.base_model,
        output_file=args.output
    )
