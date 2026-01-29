"""
Track 1 probing evaluation for the love disambiguation experiment.

Evaluates models using forced-choice logit comparison:
- Token model: probe P(ROM_TOKEN) vs P(NONROM_TOKEN) at DECISION position
- Baseline model: probe sequence logprob for "romantic" vs "non-romantic" at ANSWER position

This provides reviewer-proof evaluation by measuring decision geometry directly,
avoiding confounds from generation/parsing.
"""
from __future__ import annotations

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Literal

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from utils import load_jsonl, format_baseline_input, format_internal_token_input


# Decision tokens
ROM_TOKEN = "⟦LOVE_ROM⟧"
NONROM_TOKEN = "⟦LOVE_NONROM⟧"


@dataclass
class ProbeResult:
    """Per-example probe result."""
    example_id: int
    ground_truth: str
    bucket: str

    # For token model: decision token probabilities
    p_rom_token: float | None = None
    p_nonrom_token: float | None = None
    token_predicted: str | None = None
    token_margin: float | None = None

    # For baseline/both: label sequence logprobs
    logp_romantic: float | None = None
    logp_nonromantic: float | None = None
    label_predicted: str | None = None
    label_margin: float | None = None

    # Debug flags
    found_decision_token: bool | None = None
    found_label_subseq: bool | None = None

    # Overall correctness
    correct: bool | None = None


class Track1Prober:
    """Prober for Track 1 models."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        model_type: Literal["baseline", "internal_token"] = "baseline",
    ):
        self.model_type = model_type
        self.model_path = model_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        print(f"Loading base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Resize embeddings if internal token model
        if model_type == "internal_token":
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        print(f"Loading adapter from: {model_path}")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()

        self.device = next(self.model.parameters()).device

        # Get token IDs for decision tokens (if applicable)
        if model_type == "internal_token":
            self.rom_token_id = self.tokenizer.convert_tokens_to_ids(ROM_TOKEN)
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(NONROM_TOKEN)

            # Verify tokens are valid
            if self.rom_token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"ROM token not found in vocabulary: {ROM_TOKEN}")
            if self.nonrom_token_id == self.tokenizer.unk_token_id:
                raise ValueError(f"NONROM token not found in vocabulary: {NONROM_TOKEN}")

            print(f"Token IDs: ROM={self.rom_token_id}, NONROM={self.nonrom_token_id}")

    def build_prefilled_prompt(self, example: dict, prefill: str) -> str:
        """Build prompt with assistant prefill for probing.

        Args:
            example: The example dict with 'scenario' key
            prefill: The text to prefill after assistant start (e.g., "DECISION: " or "ANSWER: ")

        Returns:
            Full prompt string ending with the prefill
        """
        if self.model_type == "baseline":
            user_content = format_baseline_input(example)
        else:
            user_content = format_internal_token_input(example)

        messages = [{"role": "user", "content": user_content}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Append prefill (e.g., "DECISION: " or "ANSWER: ")
        return prompt + prefill

    @torch.no_grad()
    def probe_decision_token(self, example: dict) -> dict:
        """Probe for decision token probability at DECISION: position.

        Returns dict with p_rom_token, p_nonrom_token, predicted, margin.
        """
        if self.model_type != "internal_token":
            raise ValueError("Decision token probing only for internal_token model")

        # Build prompt with "DECISION: " prefill (with space)
        prefix_text = self.build_prefilled_prompt(example, "DECISION: ")

        # Tokenize
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        if len(prefix_ids) == 0:
            raise ValueError("Prefix must be non-empty")

        input_tensor = torch.tensor([prefix_ids], device=self.device)

        # Forward pass
        outputs = self.model(input_tensor)
        logits = outputs.logits[0, -1, :]  # Last position predicts next token
        probs = torch.softmax(logits, dim=-1)

        p_rom = probs[self.rom_token_id].item()
        p_nonrom = probs[self.nonrom_token_id].item()

        predicted = "romantic" if p_rom > p_nonrom else "non-romantic"
        margin = p_rom - p_nonrom  # Signed margin

        return {
            "p_rom_token": p_rom,
            "p_nonrom_token": p_nonrom,
            "token_predicted": predicted,
            "token_margin": margin,
            "found_decision_token": True  # We successfully probed
        }

    @torch.no_grad()
    def compute_sequence_logprob(self, prefix_text: str, completion: str) -> float:
        """Compute log P(completion | prefix) by summing token-by-token logprobs."""
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        if len(prefix_ids) == 0:
            raise ValueError("Prefix must be non-empty")

        full_ids = prefix_ids + completion_ids
        input_tensor = torch.tensor([full_ids], device=self.device)

        logits = self.model(input_tensor).logits
        logprobs = torch.log_softmax(logits, dim=-1)

        # logits index that predicts completion[0] is len(prefix_ids)-1
        start = len(prefix_ids) - 1
        end = start + len(completion_ids)

        # target token ids for completion
        target = torch.tensor(completion_ids, device=self.device)

        # gather logprobs for each completion token
        span = logprobs[0, start:end, :]
        token_logps = span.gather(1, target[:, None]).squeeze(1)

        return float(token_logps.sum().item())

    def probe_label(self, example: dict) -> dict:
        """Probe for label sequence logprob at ANSWER: position.

        Works for both baseline and internal_token models.
        Returns dict with logp_romantic, logp_nonromantic, predicted, margin.
        """
        # For internal token model, we need to include DECISION line in prefill
        if self.model_type == "internal_token":
            # We need to know what decision token to include
            # For probing, we'll probe after the full DECISION line
            # This means we probe at ANSWER: position regardless of which token was emitted
            # We'll do two probes: one assuming ROM token, one assuming NONROM token
            # But for simplicity, let's just probe the label conditioned on the ground truth decision

            # Actually, for Track 1 evaluation we want to see if label matches decision
            # So let's probe the label after a fixed prefill that includes the decision
            # But we don't know the decision yet...

            # Simpler approach: probe label at ANSWER: position after DECISION: <token>\n
            # We'll use the ground truth to set the decision (for evaluating label head)
            gt = example.get("label", "")
            if gt == "romantic":
                decision_prefill = f"DECISION: {ROM_TOKEN}\nANSWER: "
            else:
                decision_prefill = f"DECISION: {NONROM_TOKEN}\nANSWER: "

            prefix_text = self.build_prefilled_prompt(example, decision_prefill)
        else:
            # Baseline: just "ANSWER: " prefill
            prefix_text = self.build_prefilled_prompt(example, "ANSWER: ")

        # Compute sequence logprobs for both labels
        logp_romantic = self.compute_sequence_logprob(prefix_text, "romantic")
        logp_nonromantic = self.compute_sequence_logprob(prefix_text, "non-romantic")

        predicted = "romantic" if logp_romantic > logp_nonromantic else "non-romantic"
        margin = logp_romantic - logp_nonromantic  # Signed margin

        return {
            "logp_romantic": logp_romantic,
            "logp_nonromantic": logp_nonromantic,
            "label_predicted": predicted,
            "label_margin": margin,
            "found_label_subseq": True
        }

    def probe_example(self, example: dict, example_id: int) -> ProbeResult:
        """Run full probing on a single example."""
        result = ProbeResult(
            example_id=example_id,
            ground_truth=example.get("label", "unknown"),
            bucket=example.get("bucket", "unknown")
        )

        # Probe decision token (internal_token model only)
        if self.model_type == "internal_token":
            try:
                token_result = self.probe_decision_token(example)
                result.p_rom_token = token_result["p_rom_token"]
                result.p_nonrom_token = token_result["p_nonrom_token"]
                result.token_predicted = token_result["token_predicted"]
                result.token_margin = token_result["token_margin"]
                result.found_decision_token = token_result["found_decision_token"]
            except Exception as e:
                print(f"Decision token probe failed for example {example_id}: {e}")
                result.found_decision_token = False

        # Probe label
        try:
            label_result = self.probe_label(example)
            result.logp_romantic = label_result["logp_romantic"]
            result.logp_nonromantic = label_result["logp_nonromantic"]
            result.label_predicted = label_result["label_predicted"]
            result.label_margin = label_result["label_margin"]
            result.found_label_subseq = label_result["found_label_subseq"]
        except Exception as e:
            print(f"Label probe failed for example {example_id}: {e}")
            result.found_label_subseq = False

        # Determine correctness based on model type
        if self.model_type == "internal_token":
            # Primary evaluation: decision token correctness
            result.correct = result.token_predicted == result.ground_truth
        else:
            # Baseline: label correctness
            result.correct = result.label_predicted == result.ground_truth

        return result


def evaluate_track1(
    model_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_type: Literal["baseline", "internal_token"] = "baseline",
    output_file: str = None
) -> tuple[list[ProbeResult], dict]:
    """Run Track 1 probe evaluation on a model."""

    prober = Track1Prober(
        model_path=model_path,
        base_model_name=base_model_name,
        model_type=model_type
    )

    test_data = load_jsonl(test_data_path)
    results = []

    print(f"Probing {len(test_data)} examples...")
    for i, example in enumerate(tqdm(test_data)):
        result = prober.probe_example(example, i)
        results.append(result)

    # Compute metrics
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / len(results) if results else 0

    metrics = {
        "accuracy": accuracy,
        "total_examples": len(results),
        "correct": correct,
    }

    # Token-specific metrics
    if model_type == "internal_token":
        token_margins = [r.token_margin for r in results if r.token_margin is not None]
        metrics["avg_token_margin"] = sum(token_margins) / len(token_margins) if token_margins else 0
        metrics["token_probe_success_rate"] = sum(1 for r in results if r.found_decision_token) / len(results)

    # Label probe metrics (both models)
    label_margins = [r.label_margin for r in results if r.label_margin is not None]
    metrics["avg_label_margin"] = sum(label_margins) / len(label_margins) if label_margins else 0

    # Per-bucket accuracy
    bucket_correct = {}
    bucket_total = {}
    for r in results:
        bucket = r.bucket
        bucket_total[bucket] = bucket_total.get(bucket, 0) + 1
        if r.correct:
            bucket_correct[bucket] = bucket_correct.get(bucket, 0) + 1

    metrics["per_bucket_accuracy"] = {
        b: bucket_correct.get(b, 0) / bucket_total[b]
        for b in bucket_total
    }

    # Save results if output file specified
    if output_file:
        output = {
            "model_path": model_path,
            "model_type": model_type,
            "metrics": metrics,
            "results": [asdict(r) for r in results]
        }
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_file}")

    return results, metrics


def compare_track1(
    baseline_path: str,
    internal_token_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_file: str = "track1_results.json"
):
    """Run Track 1 comparison between baseline and internal-token models."""

    all_results = {}

    # Evaluate baseline
    print("\n" + "="*50)
    print("EVALUATING BASELINE MODEL (Track 1)")
    print("="*50)
    baseline_results, baseline_metrics = evaluate_track1(
        model_path=baseline_path,
        test_data_path=test_data_path,
        base_model_name=base_model_name,
        model_type="baseline"
    )
    all_results["baseline"] = {
        "metrics": baseline_metrics,
        "results": [asdict(r) for r in baseline_results]
    }

    # Evaluate internal token model
    print("\n" + "="*50)
    print("EVALUATING INTERNAL TOKEN MODEL (Track 1)")
    print("="*50)
    internal_results, internal_metrics = evaluate_track1(
        model_path=internal_token_path,
        test_data_path=test_data_path,
        base_model_name=base_model_name,
        model_type="internal_token"
    )
    all_results["internal_token"] = {
        "metrics": internal_metrics,
        "results": [asdict(r) for r in internal_results]
    }

    # Print comparison
    print("\n" + "="*50)
    print("TRACK 1 COMPARISON SUMMARY")
    print("="*50)

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Internal Token':>15} {'Delta':>15}")
    print("-" * 75)

    for metric in ["accuracy", "avg_label_margin"]:
        b_val = baseline_metrics.get(metric, 0) or 0
        i_val = internal_metrics.get(metric, 0) or 0
        delta = i_val - b_val
        delta_str = f"{delta:+.4f}" if delta != 0 else "0.0000"
        print(f"{metric:<30} {b_val:>15.4f} {i_val:>15.4f} {delta_str:>15}")

    # Token-specific metrics
    print(f"\n{'Token-specific metrics':<30}")
    print(f"{'avg_token_margin':<30} {'N/A':>15} {internal_metrics.get('avg_token_margin', 0):>15.4f}")

    # Per-bucket comparison
    print(f"\n{'Per-bucket accuracy:':<30}")
    all_buckets = set(baseline_metrics.get("per_bucket_accuracy", {}).keys()) | \
                  set(internal_metrics.get("per_bucket_accuracy", {}).keys())
    for bucket in sorted(all_buckets):
        b_acc = baseline_metrics.get("per_bucket_accuracy", {}).get(bucket, 0)
        i_acc = internal_metrics.get("per_bucket_accuracy", {}).get(bucket, 0)
        delta = i_acc - b_acc
        print(f"  {bucket:<28} {b_acc:>15.4f} {i_acc:>15.4f} {delta:>+15.4f}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Track 1 probing evaluation")
    parser.add_argument("--baseline_path", type=str, default="models/baseline")
    parser.add_argument("--internal_token_path", type=str, default="models/internal_token")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="track1_results.json")
    args = parser.parse_args()

    compare_track1(
        baseline_path=args.baseline_path,
        internal_token_path=args.internal_token_path,
        test_data_path=args.test_data,
        base_model_name=args.base_model,
        output_file=args.output
    )
