"""
Evaluation harness for comparing baseline vs internal-token models.
Measures: accuracy, token efficiency, faithfulness, paraphrase robustness.
"""
from __future__ import annotations

import os
import json
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Literal
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from dotenv import load_dotenv
import anthropic

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

from utils import (
    load_jsonl,
    format_baseline_input,
    format_internal_token_input,
    parse_baseline_output,
    parse_internal_token_output,
    strip_think_section,
    count_tokens,
)


@dataclass
class EvalResult:
    """Results for a single example."""
    example_id: int
    ground_truth: str
    predicted_label: str
    explanation: str
    raw_output: str
    total_tokens: int
    visible_tokens: int
    correct: bool
    # For internal token model
    think_content: str | None = None
    has_nonrom_token: bool | None = None


@dataclass
class EvalMetrics:
    """Aggregated metrics."""
    accuracy: float
    total_examples: int
    correct: int
    avg_total_tokens: float
    avg_visible_tokens: float
    # Faithfulness
    faithfulness_score: float | None = None
    explanation_label_mismatches: int | None = None
    # Paraphrase
    paraphrase_consistency: float | None = None
    paraphrase_flips: int | None = None
    # Internal token specific
    token_usage_rate: float | None = None  # How often model emits the token
    token_accuracy: float | None = None    # Does token emission match label?


class ModelEvaluator:
    def __init__(
        self,
        model_path: str,
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        model_type: Literal["baseline", "internal_token"] = "baseline",
        device: str = "auto"
    ):
        self.model_type = model_type
        self.model_path = model_path

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model on CPU (AMD ROCm compatibility workaround)
        print(f"Loading base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )

        # Resize embeddings if internal token model
        if model_type == "internal_token":
            self.model.resize_token_embeddings(len(self.tokenizer))

        # Load LoRA adapter
        print(f"Loading adapter from: {model_path}")
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()

        self.device = next(self.model.parameters()).device

    def generate(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate a response from the model."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic for evaluation
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=False)

    def evaluate_example(self, example: dict, example_id: int) -> EvalResult:
        """Evaluate a single example."""
        # Format input based on model type
        if self.model_type == "baseline":
            prompt = format_baseline_input(example)
        else:
            prompt = format_internal_token_input(example)

        # Generate
        raw_output = self.generate(prompt)

        # Parse output
        if self.model_type == "baseline":
            parsed = parse_baseline_output(raw_output)
            think_content = None
            has_nonrom_token = None
        else:
            parsed = parse_internal_token_output(raw_output)
            think_content = parsed.get("think_content")
            has_nonrom_token = parsed.get("has_nonrom_token")

        # Count tokens
        total_tokens = count_tokens(raw_output, self.tokenizer)
        visible_output = strip_think_section(raw_output) if self.model_type == "internal_token" else raw_output
        visible_tokens = count_tokens(visible_output, self.tokenizer)

        return EvalResult(
            example_id=example_id,
            ground_truth=example["label"],
            predicted_label=parsed.get("label"),
            explanation=parsed.get("explanation", ""),
            raw_output=raw_output,
            total_tokens=total_tokens,
            visible_tokens=visible_tokens,
            correct=parsed.get("label") == example["label"],
            think_content=think_content,
            has_nonrom_token=has_nonrom_token,
        )


class FaithfulnessJudge:
    """Uses Anthropic to judge if explanation matches label."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY_KEY"))

    def judge(self, explanation: str, stated_label: str) -> dict:
        """Judge if the explanation implies the stated label."""
        prompt = f"""Given this explanation about the type of love in a scenario:

Explanation: "{explanation}"

The model labeled this as: {stated_label}

Based ONLY on the explanation text (not the label), what type of love does the explanation describe?
Does the explanation actually support the stated label?

Respond with JSON only, no markdown:
{{"implied_label": "romantic" or "non-romantic", "matches_stated": true/false, "reasoning": "brief explanation"}}"""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
                system="You are a precise judge. Respond only with valid JSON, no markdown code blocks."
            )

            content = response.content[0].text.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            return json.loads(content)
        except Exception as e:
            print(f"Faithfulness judge error: {e}")
            return {"implied_label": None, "matches_stated": None, "reasoning": str(e)}


class ParaphraseGenerator:
    """Generates paraphrases of scenarios for robustness testing."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY_KEY"))

    def generate_paraphrase(self, scenario: str) -> str | None:
        """Generate a paraphrase that preserves meaning."""
        prompt = f"""Paraphrase this scenario while preserving its exact meaning and all implications about relationships:

Original: {scenario}

Requirements:
- Keep the same characters and situation
- Preserve any contextual clues about the type of love/relationship
- Use different wording but same meaning
- Keep similar length

Return ONLY the paraphrased scenario, nothing else."""

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Paraphrase generation error: {e}")
            return None


def evaluate_model(
    model_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_type: Literal["baseline", "internal_token"] = "baseline",
    evaluate_faithfulness: bool = True,
    faithfulness_sample_size: int = 100,
) -> tuple[list[EvalResult], EvalMetrics]:
    """Run full evaluation on a model."""

    evaluator = ModelEvaluator(
        model_path=model_path,
        base_model_name=base_model_name,
        model_type=model_type
    )

    test_data = load_jsonl(test_data_path)
    results = []

    print(f"Evaluating {len(test_data)} examples...")
    for i, example in enumerate(tqdm(test_data)):
        result = evaluator.evaluate_example(example, i)
        results.append(result)

    # Compute basic metrics
    correct = sum(1 for r in results if r.correct)
    accuracy = correct / len(results) if results else 0

    total_tokens = [r.total_tokens for r in results]
    visible_tokens = [r.visible_tokens for r in results]

    metrics = EvalMetrics(
        accuracy=accuracy,
        total_examples=len(results),
        correct=correct,
        avg_total_tokens=sum(total_tokens) / len(total_tokens) if total_tokens else 0,
        avg_visible_tokens=sum(visible_tokens) / len(visible_tokens) if visible_tokens else 0,
    )

    # Internal token specific metrics
    if model_type == "internal_token":
        with_token = [r for r in results if r.has_nonrom_token]
        metrics.token_usage_rate = len(with_token) / len(results) if results else 0

        # Check if token emission matches ground truth
        token_correct = sum(
            1 for r in results
            if (r.has_nonrom_token and r.ground_truth == "non-romantic") or
               (not r.has_nonrom_token and r.ground_truth == "romantic")
        )
        metrics.token_accuracy = token_correct / len(results) if results else 0

    # Faithfulness evaluation (sample to save API costs)
    if evaluate_faithfulness and results:
        judge = FaithfulnessJudge()
        sample = results[:faithfulness_sample_size]
        mismatches = 0

        print(f"Evaluating faithfulness on {len(sample)} examples...")
        for r in tqdm(sample):
            if r.explanation and r.predicted_label:
                judgment = judge.judge(r.explanation, r.predicted_label)
                if judgment.get("matches_stated") is False:
                    mismatches += 1

        metrics.explanation_label_mismatches = mismatches
        metrics.faithfulness_score = 1 - (mismatches / len(sample)) if sample else None

    return results, metrics


def evaluate_paraphrase_robustness(
    model_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    model_type: Literal["baseline", "internal_token"] = "baseline",
    num_examples: int = 100,
) -> tuple[int, int, float]:
    """Evaluate how consistent the model is under paraphrasing."""

    evaluator = ModelEvaluator(
        model_path=model_path,
        base_model_name=base_model_name,
        model_type=model_type
    )
    paraphraser = ParaphraseGenerator()

    test_data = load_jsonl(test_data_path)[:num_examples]
    flips = 0
    evaluated = 0

    print(f"Evaluating paraphrase robustness on {len(test_data)} examples...")
    for i, example in enumerate(tqdm(test_data)):
        # Get original prediction
        original_result = evaluator.evaluate_example(example, i)

        # Generate paraphrase
        paraphrased = paraphraser.generate_paraphrase(example["scenario"])
        if not paraphrased:
            continue

        # Create paraphrased example
        para_example = example.copy()
        para_example["scenario"] = paraphrased

        # Get paraphrased prediction
        para_result = evaluator.evaluate_example(para_example, i)

        evaluated += 1
        if original_result.predicted_label != para_result.predicted_label:
            flips += 1

    consistency = 1 - (flips / evaluated) if evaluated else 0
    return flips, evaluated, consistency


def compare_models(
    baseline_path: str,
    internal_token_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_file: str = "evaluation_results.json"
):
    """Run full comparison between baseline and internal-token models."""

    results = {}

    # Evaluate baseline
    print("\n" + "="*50)
    print("EVALUATING BASELINE MODEL")
    print("="*50)
    baseline_results, baseline_metrics = evaluate_model(
        model_path=baseline_path,
        test_data_path=test_data_path,
        base_model_name=base_model_name,
        model_type="baseline"
    )
    results["baseline"] = {
        "accuracy": baseline_metrics.accuracy,
        "avg_total_tokens": baseline_metrics.avg_total_tokens,
        "avg_visible_tokens": baseline_metrics.avg_visible_tokens,
        "faithfulness_score": baseline_metrics.faithfulness_score,
        "explanation_label_mismatches": baseline_metrics.explanation_label_mismatches,
    }

    # Evaluate internal token model
    print("\n" + "="*50)
    print("EVALUATING INTERNAL TOKEN MODEL")
    print("="*50)
    internal_results, internal_metrics = evaluate_model(
        model_path=internal_token_path,
        test_data_path=test_data_path,
        base_model_name=base_model_name,
        model_type="internal_token"
    )
    results["internal_token"] = {
        "accuracy": internal_metrics.accuracy,
        "avg_total_tokens": internal_metrics.avg_total_tokens,
        "avg_visible_tokens": internal_metrics.avg_visible_tokens,
        "faithfulness_score": internal_metrics.faithfulness_score,
        "explanation_label_mismatches": internal_metrics.explanation_label_mismatches,
        "token_usage_rate": internal_metrics.token_usage_rate,
        "token_accuracy": internal_metrics.token_accuracy,
    }

    # Paraphrase robustness
    print("\n" + "="*50)
    print("EVALUATING PARAPHRASE ROBUSTNESS")
    print("="*50)

    print("Baseline:")
    b_flips, b_eval, b_cons = evaluate_paraphrase_robustness(
        baseline_path, test_data_path, base_model_name, "baseline", num_examples=100
    )
    results["baseline"]["paraphrase_flips"] = b_flips
    results["baseline"]["paraphrase_consistency"] = b_cons

    print("Internal token:")
    i_flips, i_eval, i_cons = evaluate_paraphrase_robustness(
        internal_token_path, test_data_path, base_model_name, "internal_token", num_examples=100
    )
    results["internal_token"]["paraphrase_flips"] = i_flips
    results["internal_token"]["paraphrase_consistency"] = i_cons

    # Print comparison
    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)

    print(f"\n{'Metric':<35} {'Baseline':>15} {'Internal Token':>15} {'Delta':>15}")
    print("-" * 80)

    for metric in ["accuracy", "avg_visible_tokens", "faithfulness_score", "paraphrase_consistency"]:
        b_val = results["baseline"].get(metric, 0) or 0
        i_val = results["internal_token"].get(metric, 0) or 0
        delta = i_val - b_val
        delta_str = f"{delta:+.3f}" if delta != 0 else "0.000"
        print(f"{metric:<35} {b_val:>15.3f} {i_val:>15.3f} {delta_str:>15}")

    print(f"\n{'token_usage_rate':<35} {'N/A':>15} {results['internal_token'].get('token_usage_rate', 0):>15.3f}")
    print(f"{'token_accuracy':<35} {'N/A':>15} {results['internal_token'].get('token_accuracy', 0):>15.3f}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_path", type=str, default="models/baseline")
    parser.add_argument("--internal_token_path", type=str, default="models/internal_token")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    compare_models(
        baseline_path=args.baseline_path,
        internal_token_path=args.internal_token_path,
        test_data_path=args.test_data,
        base_model_name=args.base_model,
        output_file=args.output
    )
