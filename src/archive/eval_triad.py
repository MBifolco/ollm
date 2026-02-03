"""
Evaluate retrain triad models.

Evaluates all 18 models (Train-O/R/M × Baseline/Token × 3 seeds) on both Test-O and Test-R.
Computes AUC using forced-choice logit probing methodology from exp3a_evaluate.py.

Usage:
    python src/eval_triad.py
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from collections import defaultdict
from itertools import product

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


class SingleModelEvaluator:
    """Evaluator for a single model."""

    def __init__(
        self,
        model_path: str,
        model_type: str,  # "baseline" or "token"
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path
        self.model_type = model_type
        self.base_model_name = base_model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        if model_type == "token":
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        if model_type == "token":
            self.rom_token_id = self.tokenizer.convert_tokens_to_ids(ROM_TOKEN)
            self.nonrom_token_id = self.tokenizer.convert_tokens_to_ids(NONROM_TOKEN)

    def build_prefix(self, example: dict) -> str:
        """Build prefix for evaluation."""
        if self.model_type == "token":
            user_content = format_internal_token_input(example)
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + "DECISION: "
        else:
            user_content = format_baseline_input(example)
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + "ANSWER: "

    @torch.no_grad()
    def evaluate_token_model(self, example: dict) -> dict:
        """Evaluate token model via argmax over decision tokens."""
        prefix = self.build_prefix(example)
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        outputs = self.model(input_tensor)
        logits = outputs.logits[0, -1, :]

        p_rom = logits[self.rom_token_id].item()
        p_nonrom = logits[self.nonrom_token_id].item()

        predicted = "romantic" if p_rom > p_nonrom else "non-romantic"
        margin = p_rom - p_nonrom

        return {"predicted": predicted, "margin": margin}

    @torch.no_grad()
    def compute_sequence_logprob(self, prefix_text: str, completion: str) -> float:
        """Compute log P(completion | prefix)."""
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

        full_ids = prefix_ids + completion_ids
        input_tensor = torch.tensor([full_ids], device=self.device)

        logits = self.model(input_tensor).logits
        logprobs = torch.log_softmax(logits, dim=-1)

        start = len(prefix_ids) - 1
        end = start + len(completion_ids)

        target = torch.tensor(completion_ids, device=self.device)
        span = logprobs[0, start:end, :]
        token_logps = span.gather(1, target[:, None]).squeeze(1)

        return float(token_logps.sum().item())

    def evaluate_baseline(self, example: dict) -> dict:
        """Evaluate baseline via forced-choice sequence logprob."""
        prefix = self.build_prefix(example)

        logp_romantic = self.compute_sequence_logprob(prefix, "romantic")
        logp_nonromantic = self.compute_sequence_logprob(prefix, "non-romantic")

        predicted = "romantic" if logp_romantic > logp_nonromantic else "non-romantic"
        margin = logp_romantic - logp_nonromantic

        return {"predicted": predicted, "margin": margin}

    def evaluate(self, example: dict) -> dict:
        """Evaluate a single example."""
        if self.model_type == "token":
            return self.evaluate_token_model(example)
        else:
            return self.evaluate_baseline(example)

    def evaluate_dataset(self, data: list[dict]) -> dict:
        """Evaluate on a full dataset and return metrics."""
        y_true = []
        y_scores = []
        correct = 0

        for example in tqdm(data, desc=f"Evaluating {Path(self.model_path).name}"):
            result = self.evaluate(example)
            gt = example["label"]
            y_true.append(1 if gt == "romantic" else 0)
            y_scores.append(result["margin"])
            if result["predicted"] == gt:
                correct += 1

        accuracy = correct / len(data)
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = None

        return {"accuracy": accuracy, "auc": auc, "n": len(data)}

    def cleanup(self):
        """Release GPU memory."""
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def run_triad_evaluation(
    models_dir: str = "models/triad",
    test_o_path: str = "data/k2_love/O/val.jsonl",
    test_r_path: str = "data/k2_love/R/val.jsonl",
    output_file: str = "triad_results.json"
):
    """Run evaluation on all 18 triad models."""

    # Load test data
    print("Loading test data...")
    test_o = load_jsonl(test_o_path)
    test_r = load_jsonl(test_r_path)
    print(f"Test-O: {len(test_o)} examples, Test-R: {len(test_r)} examples")

    # Define model configurations
    train_sets = ["O", "R", "M"]
    model_types = ["baseline", "token"]
    seeds = [0, 1, 2]

    results = {}

    for train_set, model_type in product(train_sets, model_types):
        key = f"{model_type}_{train_set}"
        results[key] = {
            "test_O": {"seeds": []},
            "test_R": {"seeds": []}
        }

        for seed in seeds:
            model_name = f"{model_type}_{train_set}_seed{seed}"
            model_path = f"{models_dir}/{model_name}"

            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*60}")

            evaluator = SingleModelEvaluator(
                model_path=model_path,
                model_type=model_type
            )

            # Evaluate on Test-O
            print(f"\n  Test-O:")
            metrics_o = evaluator.evaluate_dataset(test_o)
            results[key]["test_O"]["seeds"].append({
                "seed": seed,
                "auc": metrics_o["auc"],
                "accuracy": metrics_o["accuracy"]
            })
            print(f"    AUC: {metrics_o['auc']:.4f}, Accuracy: {metrics_o['accuracy']:.2%}")

            # Evaluate on Test-R
            print(f"\n  Test-R:")
            metrics_r = evaluator.evaluate_dataset(test_r)
            results[key]["test_R"]["seeds"].append({
                "seed": seed,
                "auc": metrics_r["auc"],
                "accuracy": metrics_r["accuracy"]
            })
            print(f"    AUC: {metrics_r['auc']:.4f}, Accuracy: {metrics_r['accuracy']:.2%}")

            evaluator.cleanup()

    # Compute aggregates (mean ± std)
    print("\n" + "="*80)
    print("AGGREGATING RESULTS")
    print("="*80)

    for key in results:
        for test_set in ["test_O", "test_R"]:
            aucs = [s["auc"] for s in results[key][test_set]["seeds"] if s["auc"] is not None]
            accs = [s["accuracy"] for s in results[key][test_set]["seeds"]]

            results[key][test_set]["mean_auc"] = np.mean(aucs) if aucs else None
            results[key][test_set]["std_auc"] = np.std(aucs) if aucs else None
            results[key][test_set]["mean_accuracy"] = np.mean(accs) if accs else None
            results[key][test_set]["std_accuracy"] = np.std(accs) if accs else None

    # Print summary table
    print("\n" + "="*80)
    print("TRIAD EVALUATION RESULTS: AUC (mean ± std over 3 seeds)")
    print("="*80)

    print(f"\n{'Train Set':<12} {'Model':<12} {'Test-O AUC':<20} {'Test-R AUC':<20} {'Δ (O→R)':<12}")
    print("-"*80)

    for train_set in train_sets:
        for model_type in model_types:
            key = f"{model_type}_{train_set}"
            m_o = results[key]["test_O"]
            m_r = results[key]["test_R"]

            auc_o = f"{m_o['mean_auc']:.4f} ± {m_o['std_auc']:.4f}" if m_o['mean_auc'] else "N/A"
            auc_r = f"{m_r['mean_auc']:.4f} ± {m_r['std_auc']:.4f}" if m_r['mean_auc'] else "N/A"

            delta = m_r['mean_auc'] - m_o['mean_auc'] if m_o['mean_auc'] and m_r['mean_auc'] else None
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"

            print(f"Train-{train_set:<7} {model_type:<12} {auc_o:<20} {auc_r:<20} {delta_str:<12}")
        print()

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrain triad models")
    parser.add_argument("--models_dir", type=str, default="models/triad")
    parser.add_argument("--test_o", type=str, default="data/k2_love/O/val.jsonl")
    parser.add_argument("--test_r", type=str, default="data/k2_love/R/val.jsonl")
    parser.add_argument("--output", type=str, default="triad_results.json")
    args = parser.parse_args()

    run_triad_evaluation(
        models_dir=args.models_dir,
        test_o_path=args.test_o,
        test_r_path=args.test_r,
        output_file=args.output
    )
