"""
Ablation Evaluation Script (E2)

Evaluates ablation models using forced-choice logit probing.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


# Token definitions
RANDOM_A = "⟦RAND_A⟧"
RANDOM_B = "⟦RAND_B⟧"
SINGLE_TOKEN = "⟦LOVE_NONROM⟧"


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


class AblationEvaluator:
    """Evaluator for ablation models."""

    def __init__(
        self,
        model_path: str,
        ablation_type: str,  # "baseline-10ep", "random-token", "single-token"
        base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    ):
        self.model_path = model_path
        self.ablation_type = ablation_type

        print(f"Loading {ablation_type} model from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # Resize if needed
        if ablation_type in ["random-token", "single-token"]:
            self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Get token IDs based on ablation type
        if ablation_type == "random-token":
            self.token_a_id = self.tokenizer.convert_tokens_to_ids(RANDOM_A)
            self.token_b_id = self.tokenizer.convert_tokens_to_ids(RANDOM_B)
            print(f"Token IDs: A={self.token_a_id}, B={self.token_b_id}")
        elif ablation_type == "single-token":
            self.token_id = self.tokenizer.convert_tokens_to_ids(SINGLE_TOKEN)
            print(f"Token ID: {self.token_id}")

    def build_prefix(self, example: dict) -> str:
        """Build evaluation prefix based on ablation type."""
        scenario = example["scenario"]

        if self.ablation_type == "baseline-10ep":
            user_content = f"""Scenario: {scenario}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + "ANSWER: "

        elif self.ablation_type == "random-token":
            user_content = f"""Scenario: {scenario}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦RAND_A⟧ or ⟦RAND_B⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + "DECISION: "

        elif self.ablation_type == "single-token":
            user_content = f"""Scenario: {scenario}

Classify whether the use of "love" is romantic or non-romantic.
If non-romantic, emit ⟦LOVE_NONROM⟧ before the label.

Output format:
DECISION: <token or empty>
ANSWER: <label>"""
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return prompt + "DECISION: "

    @torch.no_grad()
    def evaluate_baseline_10ep(self, example: dict) -> dict:
        """Evaluate baseline-10ep via sequence logprob."""
        prefix = self.build_prefix(example)
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        logprobs = {}
        for label in ["romantic", "non-romantic"]:
            completion_ids = self.tokenizer.encode(label, add_special_tokens=False)
            full_ids = prefix_ids + completion_ids
            input_tensor = torch.tensor([full_ids], device=self.device)

            outputs = self.model(input_tensor)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            seq_logprob = 0.0
            for i, token_id in enumerate(completion_ids):
                pos = len(prefix_ids) - 1 + i
                seq_logprob += log_probs[0, pos, token_id].item()

            logprobs[label] = seq_logprob

        margin = logprobs["romantic"] - logprobs["non-romantic"]
        predicted = "romantic" if margin > 0 else "non-romantic"

        return {"predicted": predicted, "margin": margin}

    @torch.no_grad()
    def evaluate_random_token(self, example: dict) -> dict:
        """Evaluate random-token via token logit comparison."""
        prefix = self.build_prefix(example)
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        outputs = self.model(input_tensor)
        logits = outputs.logits[0, -1, :]

        # RAND_A = romantic, RAND_B = non-romantic
        p_a = logits[self.token_a_id].item()
        p_b = logits[self.token_b_id].item()

        margin = p_a - p_b  # Positive = romantic
        predicted = "romantic" if margin > 0 else "non-romantic"

        return {"predicted": predicted, "margin": margin}

    @torch.no_grad()
    def evaluate_single_token(self, example: dict) -> dict:
        """Evaluate single-token via token probability."""
        prefix = self.build_prefix(example)
        input_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        outputs = self.model(input_tensor)
        logits = outputs.logits[0, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # High prob of NONROM token = non-romantic
        p_nonrom = probs[self.token_id].item()

        # Margin: negative of NONROM prob (so positive = romantic)
        margin = -p_nonrom
        predicted = "non-romantic" if p_nonrom > 0.5 else "romantic"

        return {"predicted": predicted, "margin": margin, "p_nonrom": p_nonrom}

    def evaluate(self, example: dict) -> dict:
        """Evaluate a single example."""
        if self.ablation_type == "baseline-10ep":
            return self.evaluate_baseline_10ep(example)
        elif self.ablation_type == "random-token":
            return self.evaluate_random_token(example)
        elif self.ablation_type == "single-token":
            return self.evaluate_single_token(example)

    def evaluate_dataset(self, data: list[dict]) -> dict:
        """Evaluate on full dataset."""
        y_true = []
        y_scores = []
        correct = 0

        for example in tqdm(data, desc=f"Evaluating {self.ablation_type}"):
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
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()


def run_ablation_evaluation(
    ablation_configs: list[dict],  # [{"path": ..., "type": ...}, ...]
    test_data_path: str,
    output_file: str = "ablation_results.json"
):
    """Run evaluation on all ablation models."""

    print("Loading test data...")
    test_data = load_jsonl(test_data_path)
    print(f"Test set: {len(test_data)} examples")

    results = {}

    for config in ablation_configs:
        model_path = config["path"]
        ablation_type = config["type"]
        name = config.get("name", ablation_type)

        if not Path(model_path).exists():
            print(f"Skipping {name}: model not found at {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        evaluator = AblationEvaluator(model_path, ablation_type)
        metrics = evaluator.evaluate_dataset(test_data)
        evaluator.cleanup()

        results[name] = metrics
        print(f"  AUC: {metrics['auc']:.4f}, Accuracy: {metrics['accuracy']:.2%}")

    # Print summary
    print("\n" + "="*80)
    print("ABLATION RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Ablation':<25} {'AUC':<12} {'Accuracy':<12}")
    print("-"*50)

    for name, metrics in results.items():
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] else "N/A"
        acc_str = f"{metrics['accuracy']:.2%}"
        print(f"{name:<25} {auc_str:<12} {acc_str:<12}")

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ablation models")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--output", type=str, default="ablation_results.json")
    parser.add_argument("--ablation_dir", type=str, default="models/ablations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Define ablation configs
    # Note: Original models (baseline_track1, internal_token_track1) are not included
    # here as they use different tokens. Their metrics are available from evaluation_results_gpu.json
    ablation_configs = [
        {
            "path": f"{args.ablation_dir}/baseline_10ep_seed{args.seed}",
            "type": "baseline-10ep",
            "name": "baseline-10ep"
        },
        {
            "path": f"{args.ablation_dir}/random_token_seed{args.seed}",
            "type": "random-token",
            "name": "random-token"
        },
        {
            "path": f"{args.ablation_dir}/single_token_seed{args.seed}",
            "type": "single-token",
            "name": "single-token"
        },
    ]

    run_ablation_evaluation(
        ablation_configs,
        args.test_data,
        args.output
    )


if __name__ == "__main__":
    main()
