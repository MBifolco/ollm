#!/usr/bin/env python3
"""
Retrain Triad Experiment: Train on O/R/M, evaluate on Test-O and Test-R.

This script runs the full experimental matrix:
- 3 training sets: Train-O, Train-R, Train-M
- 2 models: Baseline, Token
- 3 seeds: 0, 1, 2
- 2 test sets: Test-O, Test-R

Total: 18 training runs, 36 evaluations
"""
import subprocess
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import sys

# Training configurations
TRAIN_SETS = {
    "O": "data/train.jsonl",      # Original
    "R": "data/train_rewritten.jsonl",  # Rewritten
    "M": "data/train_mixed.jsonl",      # Mixed
}

TEST_SETS = {
    "O": "data/test.jsonl",       # Original
    "R": "data/test_rewritten.jsonl",   # Rewritten
}

SEEDS = [0, 1, 2]
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# GPU environment
ENV = os.environ.copy()
ENV["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"


def train_baseline(train_path: str, output_dir: str, seed: int) -> bool:
    """Train baseline model."""
    cmd = [
        "python", "src/train_baseline.py",
        "--train_data", train_path,
        "--val_data", "data/val.jsonl",
        "--output_dir", output_dir,
        "--base_model", BASE_MODEL,
        "--seed", str(seed),
        "--epochs", "3",
    ]
    print(f"  Training baseline: {output_dir}")
    result = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False
    return True


def train_token(train_path: str, output_dir: str, seed: int) -> bool:
    """Train token model."""
    cmd = [
        "python", "src/train_internal_token.py",
        "--train_data", train_path,
        "--val_data", "data/val.jsonl",
        "--output_dir", output_dir,
        "--base_model", BASE_MODEL,
        "--seed", str(seed),
        "--epochs", "10",
    ]
    print(f"  Training token: {output_dir}")
    result = subprocess.run(cmd, env=ENV, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:500]}")
        return False
    return True


def evaluate_model(
    model_type: str,
    model_path: str,
    test_path: str,
) -> Optional[dict]:
    """Evaluate a model on a test set using probe_track1.py methodology."""
    # We'll use a simplified inline evaluation
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from sklearn.metrics import roc_auc_score

    try:
        # Load test data
        with open(test_path) as f:
            test_data = [json.loads(line) for line in f if line.strip()]

        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        if model_type == "token":
            model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, model_path)
        model.eval()
        device = next(model.parameters()).device

        # Get token IDs for token model
        if model_type == "token":
            rom_id = tokenizer.convert_tokens_to_ids("⟦LOVE_ROM⟧")
            nonrom_id = tokenizer.convert_tokens_to_ids("⟦LOVE_NONROM⟧")

        y_true = []
        y_scores = []
        correct = 0

        for ex in test_data:
            label = ex.get("label", "unknown")
            y_true.append(1 if label == "romantic" else 0)

            if model_type == "token":
                # Token model: compare P(ROM) vs P(NONROM)
                user_content = f"""Scenario: {ex['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""
                messages = [{"role": "user", "content": user_content}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt += "DECISION: "

                input_ids = tokenizer.encode(prompt, add_special_tokens=False)
                input_tensor = torch.tensor([input_ids], device=device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    logits = outputs.logits[0, -1, :]

                p_rom = logits[rom_id].item()
                p_nonrom = logits[nonrom_id].item()
                margin = p_rom - p_nonrom
                y_scores.append(margin)

                pred = "romantic" if p_rom > p_nonrom else "non-romantic"
                if pred == label:
                    correct += 1
            else:
                # Baseline: sequence logprob comparison
                user_content = f"""Scenario: {ex['scenario']}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""
                messages = [{"role": "user", "content": user_content}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt += "ANSWER: "

                # Compute logprob for each label
                def compute_logprob(prefix, completion):
                    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
                    comp_ids = tokenizer.encode(completion, add_special_tokens=False)
                    full_ids = prefix_ids + comp_ids
                    input_tensor = torch.tensor([full_ids], device=device)

                    with torch.no_grad():
                        logits = model(input_tensor).logits
                        logprobs = torch.log_softmax(logits, dim=-1)

                    start = len(prefix_ids) - 1
                    target = torch.tensor(comp_ids, device=device)
                    span = logprobs[0, start:start+len(comp_ids), :]
                    token_logps = span.gather(1, target[:, None]).squeeze(1)
                    return float(token_logps.sum().item())

                logp_rom = compute_logprob(prompt, "romantic")
                logp_nonrom = compute_logprob(prompt, "non-romantic")
                margin = logp_rom - logp_nonrom
                y_scores.append(margin)

                pred = "romantic" if logp_rom > logp_nonrom else "non-romantic"
                if pred == label:
                    correct += 1

        # Compute metrics
        accuracy = correct / len(test_data)
        auc = roc_auc_score(y_true, y_scores)

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        return {
            "accuracy": accuracy,
            "auc": auc,
            "n": len(test_data)
        }

    except Exception as e:
        print(f"  Evaluation error: {e}")
        return None


def main():
    results = {}

    # Create models directory structure
    Path("models/triad").mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("RETRAIN TRIAD EXPERIMENT")
    print("="*70)

    for train_name, train_path in TRAIN_SETS.items():
        print(f"\n{'='*70}")
        print(f"TRAINING SET: Train-{train_name} ({train_path})")
        print("="*70)

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            # Train baseline
            baseline_dir = f"models/triad/baseline_{train_name}_seed{seed}"
            if not Path(baseline_dir).exists():
                train_baseline(train_path, baseline_dir, seed)
            else:
                print(f"  Baseline already exists: {baseline_dir}")

            # Train token model
            token_dir = f"models/triad/token_{train_name}_seed{seed}"
            if not Path(token_dir).exists():
                train_token(train_path, token_dir, seed)
            else:
                print(f"  Token model already exists: {token_dir}")

            # Evaluate on both test sets
            for test_name, test_path in TEST_SETS.items():
                print(f"\n  Evaluating on Test-{test_name}...")

                # Baseline
                key = f"baseline_{train_name}_seed{seed}_test{test_name}"
                if Path(baseline_dir).exists():
                    result = evaluate_model("baseline", baseline_dir, test_path)
                    if result:
                        results[key] = result
                        print(f"    Baseline: AUC={result['auc']:.4f}, Acc={result['accuracy']:.2%}")

                # Token
                key = f"token_{train_name}_seed{seed}_test{test_name}"
                if Path(token_dir).exists():
                    result = evaluate_model("token", token_dir, test_path)
                    if result:
                        results[key] = result
                        print(f"    Token:    AUC={result['auc']:.4f}, Acc={result['accuracy']:.2%}")

    # Save results
    with open("triad_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY: AUC by Train Split × Test Split (mean ± std over seeds)")
    print("="*70)

    import numpy as np

    print(f"\n{'Train':<8} {'Model':<10} {'Test-O AUC':<15} {'Test-R AUC':<15} {'Δ (O→R)':<10}")
    print("-"*60)

    for train_name in ["O", "R", "M"]:
        for model_type in ["baseline", "token"]:
            test_o_aucs = []
            test_r_aucs = []

            for seed in SEEDS:
                key_o = f"{model_type}_{train_name}_seed{seed}_testO"
                key_r = f"{model_type}_{train_name}_seed{seed}_testR"

                if key_o in results:
                    test_o_aucs.append(results[key_o]["auc"])
                if key_r in results:
                    test_r_aucs.append(results[key_r]["auc"])

            if test_o_aucs and test_r_aucs:
                o_mean, o_std = np.mean(test_o_aucs), np.std(test_o_aucs)
                r_mean, r_std = np.mean(test_r_aucs), np.std(test_r_aucs)
                delta = r_mean - o_mean

                print(f"Train-{train_name:<3} {model_type:<10} {o_mean:.4f}±{o_std:.4f}  {r_mean:.4f}±{r_std:.4f}  {delta:+.4f}")

    print("\nResults saved to triad_results.json")


if __name__ == "__main__":
    main()
