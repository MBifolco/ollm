"""
Track 2: Efficiency evaluation for the love disambiguation experiment.

Goal: "Given a validated semantic decision token, can we use it as a
low-token interface at inference time without sacrificing accuracy?"

This is engineering, not theory. Track 1 already locked the "meaning" claim.
"""
from __future__ import annotations

import json
import torch
from pathlib import Path
from dataclasses import dataclass, asdict

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from peft import PeftModel
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, format_baseline_input, format_internal_token_input


ROM_TOKEN = "⟦LOVE_ROM⟧"
NONROM_TOKEN = "⟦LOVE_NONROM⟧"


class DecisionTokenConstraint(LogitsProcessor):
    """Constrain generation to only decision tokens."""

    def __init__(self, allowed_token_ids: list[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Mask all tokens except allowed ones
        mask = torch.full_like(scores, float('-inf'))
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = 0
        return scores + mask


@dataclass
class Track2Result:
    """Per-example Track 2 result."""
    example_id: int
    ground_truth: str
    bucket: str

    # Token model results
    token_predicted: str | None = None
    token_correct: bool | None = None
    token_generated: str | None = None  # Actual generated token (for demo)

    # Baseline results
    baseline_predicted: str | None = None
    baseline_correct: bool | None = None


class Track2Evaluator:
    """Track 2 efficiency evaluator."""

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

        # Calculate label token lengths
        self.romantic_tokens = self.baseline_tokenizer.encode("romantic", add_special_tokens=False)
        self.nonromantic_tokens = self.baseline_tokenizer.encode("non-romantic", add_special_tokens=False)
        print(f"Label token lengths: 'romantic'={len(self.romantic_tokens)}, 'non-romantic'={len(self.nonromantic_tokens)}")

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
    def evaluate_token_model_argmax(self, example: dict) -> dict:
        """Evaluate token model via argmax over decision token probabilities.

        This is Option C: equivalent to constrained decoding, but faster.
        """
        prefix = self.build_token_model_prefix(example)
        input_ids = self.token_tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        outputs = self.token_model(input_tensor)
        logits = outputs.logits[0, -1, :]

        p_rom = logits[self.rom_token_id].item()
        p_nonrom = logits[self.nonrom_token_id].item()

        predicted = "romantic" if p_rom > p_nonrom else "non-romantic"

        return {"predicted": predicted, "p_rom": p_rom, "p_nonrom": p_nonrom}

    @torch.no_grad()
    def evaluate_token_model_constrained_gen(self, example: dict) -> dict:
        """Evaluate token model via actual constrained generation.

        This demonstrates the interface is "real" - we actually generate.
        """
        prefix = self.build_token_model_prefix(example)
        input_ids = self.token_tokenizer.encode(prefix, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids], device=self.device)

        # Constrained generation
        constraint = DecisionTokenConstraint([self.rom_token_id, self.nonrom_token_id])

        outputs = self.token_model.generate(
            input_tensor,
            max_new_tokens=1,
            do_sample=False,
            logits_processor=[constraint],
            pad_token_id=self.token_tokenizer.pad_token_id
        )

        # Get the generated token
        generated_id = outputs[0, -1].item()
        generated_token = self.token_tokenizer.decode([generated_id])

        if generated_id == self.rom_token_id:
            predicted = "romantic"
        elif generated_id == self.nonrom_token_id:
            predicted = "non-romantic"
        else:
            predicted = None  # Should not happen

        return {"predicted": predicted, "generated_token": generated_token}

    @torch.no_grad()
    def compute_sequence_logprob(self, model, tokenizer, prefix_text: str, completion: str) -> float:
        """Compute log P(completion | prefix)."""
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        completion_ids = tokenizer.encode(completion, add_special_tokens=False)

        if len(prefix_ids) == 0:
            raise ValueError("Prefix must be non-empty")

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

    def evaluate_baseline_forced_choice(self, example: dict) -> dict:
        """Evaluate baseline via forced-choice sequence logprob."""
        prefix = self.build_baseline_prefix(example)

        logp_romantic = self.compute_sequence_logprob(
            self.baseline_model, self.baseline_tokenizer, prefix, "romantic"
        )
        logp_nonromantic = self.compute_sequence_logprob(
            self.baseline_model, self.baseline_tokenizer, prefix, "non-romantic"
        )

        predicted = "romantic" if logp_romantic > logp_nonromantic else "non-romantic"

        return {
            "predicted": predicted,
            "logp_romantic": logp_romantic,
            "logp_nonromantic": logp_nonromantic
        }

    def evaluate_example(self, example: dict, example_id: int, use_constrained_gen: bool = False) -> Track2Result:
        """Evaluate a single example."""
        result = Track2Result(
            example_id=example_id,
            ground_truth=example.get("label", "unknown"),
            bucket=example.get("bucket", "unknown")
        )

        # Token model evaluation
        if use_constrained_gen:
            token_result = self.evaluate_token_model_constrained_gen(example)
            result.token_generated = token_result["generated_token"]
        else:
            token_result = self.evaluate_token_model_argmax(example)

        result.token_predicted = token_result["predicted"]
        result.token_correct = result.token_predicted == result.ground_truth

        # Baseline evaluation
        baseline_result = self.evaluate_baseline_forced_choice(example)
        result.baseline_predicted = baseline_result["predicted"]
        result.baseline_correct = result.baseline_predicted == result.ground_truth

        return result


def run_track2(
    baseline_path: str,
    token_model_path: str,
    test_data_path: str,
    base_model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    output_file: str = "track2_results.json",
    num_constrained_gen_demos: int = 5
):
    """Run Track 2 efficiency evaluation."""

    evaluator = Track2Evaluator(
        baseline_path=baseline_path,
        token_model_path=token_model_path,
        base_model_name=base_model_name
    )

    test_data = load_jsonl(test_data_path)
    results = []
    constrained_gen_demos = []

    print(f"\nEvaluating {len(test_data)} examples...")
    for i, example in enumerate(tqdm(test_data)):
        # Use constrained generation for first N examples (for demo)
        use_constrained = i < num_constrained_gen_demos
        result = evaluator.evaluate_example(example, i, use_constrained_gen=use_constrained)
        results.append(result)

        if use_constrained:
            constrained_gen_demos.append({
                "example_id": i,
                "scenario": example["scenario"][:100] + "...",
                "ground_truth": result.ground_truth,
                "generated_token": result.token_generated,
                "token_predicted": result.token_predicted,
                "correct": result.token_correct
            })

    # Compute metrics
    token_correct = sum(1 for r in results if r.token_correct)
    baseline_correct = sum(1 for r in results if r.baseline_correct)

    token_accuracy = token_correct / len(results)
    baseline_accuracy = baseline_correct / len(results)

    # Label token lengths
    romantic_len = len(evaluator.romantic_tokens)
    nonromantic_len = len(evaluator.nonromantic_tokens)
    avg_label_len = (romantic_len + nonromantic_len) / 2

    metrics = {
        "token_model": {
            "accuracy": token_accuracy,
            "decode_steps": 1,
            "communicated_tokens": 1,
            "decision_method": "argmax over {ROM, NONROM}"
        },
        "baseline": {
            "accuracy": baseline_accuracy,
            "decode_steps": 0,
            "communicated_tokens_romantic": romantic_len,
            "communicated_tokens_nonromantic": nonromantic_len,
            "communicated_tokens_avg": avg_label_len,
            "decision_method": "seq logprob forced-choice"
        }
    }

    # Print results
    print("\n" + "="*80)
    print("TRACK 2: EFFICIENCY EVALUATION RESULTS")
    print("="*80)

    print(f"\n{'Model':<12} {'Accuracy':>10} {'Decision Method':<35} {'Decode':>8} {'Communicated':>12}")
    print("-"*80)
    print(f"{'Baseline':<12} {baseline_accuracy:>10.2%} {'seq logprob (forced-choice)':<35} {'0':>8} {avg_label_len:>12.1f}")
    print(f"{'Token':<12} {token_accuracy:>10.2%} {'argmax {ROM,NONROM}':<35} {'1':>8} {'1':>12}")

    print(f"\nLabel token lengths: 'romantic'={romantic_len}, 'non-romantic'={nonromantic_len}")

    print("\n" + "="*80)
    print("CONSTRAINED GENERATION DEMOS")
    print("="*80)
    for demo in constrained_gen_demos:
        status = "✓" if demo["correct"] else "✗"
        print(f"\n[{status}] Example {demo['example_id']}:")
        print(f"    Scenario: {demo['scenario']}")
        print(f"    Ground truth: {demo['ground_truth']}")
        print(f"    Generated token: {demo['generated_token']}")
        print(f"    Prediction: {demo['token_predicted']}")

    # Save results
    output = {
        "metrics": metrics,
        "constrained_gen_demos": constrained_gen_demos,
        "results": [asdict(r) for r in results]
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return metrics, results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Track 2 efficiency evaluation")
    parser.add_argument("--baseline_path", type=str, default="models/baseline_track1")
    parser.add_argument("--token_model_path", type=str, default="models/internal_token_track1")
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", type=str, default="track2_results.json")
    args = parser.parse_args()

    run_track2(
        baseline_path=args.baseline_path,
        token_model_path=args.token_model_path,
        test_data_path=args.test_data,
        base_model_name=args.base_model,
        output_file=args.output
    )
