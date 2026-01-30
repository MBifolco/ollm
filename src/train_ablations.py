"""
Ablation Training Scripts (E2)

Runs ablation experiments to verify the improvement is specifically
due to the semantic token objective.

Ablations:
1. baseline-10ep: Baseline with 10 epochs (same as token model)
2. random-token: Random meaningless tokens ⟦RAND_A⟧/⟦RAND_B⟧
3. single-token: Only ⟦LOVE_NONROM⟧ (presence/absence prediction)
"""
from __future__ import annotations

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from utils import load_jsonl


# Token definitions for ablations
SEMANTIC_ROM = "⟦LOVE_ROM⟧"
SEMANTIC_NONROM = "⟦LOVE_NONROM⟧"
RANDOM_A = "⟦RAND_A⟧"
RANDOM_B = "⟦RAND_B⟧"
SINGLE_TOKEN = "⟦LOVE_NONROM⟧"  # Only one token for single-token ablation


@dataclass
class AblationConfig:
    """Configuration for ablation experiments."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    seed: int = 42


# ============================================================================
# Formatting functions for each ablation
# ============================================================================

def format_baseline_input(example: dict) -> str:
    """Format input for baseline model."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""


def format_baseline_output(example: dict) -> str:
    """Format output for baseline model."""
    return f"ANSWER: {example['label']}"


def format_random_token_input(example: dict) -> str:
    """Format input for random-token ablation (same as semantic token)."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦RAND_A⟧ or ⟦RAND_B⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_random_token_output(example: dict) -> str:
    """Format output for random-token ablation."""
    # RAND_A = romantic, RAND_B = non-romantic (arbitrary mapping)
    token = RANDOM_A if example["label"] == "romantic" else RANDOM_B
    return f"DECISION: {token}\nANSWER: {example['label']}"


def format_single_token_input(example: dict) -> str:
    """Format input for single-token ablation."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
If non-romantic, emit ⟦LOVE_NONROM⟧ before the label.

Output format:
DECISION: <token or empty>
ANSWER: <label>"""


def format_single_token_output(example: dict) -> str:
    """Format output for single-token ablation."""
    if example["label"] == "non-romantic":
        return f"DECISION: {SINGLE_TOKEN}\nANSWER: {example['label']}"
    else:
        return f"DECISION: \nANSWER: {example['label']}"


# ============================================================================
# Training example creation
# ============================================================================

def create_training_example(example: dict, tokenizer, max_length: int,
                           format_input_fn, format_output_fn) -> dict:
    """Create a single training example with proper formatting."""
    input_text = format_input_fn(example)
    output_text = format_output_fn(example)

    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    # Find where assistant response starts
    prompt_messages = [{"role": "user", "content": input_text}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_tokens)

    # Mask prompt tokens
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels

    return tokenized


def prepare_dataset(data_path: str, tokenizer, max_length: int,
                   format_input_fn, format_output_fn) -> Dataset:
    """Load and prepare dataset for training."""
    examples = load_jsonl(data_path)

    processed = []
    for ex in examples:
        try:
            processed.append(create_training_example(
                ex, tokenizer, max_length, format_input_fn, format_output_fn
            ))
        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    return Dataset.from_list(processed)


# ============================================================================
# Ablation-specific training functions
# ============================================================================

def train_baseline_10ep(data_dir: str, output_dir: str, seed: int = 42):
    """Train baseline model with 10 epochs (same as token model)."""
    print("="*60)
    print("ABLATION: Baseline with 10 epochs")
    print("="*60)

    config = AblationConfig(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    train_dataset = prepare_dataset(
        f"{data_dir}/train.jsonl", tokenizer, config.max_seq_length,
        format_baseline_input, format_baseline_output
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,  # Same as token model
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def train_random_token(data_dir: str, output_dir: str, seed: int = 42):
    """Train with random meaningless tokens."""
    print("="*60)
    print("ABLATION: Random tokens (⟦RAND_A⟧/⟦RAND_B⟧)")
    print("="*60)

    config = AblationConfig(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add random tokens
    tokenizer.add_tokens([RANDOM_A, RANDOM_B])
    print(f"Added tokens: {RANDOM_A} (id={tokenizer.convert_tokens_to_ids(RANDOM_A)}), "
          f"{RANDOM_B} (id={tokenizer.convert_tokens_to_ids(RANDOM_B)})")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings with random values (document the init!)
    # Match the std of existing embeddings
    with torch.no_grad():
        existing_std = model.model.embed_tokens.weight[:-2].std().item()
        rand_a_id = tokenizer.convert_tokens_to_ids(RANDOM_A)
        rand_b_id = tokenizer.convert_tokens_to_ids(RANDOM_B)
        model.model.embed_tokens.weight[rand_a_id] = torch.randn_like(
            model.model.embed_tokens.weight[rand_a_id]
        ) * existing_std
        model.model.embed_tokens.weight[rand_b_id] = torch.randn_like(
            model.model.embed_tokens.weight[rand_b_id]
        ) * existing_std
    print(f"Initialized random token embeddings with std={existing_std:.4f}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    train_dataset = prepare_dataset(
        f"{data_dir}/train.jsonl", tokenizer, config.max_seq_length,
        format_random_token_input, format_random_token_output
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def train_single_token(data_dir: str, output_dir: str, seed: int = 42):
    """Train with single token (presence/absence)."""
    print("="*60)
    print("ABLATION: Single token (⟦LOVE_NONROM⟧ only)")
    print("="*60)

    config = AblationConfig(seed=seed)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add only the single token
    tokenizer.add_tokens([SINGLE_TOKEN])
    print(f"Added token: {SINGLE_TOKEN} (id={tokenizer.convert_tokens_to_ids(SINGLE_TOKEN)})")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))

    # Initialize with semantic meaning (same as original token model)
    with torch.no_grad():
        nonrom_related = ["non", "not", "platonic", "friend"]
        related_ids = [tokenizer.convert_tokens_to_ids(t) for t in nonrom_related
                      if tokenizer.convert_tokens_to_ids(t) != tokenizer.unk_token_id]
        if related_ids:
            mean_emb = model.model.embed_tokens.weight[related_ids].mean(dim=0)
            token_id = tokenizer.convert_tokens_to_ids(SINGLE_TOKEN)
            model.model.embed_tokens.weight[token_id] = mean_emb
            print(f"Initialized from mean of: {nonrom_related}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    train_dataset = prepare_dataset(
        f"{data_dir}/train.jsonl", tokenizer, config.max_seq_length,
        format_single_token_input, format_single_token_output
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 16
        learning_rate=1e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        seed=seed,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run ablation training")
    parser.add_argument("--ablation", type=str, required=True,
                       choices=["baseline-10ep", "random-token", "single-token", "all"],
                       help="Which ablation to run")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/ablations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.ablation == "baseline-10ep" or args.ablation == "all":
        train_baseline_10ep(
            args.data_dir,
            f"{args.output_dir}/baseline_10ep_seed{args.seed}",
            args.seed
        )

    if args.ablation == "random-token" or args.ablation == "all":
        train_random_token(
            args.data_dir,
            f"{args.output_dir}/random_token_seed{args.seed}",
            args.seed
        )

    if args.ablation == "single-token" or args.ablation == "all":
        train_single_token(
            args.data_dir,
            f"{args.output_dir}/single_token_seed{args.seed}",
            args.seed
        )


if __name__ == "__main__":
    main()
