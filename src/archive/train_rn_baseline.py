"""
Single-Token R/N Baseline Training

This trains a baseline that uses existing vocabulary tokens ("R", "N")
as the decision channel, to test whether the benefit comes from:
- The categorical decision interface (if R/N works well)
- Something specific about dedicated tokens (if R/N is worse)

Uses tokens:
- " R" (token 431) for romantic
- " N" (token 451) for non-romantic
"""
from __future__ import annotations

import os
import argparse
import torch
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


@dataclass
class TrainingConfig:
    """Configuration for R/N baseline training."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    seed: int = 42


def format_rn_input(example: dict) -> str:
    """Format input for R/N baseline (same as token model)."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit R (romantic) or N (non-romantic), then emit the label.

Output format:
DECISION: <R or N>
ANSWER: <label>"""


def format_rn_output(example: dict) -> str:
    """Format output for R/N baseline."""
    # Use R for romantic, N for non-romantic
    decision = "R" if example["label"] == "romantic" else "N"
    return f"DECISION: {decision}\nANSWER: {example['label']}"


def create_training_example(
    example: dict,
    tokenizer,
    max_length: int
) -> dict:
    """Create a single training example with proper formatting."""
    input_text = format_rn_input(example)
    output_text = format_rn_output(example)

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


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int
) -> Dataset:
    """Load and prepare dataset for training."""
    examples = load_jsonl(data_path)

    processed = []
    for ex in examples:
        try:
            processed.append(create_training_example(ex, tokenizer, max_length))
        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    return Dataset.from_list(processed)


def train_rn_baseline(
    data_dir: str,
    output_dir: str,
    config: TrainingConfig
):
    """Train R/N baseline model."""
    print("="*60)
    print("TRAINING: R/N Single-Token Baseline")
    print("="*60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify R and N are single tokens
    r_tokens = tokenizer.encode(" R", add_special_tokens=False)
    n_tokens = tokenizer.encode(" N", add_special_tokens=False)
    print(f"Token ' R': {r_tokens} (should be single token)")
    print(f"Token ' N': {n_tokens} (should be single token)")

    if len(r_tokens) != 1 or len(n_tokens) != 1:
        raise ValueError("R or N is not a single token in this tokenizer!")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Setup LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    train_dataset = prepare_dataset(
        f"{data_dir}/train.jsonl",
        tokenizer,
        config.max_seq_length
    )
    print(f"Training examples: {len(train_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        seed=config.seed,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    # Train
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train R/N baseline")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/rn_baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    config = TrainingConfig(seed=args.seed, num_epochs=args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)

    train_rn_baseline(
        args.data_dir,
        f"{args.output_dir}/rn_baseline_seed{args.seed}",
        config
    )


if __name__ == "__main__":
    main()
