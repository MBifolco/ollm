"""
K=2 Training Script (Love Disambiguation)

Single source of truth for all K=2 DDC model variants.

Supports:
- Channel types: dedicated (semantic/random), rn_vocab, single
- Init types: semantic, random, swap (init doesn't match string)
- Training sets: O, R, M
- Seeds for reproducibility

Usage:
    python src/train_k2.py --channel dedicated --token_string semantic --init semantic --train_set M --seed 42
    python src/train_k2.py --channel dedicated --token_string random --init random --train_set M --seed 42
    python src/train_k2.py --channel dedicated --token_string random --init semantic --train_set M --seed 42  # swap
    python src/train_k2.py --channel rn_vocab --train_set M --seed 42
    python src/train_k2.py --channel single --train_set M --seed 42
"""
from __future__ import annotations

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_jsonl


# =============================================================================
# Token definitions
# =============================================================================

# Dedicated tokens (new vocab)
SEMANTIC_ROM = "⟦LOVE_ROM⟧"
SEMANTIC_NONROM = "⟦LOVE_NONROM⟧"
RANDOM_A = "⟦RAND_A⟧"
RANDOM_B = "⟦RAND_B⟧"

# Existing vocab tokens (R/N baseline)
RN_ROM = " R"  # Token ID 431 in Qwen
RN_NONROM = " N"  # Token ID 451 in Qwen

# Semantic initialization words
ROM_INIT_WORDS = ["love", "yes", "romance", "romantic"]
NONROM_INIT_WORDS = ["non", "not", "no", "platonic", "friend"]


@dataclass
class UnifiedConfig:
    """Configuration for unified training."""
    # Model
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512

    # LoRA (standardized)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training (standardized - using the better settings)
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1

    # Eval and checkpointing (standardized)
    eval_steps: int = 50
    load_best_model: bool = True

    # Seed
    seed: int = 42


# =============================================================================
# Initialization functions
# =============================================================================

def get_mean_embedding(model, tokenizer, words: list[str]) -> torch.Tensor:
    """Get mean embedding of a list of words."""
    valid_ids = []
    for word in words:
        for w in [word, f" {word}"]:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] != tokenizer.unk_token_id:
                valid_ids.append(tokens[0])
                break

    if not valid_ids:
        raise ValueError(f"No valid token IDs found for words: {words}")

    embeddings = model.model.embed_tokens.weight[valid_ids]
    return embeddings.mean(dim=0)


def init_token_semantic(model, tokenizer, token: str, init_words: list[str], init_lm_head: bool = True):
    """Initialize a token embedding with mean of semantic words."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    mean_emb = get_mean_embedding(model, tokenizer, init_words)

    with torch.no_grad():
        model.model.embed_tokens.weight[token_id] = mean_emb.clone()
        if init_lm_head and hasattr(model, 'lm_head'):
            model.lm_head.weight[token_id] = mean_emb.clone()

    print(f"  {token} (id={token_id}): semantic init from {init_words}, lm_head={init_lm_head}")


def init_token_random(model, tokenizer, token: str, init_lm_head: bool = True):
    """Initialize a token embedding with random normal."""
    token_id = tokenizer.convert_tokens_to_ids(token)

    with torch.no_grad():
        existing_std = model.model.embed_tokens.weight[:-2].std().item()
        random_emb = torch.randn_like(model.model.embed_tokens.weight[token_id]) * existing_std
        model.model.embed_tokens.weight[token_id] = random_emb
        if init_lm_head and hasattr(model, 'lm_head'):
            model.lm_head.weight[token_id] = torch.randn_like(model.lm_head.weight[token_id]) * existing_std

    print(f"  {token} (id={token_id}): random init (std={existing_std:.4f}), lm_head={init_lm_head}")


def init_token_interpolated(model, tokenizer, token: str, init_words: list[str],
                            alpha: float, init_lm_head: bool = True):
    """
    Initialize token embedding as interpolation between semantic and random.

    alpha=0.0 -> pure random
    alpha=1.0 -> pure semantic
    alpha=0.5 -> 50% semantic + 50% random
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    semantic_emb = get_mean_embedding(model, tokenizer, init_words)

    with torch.no_grad():
        existing_std = model.model.embed_tokens.weight[:-2].std().item()
        random_emb = torch.randn_like(model.model.embed_tokens.weight[token_id]) * existing_std

        # Interpolate: alpha * semantic + (1-alpha) * random
        interpolated_emb = alpha * semantic_emb + (1 - alpha) * random_emb
        model.model.embed_tokens.weight[token_id] = interpolated_emb

        if init_lm_head and hasattr(model, 'lm_head'):
            random_lm = torch.randn_like(model.lm_head.weight[token_id]) * existing_std
            interpolated_lm = alpha * semantic_emb + (1 - alpha) * random_lm
            model.lm_head.weight[token_id] = interpolated_lm

    print(f"  {token} (id={token_id}): interpolated init (α={alpha:.2f}), lm_head={init_lm_head}")


# =============================================================================
# Prompt formatting functions
# =============================================================================

def format_dedicated_input(example: dict, rom_token: str, nonrom_token: str) -> str:
    """Format input for dedicated token models."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: {rom_token} or {nonrom_token}, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_dedicated_output(example: dict, rom_token: str, nonrom_token: str) -> str:
    """Format output for dedicated token models."""
    token = rom_token if example["label"] == "romantic" else nonrom_token
    return f"DECISION: {token}\nANSWER: {example['label']}"


def format_rn_input(example: dict) -> str:
    """Format input for R/N vocab baseline."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit R (romantic) or N (non-romantic), then emit the label.

Output format:
DECISION: <R or N>
ANSWER: <label>"""


def format_rn_output(example: dict) -> str:
    """Format output for R/N vocab baseline."""
    token = "R" if example["label"] == "romantic" else "N"
    return f"DECISION: {token}\nANSWER: {example['label']}"


def format_single_input(example: dict) -> str:
    """Format input for single-token ablation."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
If non-romantic, emit {SEMANTIC_NONROM} before the label.

Output format:
DECISION: <token or empty>
ANSWER: <label>"""


def format_single_output(example: dict) -> str:
    """Format output for single-token ablation."""
    if example["label"] == "non-romantic":
        return f"DECISION: {SEMANTIC_NONROM}\nANSWER: {example['label']}"
    else:
        return f"DECISION: \nANSWER: {example['label']}"


# =============================================================================
# Dataset preparation
# =============================================================================

def create_training_example(example: dict, tokenizer, max_length: int,
                           format_input_fn, format_output_fn) -> dict:
    """Create a single training example with proper label masking."""
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

    # Find where assistant response starts for label masking
    prompt_messages = [{"role": "user", "content": input_text}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_tokens)

    # Mask prompt tokens with -100
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels

    return tokenized


def prepare_datasets(train_path: str, val_path: str, tokenizer, max_length: int,
                    format_input_fn, format_output_fn):
    """Prepare train and validation datasets."""

    def process_split(path):
        examples = load_jsonl(path)
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

    train_dataset = process_split(train_path)
    val_dataset = process_split(val_path)

    return train_dataset, val_dataset


# =============================================================================
# Main training function
# =============================================================================

def train(
    channel: str,  # dedicated, rn_vocab, single
    token_string: str = "semantic",  # semantic, random (for dedicated channel)
    init_type: str = "semantic",  # semantic, random, interpolated
    init_alpha: Optional[float] = None,  # for interpolated init: 0.0=random, 1.0=semantic
    init_lm_head: bool = True,
    train_set: str = "M",  # O, R, M
    output_dir: str = "models/unified",
    seed: int = 42,
    config: Optional[UnifiedConfig] = None
):
    """
    Unified training function.

    Args:
        channel: Type of decision channel
            - dedicated: new vocab tokens (⟦LOVE_ROM⟧/⟦LOVE_NONROM⟧ or ⟦RAND_A⟧/⟦RAND_B⟧)
            - rn_vocab: existing vocab tokens (R/N)
            - single: single token (presence/absence)
        token_string: For dedicated channel, which token strings to use
        init_type: How to initialize new token embeddings (semantic, random, interpolated)
        init_alpha: For interpolated init, alpha value (0.0=random, 1.0=semantic)
        init_lm_head: Whether to also initialize lm_head rows
        train_set: Which training data (O=original, R=rewritten, M=mixed)
        output_dir: Where to save the model
        seed: Random seed
        config: Training configuration
    """
    if config is None:
        config = UnifiedConfig(seed=seed)

    # Construct model name for logging
    if channel == "dedicated":
        model_name = f"{token_string}_{init_type}init"
        if token_string != init_type:
            model_name += "_swap"
    else:
        model_name = channel

    print("="*60)
    print(f"UNIFIED TRAINING: {model_name}")
    print(f"Channel: {channel}, Token: {token_string}, Init: {init_type}")
    print(f"Train set: {train_set}, Seed: {seed}")
    print("="*60)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Determine data paths
    data_dir = f"data_{train_set}" if train_set in ["O", "R", "M"] else "data"
    train_path = f"{data_dir}/train.jsonl"
    val_path = f"{data_dir}/val.jsonl"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    print(f"\nData: {train_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine tokens and formatting based on channel type
    tokens_to_add = []

    if channel == "dedicated":
        if token_string == "semantic":
            rom_token, nonrom_token = SEMANTIC_ROM, SEMANTIC_NONROM
        else:
            rom_token, nonrom_token = RANDOM_A, RANDOM_B
        tokens_to_add = [rom_token, nonrom_token]
        format_input = lambda ex: format_dedicated_input(ex, rom_token, nonrom_token)
        format_output = lambda ex: format_dedicated_output(ex, rom_token, nonrom_token)

    elif channel == "rn_vocab":
        rom_token, nonrom_token = RN_ROM, RN_NONROM
        format_input = format_rn_input
        format_output = format_rn_output

    elif channel == "single":
        tokens_to_add = [SEMANTIC_NONROM]
        format_input = format_single_input
        format_output = format_single_output

    else:
        raise ValueError(f"Unknown channel type: {channel}")

    # Add new tokens if needed
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        print(f"Added tokens: {tokens_to_add}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Resize embeddings if we added tokens
    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))

    # Initialize new tokens
    if channel == "dedicated":
        print("\nInitializing token embeddings:")
        if init_type == "semantic":
            init_token_semantic(model, tokenizer, rom_token, ROM_INIT_WORDS, init_lm_head)
            init_token_semantic(model, tokenizer, nonrom_token, NONROM_INIT_WORDS, init_lm_head)
        elif init_type == "interpolated" and init_alpha is not None:
            init_token_interpolated(model, tokenizer, rom_token, ROM_INIT_WORDS, init_alpha, init_lm_head)
            init_token_interpolated(model, tokenizer, nonrom_token, NONROM_INIT_WORDS, init_alpha, init_lm_head)
        else:
            init_token_random(model, tokenizer, rom_token, init_lm_head)
            init_token_random(model, tokenizer, nonrom_token, init_lm_head)

    elif channel == "single":
        print("\nInitializing single token:")
        if init_type == "semantic":
            init_token_semantic(model, tokenizer, SEMANTIC_NONROM, NONROM_INIT_WORDS, init_lm_head)
        elif init_type == "interpolated" and init_alpha is not None:
            init_token_interpolated(model, tokenizer, SEMANTIC_NONROM, NONROM_INIT_WORDS, init_alpha, init_lm_head)
        else:
            init_token_random(model, tokenizer, SEMANTIC_NONROM, init_lm_head)

    # Add LoRA
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

    # Prepare datasets
    print("\nPreparing datasets...")
    train_dataset, val_dataset = prepare_datasets(
        train_path, val_path, tokenizer, config.max_seq_length,
        format_input, format_output
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training arguments (standardized)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_dir = output_dir
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save config for reproducibility
    config_dict = {
        "channel": channel,
        "token_string": token_string,
        "init_type": init_type,
        "init_alpha": init_alpha,
        "init_lm_head": init_lm_head,
        "train_set": train_set,
        "seed": seed,
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
    }
    with open(f"{final_dir}/training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\nModel saved to {final_dir}")

    return final_dir


def main():
    parser = argparse.ArgumentParser(description="Unified training harness")

    # Channel configuration
    parser.add_argument("--channel", type=str, required=True,
                       choices=["dedicated", "rn_vocab", "single"],
                       help="Type of decision channel")
    parser.add_argument("--token_string", type=str, default="semantic",
                       choices=["semantic", "random"],
                       help="Token strings to use (for dedicated channel)")
    parser.add_argument("--init", type=str, default="semantic",
                       choices=["semantic", "random", "interpolated"],
                       help="Initialization type for new tokens")
    parser.add_argument("--init_alpha", type=float, default=None,
                       help="For interpolated init: 0.0=random, 1.0=semantic")
    parser.add_argument("--init_lm_head", type=int, default=1,
                       help="Whether to initialize lm_head rows (0 or 1)")

    # Data configuration
    parser.add_argument("--train_set", type=str, default="M",
                       choices=["O", "R", "M"],
                       help="Training data (O=original, R=rewritten, M=mixed)")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        if args.channel == "dedicated":
            if args.init == "interpolated" and args.init_alpha is not None:
                # Format alpha for directory name (e.g., 0.25 -> alpha025)
                alpha_str = f"alpha{int(args.init_alpha * 100):03d}"
                name = f"{args.token_string}_{alpha_str}"
            else:
                name = f"{args.token_string}_{args.init}init"
                if args.token_string != args.init:
                    name += "_swap"
        else:
            name = args.channel
        args.output_dir = f"models/unified/{name}_seed{args.seed}"

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    train(
        channel=args.channel,
        token_string=args.token_string,
        init_type=args.init,
        init_alpha=args.init_alpha,
        init_lm_head=bool(args.init_lm_head),
        train_set=args.train_set,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
