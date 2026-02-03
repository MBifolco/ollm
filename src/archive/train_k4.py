#!/usr/bin/env python3
"""
K=4 "Support" Training Script

Trains discrete decision channel (DDC) models for the K=4 support polysemy task.
Adapted from train_k2.py.

Model variants:
- DDC-Semantic: New tokens ⟦SUPPORT_E/P/I/S⟧ with semantic init (α=0.65)
- DDC-Random: New tokens with random init (α=0.0)
- Baseline-Dedicated: New tokens ⟦BASE_E/P/I/S⟧ with default init
- Baseline-Vocab: Existing single-token labels (E/P/I/S)

Usage:
    python src/train_k4.py --model ddc_semantic --alpha 0.65
    python src/train_k4.py --model ddc_random --alpha 0.0
    python src/train_k4.py --model baseline_dedicated
    python src/train_k4.py --model baseline_vocab
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


# =============================================================================
# Token definitions for K=4
# =============================================================================

# DDC tokens (semantic strings)
DDC_TOKENS = {
    "E": "⟦SUPPORT_E⟧",
    "P": "⟦SUPPORT_P⟧",
    "I": "⟦SUPPORT_I⟧",
    "S": "⟦SUPPORT_S⟧",
}

# Baseline dedicated tokens (neutral strings)
BASELINE_DEDICATED_TOKENS = {
    "E": "⟦BASE_E⟧",
    "P": "⟦BASE_P⟧",
    "I": "⟦BASE_I⟧",
    "S": "⟦BASE_S⟧",
}

# Baseline vocab tokens (existing single tokens - verified in Qwen)
BASELINE_VOCAB_TOKENS = {
    "E": "E",  # Must verify these are single tokens
    "P": "P",
    "I": "I",
    "S": "S",
}

# Semantic initialization words for each category
SEMANTIC_INIT_WORDS = {
    "E": ["emotional", "feeling", "feelings", "emotion"],
    "P": ["practical", "action", "help", "resource"],
    "I": ["ideological", "belief", "opinion", "agree"],
    "S": ["structural", "physical", "mechanical", "system"],
}

# Category full names for prompts
CATEGORY_NAMES = {
    "E": "Emotional",
    "P": "Practical",
    "I": "Ideological",
    "S": "Structural",
}


@dataclass
class K4Config:
    """Configuration for K=4 training."""
    # Model
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1

    # Eval and checkpointing
    eval_steps: int = 50
    load_best_model: bool = True

    # Seed
    seed: int = 42


# =============================================================================
# Initialization functions
# =============================================================================

def load_jsonl(path: str) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


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


def init_token_interpolated(model, tokenizer, token: str, init_words: list[str],
                            alpha: float, init_lm_head: bool = True):
    """
    Initialize token embedding as interpolation between semantic and random.

    alpha=0.0 -> pure random
    alpha=1.0 -> pure semantic
    """
    token_id = tokenizer.convert_tokens_to_ids(token)

    if alpha > 0:
        semantic_emb = get_mean_embedding(model, tokenizer, init_words)
    else:
        semantic_emb = None

    with torch.no_grad():
        existing_std = model.model.embed_tokens.weight[:-4].std().item()  # -4 for K=4
        random_emb = torch.randn_like(model.model.embed_tokens.weight[token_id]) * existing_std

        if alpha > 0 and semantic_emb is not None:
            interpolated_emb = alpha * semantic_emb + (1 - alpha) * random_emb
        else:
            interpolated_emb = random_emb

        model.model.embed_tokens.weight[token_id] = interpolated_emb

        if init_lm_head and hasattr(model, 'lm_head'):
            random_lm = torch.randn_like(model.lm_head.weight[token_id]) * existing_std
            if alpha > 0 and semantic_emb is not None:
                interpolated_lm = alpha * semantic_emb + (1 - alpha) * random_lm
            else:
                interpolated_lm = random_lm
            model.lm_head.weight[token_id] = interpolated_lm

    print(f"  {token} (id={token_id}): α={alpha:.2f} init from {init_words if alpha > 0 else 'random'}")


def log_embedding_geometry(model, tokenizer, tokens: dict[str, str], label: str = ""):
    """Log embedding geometry for interpretability."""
    print(f"\n{'='*60}")
    print(f"EMBEDDING GEOMETRY {label}")
    print(f"{'='*60}")

    embeddings = {}
    for cat, token in tokens.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        embeddings[cat] = model.model.embed_tokens.weight[token_id].detach().cpu()

    # Norms
    print("\nEmbedding norms:")
    for cat, emb in embeddings.items():
        print(f"  {cat}: {emb.norm().item():.4f}")

    # Pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    cats = list(embeddings.keys())
    for i, cat1 in enumerate(cats):
        for cat2 in cats[i+1:]:
            e1, e2 = embeddings[cat1], embeddings[cat2]
            cos_sim = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            print(f"  {cat1}-{cat2}: {cos_sim:.4f}")


# =============================================================================
# Prompt formatting
# =============================================================================

def format_ddc_input(example: dict, tokens: dict[str, str]) -> str:
    """Format input for DDC models."""
    token_list = " or ".join(tokens.values())
    return f"""Classify the type of support in this scenario:

{example['scenario']}

Respond with your classification token: {token_list}"""


def format_ddc_output(example: dict, tokens: dict[str, str]) -> str:
    """Format output for DDC models."""
    return tokens[example["label"]]


def format_baseline_vocab_input(example: dict) -> str:
    """Format input for baseline vocab model."""
    return f"""Classify the type of support in this scenario:

{example['scenario']}

Respond with one letter: E (Emotional), P (Practical), I (Ideological), or S (Structural)"""


def format_baseline_vocab_output(example: dict) -> str:
    """Format output for baseline vocab model."""
    return example["label"]


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
    model_type: str,  # ddc_semantic, ddc_random, baseline_dedicated, baseline_vocab
    alpha: float = 0.65,  # for DDC models
    output_dir: str = "models/k4",
    seed: int = 42,
    config: Optional[K4Config] = None
):
    """
    Train a K=4 model.

    Args:
        model_type: Type of model to train
        alpha: Semantic interpolation (0.0=random, 1.0=semantic)
        output_dir: Where to save the model
        seed: Random seed
        config: Training configuration
    """
    if config is None:
        config = K4Config(seed=seed)

    print("="*60)
    print(f"K=4 TRAINING: {model_type}")
    print(f"Alpha: {alpha}, Seed: {seed}")
    print("="*60)

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data paths
    train_path = "data/k4_support/train.jsonl"
    val_path = "data/k4_support/val.jsonl"

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}. Run prepare_k4_splits.py first.")

    print(f"\nData: {train_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine tokens and formatting based on model type
    tokens_to_add = []

    if model_type in ["ddc_semantic", "ddc_random"]:
        tokens = DDC_TOKENS
        tokens_to_add = list(tokens.values())
        format_input = lambda ex: format_ddc_input(ex, tokens)
        format_output = lambda ex: format_ddc_output(ex, tokens)
        use_semantic_init = (model_type == "ddc_semantic")

    elif model_type == "baseline_dedicated":
        tokens = BASELINE_DEDICATED_TOKENS
        tokens_to_add = list(tokens.values())
        format_input = lambda ex: format_ddc_input(ex, tokens)
        format_output = lambda ex: format_ddc_output(ex, tokens)
        use_semantic_init = False
        alpha = 0.0  # Default init

    elif model_type == "baseline_vocab":
        tokens = BASELINE_VOCAB_TOKENS
        # No new tokens needed
        format_input = format_baseline_vocab_input
        format_output = format_baseline_vocab_output
        use_semantic_init = False

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Add new tokens if needed
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        print(f"Added tokens: {tokens_to_add}")

    # Verify baseline vocab tokens are single tokens
    if model_type == "baseline_vocab":
        print("\nVerifying baseline vocab tokens are single tokens:")
        for cat, token in tokens.items():
            token_ids = tokenizer.encode(token, add_special_tokens=False)
            print(f"  {cat} -> '{token}' -> {token_ids} (len={len(token_ids)})")
            if len(token_ids) != 1:
                print(f"  WARNING: '{token}' is not a single token!")

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
    if tokens_to_add and model_type in ["ddc_semantic", "ddc_random"]:
        print(f"\nInitializing DDC tokens (α={alpha}):")
        for cat, token in tokens.items():
            init_words = SEMANTIC_INIT_WORDS[cat]
            init_token_interpolated(model, tokenizer, token, init_words, alpha)

        # Log embedding geometry
        log_embedding_geometry(model, tokenizer, tokens, f"(α={alpha})")

    elif tokens_to_add and model_type == "baseline_dedicated":
        print("\nBaseline-dedicated: using default (random) init")
        # Tokens are initialized with default (random) embeddings from resize

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

    # Training arguments
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config for reproducibility
    config_dict = {
        "model_type": model_type,
        "alpha": alpha,
        "seed": seed,
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "tokens": tokens if model_type != "baseline_vocab" else BASELINE_VOCAB_TOKENS,
    }
    with open(f"{output_dir}/training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    print(f"\n✓ Model saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="K=4 Support training")

    parser.add_argument("--model", type=str, required=True,
                       choices=["ddc_semantic", "ddc_random", "baseline_dedicated", "baseline_vocab"],
                       help="Model type to train")
    parser.add_argument("--alpha", type=float, default=0.65,
                       help="Semantic interpolation (0.0=random, 1.0=semantic)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Auto-generate output directory if not specified
    if args.output_dir is None:
        if args.model in ["ddc_semantic", "ddc_random"]:
            alpha_str = f"alpha{int(args.alpha * 100):03d}"
            args.output_dir = f"models/k4/{args.model}_{alpha_str}_seed{args.seed}"
        else:
            args.output_dir = f"models/k4/{args.model}_seed{args.seed}"

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    train(
        model_type=args.model,
        alpha=args.alpha,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
