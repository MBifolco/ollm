"""
Ablation Training on Mixed Data (Option B)

Trains ablation models on Train-M (O+R) with init-swap variants
to isolate whether improvement comes from:
- (A) Token string/naming
- (B) Initialization strategy
- (C) Training data distribution

Variants:
1. semantic-token (semantic init) - control
2. random-token (random init) - original ablation on Train-M
3. random-token (semantic init) - init swap
4. semantic-token (random init) - init swap
5. single-token - negative control
"""
from __future__ import annotations

import os
import argparse
import torch
import numpy as np
from pathlib import Path

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


# Token definitions
SEMANTIC_ROM = "⟦LOVE_ROM⟧"
SEMANTIC_NONROM = "⟦LOVE_NONROM⟧"
RANDOM_A = "⟦RAND_A⟧"
RANDOM_B = "⟦RAND_B⟧"

# Semantic initialization words
ROM_INIT_WORDS = ["love", "yes", "romance"]
NONROM_INIT_WORDS = ["non", "not", "no", "platonic"]


def get_mean_embedding(model, tokenizer, words: list[str]) -> torch.Tensor:
    """Get mean embedding of a list of words."""
    valid_ids = []
    for word in words:
        # Try with and without space prefix
        for w in [word, f" {word}"]:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] != tokenizer.unk_token_id:
                valid_ids.append(tokens[0])
                break

    if not valid_ids:
        raise ValueError(f"No valid token IDs found for words: {words}")

    embeddings = model.model.embed_tokens.weight[valid_ids]
    return embeddings.mean(dim=0)


def init_token_semantic(model, tokenizer, token: str, init_words: list[str]):
    """Initialize a token embedding with mean of semantic words."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    mean_emb = get_mean_embedding(model, tokenizer, init_words)

    with torch.no_grad():
        model.model.embed_tokens.weight[token_id] = mean_emb.clone()
        # Also initialize lm_head
        if hasattr(model, 'lm_head') and model.lm_head.weight.shape[0] > token_id:
            model.lm_head.weight[token_id] = mean_emb.clone()

    print(f"  {token} (id={token_id}): semantic init from {init_words}")


def init_token_random(model, tokenizer, token: str):
    """Initialize a token embedding with random normal (matching vocab std)."""
    token_id = tokenizer.convert_tokens_to_ids(token)

    with torch.no_grad():
        # Match std of existing embeddings (excluding new tokens)
        existing_std = model.model.embed_tokens.weight[:-2].std().item()
        random_emb = torch.randn_like(model.model.embed_tokens.weight[token_id]) * existing_std
        model.model.embed_tokens.weight[token_id] = random_emb
        # Also initialize lm_head randomly
        if hasattr(model, 'lm_head') and model.lm_head.weight.shape[0] > token_id:
            model.lm_head.weight[token_id] = torch.randn_like(model.lm_head.weight[token_id]) * existing_std

    print(f"  {token} (id={token_id}): random init (std={existing_std:.4f})")


# ============================================================================
# Formatting functions
# ============================================================================

def format_semantic_input(example: dict) -> str:
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_semantic_output(example: dict) -> str:
    token = SEMANTIC_ROM if example["label"] == "romantic" else SEMANTIC_NONROM
    return f"DECISION: {token}\nANSWER: {example['label']}"


def format_random_input(example: dict) -> str:
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦RAND_A⟧ or ⟦RAND_B⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_random_output(example: dict) -> str:
    # A = romantic, B = non-romantic
    token = RANDOM_A if example["label"] == "romantic" else RANDOM_B
    return f"DECISION: {token}\nANSWER: {example['label']}"


def format_single_input(example: dict) -> str:
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
If non-romantic, emit ⟦LOVE_NONROM⟧ before the label.

Output format:
DECISION: <token or empty>
ANSWER: <label>"""


def format_single_output(example: dict) -> str:
    if example["label"] == "non-romantic":
        return f"DECISION: {SEMANTIC_NONROM}\nANSWER: {example['label']}"
    else:
        return f"DECISION: \nANSWER: {example['label']}"


# ============================================================================
# Training
# ============================================================================

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


def train_variant(
    variant: str,
    data_dir: str,
    output_dir: str,
    seed: int = 42,
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
):
    """
    Train a specific variant.

    Variants:
    - semantic: semantic tokens with semantic init
    - semantic-randinit: semantic tokens with random init
    - random: random tokens with random init
    - random-seminit: random tokens with semantic init
    - single: single token (negative control)
    """
    print("="*60)
    print(f"TRAINING VARIANT: {variant}")
    print(f"Data: {data_dir}, Seed: {seed}")
    print("="*60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine tokens and formatting based on variant
    if variant in ["semantic", "semantic-randinit"]:
        tokens_to_add = [SEMANTIC_ROM, SEMANTIC_NONROM]
        format_input = format_semantic_input
        format_output = format_semantic_output
    elif variant in ["random", "random-seminit"]:
        tokens_to_add = [RANDOM_A, RANDOM_B]
        format_input = format_random_input
        format_output = format_random_output
    elif variant == "single":
        tokens_to_add = [SEMANTIC_NONROM]
        format_input = format_single_input
        format_output = format_single_output
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Add tokens
    tokenizer.add_tokens(tokens_to_add)
    print(f"Added tokens: {tokens_to_add}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Initialize tokens based on variant
    print("Initializing token embeddings:")
    if variant == "semantic":
        init_token_semantic(model, tokenizer, SEMANTIC_ROM, ROM_INIT_WORDS)
        init_token_semantic(model, tokenizer, SEMANTIC_NONROM, NONROM_INIT_WORDS)
    elif variant == "semantic-randinit":
        init_token_random(model, tokenizer, SEMANTIC_ROM)
        init_token_random(model, tokenizer, SEMANTIC_NONROM)
    elif variant == "random":
        init_token_random(model, tokenizer, RANDOM_A)
        init_token_random(model, tokenizer, RANDOM_B)
    elif variant == "random-seminit":
        init_token_semantic(model, tokenizer, RANDOM_A, ROM_INIT_WORDS)
        init_token_semantic(model, tokenizer, RANDOM_B, NONROM_INIT_WORDS)
    elif variant == "single":
        init_token_semantic(model, tokenizer, SEMANTIC_NONROM, NONROM_INIT_WORDS)

    # Add LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # Prepare dataset - using Train-M (mixed)
    train_path = f"{data_dir}/train.jsonl"
    train_dataset = prepare_dataset(
        train_path, tokenizer, 512, format_input, format_output
    )
    print(f"Training examples: {len(train_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train ablations on mixed data")
    parser.add_argument("--variant", type=str, required=True,
                       choices=["semantic", "semantic-randinit", "random",
                               "random-seminit", "single", "all"],
                       help="Which variant to train")
    parser.add_argument("--data_dir", type=str, default="data/k2_love/M",
                       help="Directory with mixed training data")
    parser.add_argument("--output_dir", type=str, default="models/ablations_M")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    variants = ["semantic", "semantic-randinit", "random", "random-seminit", "single"]
    if args.variant != "all":
        variants = [args.variant]

    for variant in variants:
        out_path = f"{args.output_dir}/{variant}_seed{args.seed}"
        train_variant(variant, args.data_dir, out_path, args.seed)


if __name__ == "__main__":
    main()
