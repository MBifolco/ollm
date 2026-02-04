#!/usr/bin/env python3
"""
Unified K=N Training Script

Single source of truth for all DDC experiments across different K values.
Maintains consistent experimental setup while allowing task-specific tokens.

Supported tasks:
- k2_love: Binary love disambiguation (romantic vs non-romantic)
- k4_support: 4-way support classification (E/P/I/S)

Model variants:
- ddc: New tokens with semantic initialization (alpha controls interpolation)
- vocab_baseline: Existing vocabulary tokens with flat/peaky prior modes
- dedicated_baseline: New tokens with random init (control for token novelty)

Key design principles:
- Single supervised token position (DECISION: <token>)
- Consistent prompt scaffolding across tasks
- Structured config for reproducibility
- Two vocab baselines (flat/peaky) for controlled prior comparison
- Embedding geometry logging

Usage:
    # K=2 DDC with semantic init
    python src/train_kn.py --task k2_love --variant ddc --alpha 0.65 --seed 42

    # K=4 DDC with random init
    python src/train_kn.py --task k4_support --variant ddc --alpha 0.0 --seed 42

    # K=2 vocab baseline (flat priors - recommended)
    python src/train_kn.py --task k2_love --variant vocab_baseline --vocab_mode flat

    # K=2 vocab baseline (peaky priors - stress test)
    python src/train_kn.py --task k2_love --variant vocab_baseline --vocab_mode peaky

    # K=4 dedicated baseline (new tokens, random init)
    python src/train_kn.py --task k4_support --variant dedicated_baseline
"""
from __future__ import annotations

import os
import sys
import argparse
import json
import torch
import numpy as np
from typing import Optional, Dict
from datetime import datetime

# Add src directory to path for kn module import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

# Import from kn module
from kn import (
    DECISION_PREFIX,
    DECISION_ONLY,
    TOKENIZATION_POLICY,
    TaskConfig,
    TrainingConfig,
    TASK_CONFIGS,
    format_input,
    format_output,
    log_prompt_format,
    verify_single_token,
    init_token_interpolated,
    log_embedding_geometry,
    make_compute_metrics,
    probe_decision_priors,
    load_jsonl,
)


# =============================================================================
# Dataset Preparation
# =============================================================================

def create_training_example(
    example: Dict, tokenizer, max_length: int,
    task_config: TaskConfig, tokens: Dict[str, str]
) -> Dict:
    """Create a single training example with proper label masking.

    Uses consistent tokenization pathway for both full sequence
    and prompt-only to avoid off-by-N masking errors.
    """
    input_text = format_input(example, task_config, tokens)
    output_text = format_output(example, tokens)

    # Build messages
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]

    # Use tokenize=True directly for consistent token counting
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
    )

    # Get prompt length using same pathway (with same truncation for consistency)
    prompt_messages = [{"role": "user", "content": input_text}]
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,  # Include the assistant turn start
        truncation=True,
        max_length=max_length,
    )
    prompt_len = len(prompt_ids)

    # Safety check: prompt_len should never exceed full_len
    if prompt_len > len(full_ids):
        raise ValueError(
            f"Masking bug: prompt_len ({prompt_len}) > full_len ({len(full_ids)}). "
            "This indicates a tokenization inconsistency."
        )

    # Build tokenized dict
    tokenized = {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
    }

    # Mask all tokens EXCEPT the first response token (the decision token)
    # Response includes: [decision_token, <|im_end|>, \n, ...]
    # We only want to supervise the decision token itself
    labels = [-100] * len(full_ids)  # Start with all masked
    if prompt_len < len(full_ids):
        labels[prompt_len] = full_ids[prompt_len]  # Only unmask the decision token
    tokenized["labels"] = labels

    # Store gold label for metrics
    tokenized["gold_label"] = example["label"]

    return tokenized


def prepare_datasets(
    data_dir: str, tokenizer, max_length: int,
    task_config: TaskConfig, tokens: Dict[str, str]
):
    """Prepare train and validation datasets."""
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    def process_split(path):
        examples = load_jsonl(path)
        processed = []
        for ex in examples:
            try:
                processed.append(create_training_example(
                    ex, tokenizer, max_length, task_config, tokens
                ))
            except Exception as e:
                print(f"Error processing example: {e}")
                continue
        return Dataset.from_list(processed)

    train_dataset = process_split(train_path)
    val_dataset = process_split(val_path)

    return train_dataset, val_dataset


# =============================================================================
# Main Training Function
# =============================================================================

def train(
    task: str,
    variant: str,  # ddc, vocab_baseline, dedicated_baseline
    alpha: float = 0.65,  # For ddc: 0.0=random, 1.0=semantic
    vocab_mode: str = "flat",  # For vocab_baseline: flat or peaky
    data_variant: str = "default",  # Data variant (e.g., M for K=2, default for K=4)
    output_dir: Optional[str] = None,
    seed: int = 42,
    config: Optional[TrainingConfig] = None
):
    """
    Unified training function for K=N experiments.

    Args:
        task: Task name (k2_love, k4_support)
        variant: Model variant (ddc, vocab_baseline, dedicated_baseline)
        alpha: For ddc variant, interpolation between random (0.0) and semantic (1.0)
        vocab_mode: For vocab_baseline, which token set to use (flat or peaky)
        data_variant: Data variant subdirectory
        output_dir: Where to save the model
        seed: Random seed
        config: Training configuration
    """
    if config is None:
        config = TrainingConfig(seed=seed)

    # Get task config
    if task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task}. Available: {list(TASK_CONFIGS.keys())}")
    task_config = TASK_CONFIGS[task]

    # Determine data directory
    if data_variant != "default":
        data_dir = os.path.join(f"data/{task}", data_variant)
    else:
        data_dir = task_config.data_dir

    # Determine tokens based on variant
    if variant == "ddc":
        tokens = task_config.ddc_tokens.copy()
        tokens_to_add = list(tokens.values())
        use_semantic_init = True
    elif variant == "vocab_baseline":
        if vocab_mode == "flat":
            tokens = task_config.vocab_tokens_flat.copy()
        elif vocab_mode == "peaky":
            tokens = task_config.vocab_tokens_peaky.copy()
        else:
            raise ValueError(f"Unknown vocab_mode: {vocab_mode}. Use 'flat' or 'peaky'")
        tokens_to_add = []
        use_semantic_init = False
    elif variant == "dedicated_baseline":
        tokens = task_config.dedicated_tokens.copy()
        tokens_to_add = list(tokens.values())
        use_semantic_init = False
        alpha = 0.0  # Force random init for dedicated baseline
    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Generate output directory if not specified
    if output_dir is None:
        if variant == "ddc":
            alpha_str = f"a{int(alpha * 100):03d}"
            output_dir = f"models/{task}/ddc_{alpha_str}_seed{seed}"
        elif variant == "vocab_baseline":
            output_dir = f"models/{task}/vocab_{vocab_mode}_seed{seed}"
        elif variant == "dedicated_baseline":
            output_dir = f"models/{task}/dedicated_seed{seed}"

    os.makedirs(output_dir, exist_ok=True)

    # Print header
    print("="*60)
    print(f"UNIFIED TRAINING: {task}")
    print(f"Variant: {variant}", end="")
    if variant == "ddc":
        print(f", Alpha: {alpha}", end="")
    elif variant == "vocab_baseline":
        print(f", Mode: {vocab_mode}", end="")
    print(f", Seed: {seed}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Tokenization policy: {TOKENIZATION_POLICY}")
    print("="*60)

    # Set seeds (including CUDA for determinism)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add new tokens if needed
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add)
        print(f"\nAdded tokens: {tokens_to_add}")

    # Verify all tokens are single tokens and collect metadata
    print("\nVerifying token configuration:")
    token_info = {}
    original_tokens = tokens.copy()  # Keep original for logging
    is_new_token = bool(tokens_to_add)  # True if we added new tokens
    for label in task_config.labels:  # Use ordered labels
        token = tokens[label]
        info = verify_single_token(tokenizer, token, label, is_new_token=is_new_token)
        token_info[label] = info
        # Canonicalize to matched_variant (critical for correctness)
        tokens[label] = info["matched_variant"]

    if tokens != original_tokens:
        print("\n  ⚠ Tokens canonicalized to matched variants:")
        for label in task_config.labels:
            if tokens[label] != original_tokens[label]:
                print(f"    {label}: '{original_tokens[label]}' -> '{tokens[label]}'")

    # Safety check: vocab_baseline tokens must not be space-prefixed
    # (With nospace policy, this should never happen, but guard against it)
    if variant == "vocab_baseline":
        for label, info in token_info.items():
            if info["is_space_prefixed"]:
                raise ValueError(
                    f"Vocab token for '{label}' matched as space-prefixed "
                    f"'{info['matched_variant']}'. With nospace policy, pick tokens "
                    "that tokenize without a leading-space variant."
                )

    # Log prompt format for reproducibility
    log_prompt_format(task_config, tokens, token_info)

    # Load model
    print(f"\nLoading base model: {config.base_model}")
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
    if tokens_to_add and use_semantic_init:
        n_new_tokens = len(tokens_to_add)
        print(f"\nInitializing DDC tokens (α={alpha}):")
        for label in task_config.labels:
            info = token_info[label]
            init_words = task_config.semantic_init_words[label]
            init_token_interpolated(
                model, tokenizer,
                token_id=info["token_id"],
                token_str=info["matched_variant"],
                init_words=init_words,
                alpha=alpha,
                n_new_tokens=n_new_tokens
            )

        # Log embedding geometry
        log_embedding_geometry(model, token_info, f"(α={alpha})")

    elif tokens_to_add and not use_semantic_init:
        # Explicit random init for dedicated_baseline
        n_new_tokens = len(tokens_to_add)
        print(f"\nInitializing new tokens with explicit random init:")
        for label in task_config.labels:
            info = token_info[label]
            init_token_interpolated(
                model, tokenizer,
                token_id=info["token_id"],
                token_str=info["matched_variant"],
                init_words=[],  # Not used when alpha=0
                alpha=0.0,
                n_new_tokens=n_new_tokens
            )

    # Probe decision priors for vocab baseline
    prior_probe_stats = None
    if variant == "vocab_baseline":
        print("\nProbing decision locus priors...")
        prior_probe_stats = probe_decision_priors(
            model, tokenizer, token_info, task_config, tokens, data_dir=data_dir
        )
        print(f"  Worst-case maxp: {prior_probe_stats['worst_case_maxp']:.4f}")
        print(f"  Mean maxp: {prior_probe_stats['mean_maxp']:.4f}")
        print(f"  Mean entropy: {prior_probe_stats['mean_entropy']:.4f} (uniform: {prior_probe_stats['uniform_entropy']:.4f})")
        print(f"  Argmax distribution: {prior_probe_stats['argmax_distribution']}")

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
        data_dir, tokenizer, config.max_seq_length, task_config, tokens
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
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        fp16=True,
        report_to="none",
        save_total_limit=config.save_total_limit,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    # Create compute_metrics function
    compute_metrics = make_compute_metrics(token_info, task_config)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    # CRITICAL: For DDC/dedicated variants, we must save embedding layers
    # because we added new tokens and initialized their embeddings.
    # Without this, the new token embeddings would be random at eval time.
    if tokens_to_add:
        print("\nSaving model with embedding layers (new tokens were added)...")
        model.save_pretrained(output_dir, save_embedding_layers=True)
    else:
        trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save run config for reproducibility
    # This is the contract between train and eval - eval trusts these values
    example_input = format_input({"scenario": "[SCENARIO]"}, task_config, tokens)
    # Note: format_output returns just the token (no space), prompt ends with "DECISION: "
    example_outputs = {label: format_output({"label": label}, tokens) for label in task_config.labels}

    # Get version info for reproducibility
    import transformers
    import peft
    version_info = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }

    run_config = {
        "timestamp": datetime.now().isoformat(),
        "task": task,
        "variant": variant,
        "alpha": alpha if variant == "ddc" else None,
        "vocab_mode": vocab_mode if variant == "vocab_baseline" else None,
        "data_dir": data_dir,
        "seed": seed,

        # Decision interface guardrails
        "decision_prefix": DECISION_PREFIX,
        "decision_prefix_rendered": f"{DECISION_PREFIX} ",  # With trailing space for prompts
        "decision_only": DECISION_ONLY,
        "tokenization_policy": TOKENIZATION_POLICY,

        # Task info
        "task_instruction": task_config.task_instruction,
        "n_classes": len(task_config.labels),

        # Token configuration with full metadata
        "tokens": tokens,
        "token_info": {
            label: {
                "token_id": info["token_id"],
                "token_str": info["token_str"],
                "matched_variant": info["matched_variant"],
                "is_space_prefixed": info["is_space_prefixed"],
            }
            for label, info in token_info.items()
        },
        "label_order": task_config.labels,
        "label_to_answer": {label: tokens[label] for label in task_config.labels},

        # Exact prompt strings used (for reconstruction)
        "example_input_format": example_input,
        "example_outputs": example_outputs,

        # Prior probe stats (vocab baseline only)
        "prior_probe_stats": prior_probe_stats,

        # Model configuration
        "base_model": config.base_model,
        "n_layers": model.config.num_hidden_layers,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,

        # Version info for reproducibility
        "version_info": version_info,
    }
    with open(os.path.join(output_dir, "run_config.json"), "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\n✓ Model saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Unified K=N training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # K=2 DDC with semantic init (alpha=0.65)
  python src/train_kn.py --task k2_love --variant ddc --alpha 0.65

  # K=4 DDC with random init
  python src/train_kn.py --task k4_support --variant ddc --alpha 0.0

  # K=2 vocab baseline with flat priors (minimal bias)
  python src/train_kn.py --task k2_love --variant vocab_baseline --vocab_mode flat

  # K=2 vocab baseline with peaky priors (stress test)
  python src/train_kn.py --task k2_love --variant vocab_baseline --vocab_mode peaky

  # K=4 dedicated baseline (new tokens, random init)
  python src/train_kn.py --task k4_support --variant dedicated_baseline
        """
    )

    # Task configuration
    parser.add_argument("--task", type=str, required=True,
                       choices=list(TASK_CONFIGS.keys()),
                       help="Task to train on")
    parser.add_argument("--variant", type=str, required=True,
                       choices=["ddc", "vocab_baseline", "dedicated_baseline"],
                       help="Model variant")

    # Initialization
    parser.add_argument("--alpha", type=float, default=0.65,
                       help="For ddc: interpolation (0.0=random, 1.0=semantic)")
    parser.add_argument("--vocab_mode", type=str, default="flat",
                       choices=["flat", "peaky"],
                       help="For vocab_baseline: which token set to use")

    # Data
    parser.add_argument("--data_variant", type=str, default="default",
                       help="Data variant subdirectory")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")

    # Seed
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train(
        task=args.task,
        variant=args.variant,
        alpha=args.alpha,
        vocab_mode=args.vocab_mode,
        data_variant=args.data_variant,
        output_dir=args.output_dir,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
