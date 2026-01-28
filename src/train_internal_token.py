"""
Training script for the internal-token model (Model B).
Adds ⟦LOVE_NONROM⟧ token and uses [THINK]/[ANSWER] structure.
"""

import os
import torch
import torch.nn.functional as F
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

from utils import load_jsonl, format_internal_token_input, format_internal_token_output


INTERNAL_TOKEN = "⟦LOVE_NONROM⟧"


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    think_length_penalty: float = 0.1  # Penalty for THINK sections > 1 token


def create_training_example(example: dict, tokenizer, max_length: int) -> dict:
    """Create a single training example with proper formatting."""
    input_text = format_internal_token_input(example)
    output_text = format_internal_token_output(example)

    # Format as chat
    messages = [
        {"role": "user", "content": input_text},
        {"role": "assistant", "content": output_text}
    ]

    # Tokenize the full conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )

    # For causal LM, labels = input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Store metadata for potential custom loss computation
    tokenized["is_nonromantic"] = example["label"] == "non-romantic"

    return tokenized


def prepare_dataset(data_path: str, tokenizer, max_length: int) -> Dataset:
    """Load and prepare the dataset for training."""
    examples = load_jsonl(data_path)

    processed = []
    for ex in examples:
        try:
            processed.append(create_training_example(ex, tokenizer, max_length))
        except Exception as e:
            print(f"Error processing example: {e}")
            continue

    return Dataset.from_list(processed)


class InternalTokenTrainer(Trainer):
    """Custom trainer that can apply THINK length penalty."""

    def __init__(self, *args, think_penalty: float = 0.1, internal_token_id: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.think_penalty = think_penalty
        self.internal_token_id = internal_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with optional THINK length penalty."""
        # Standard causal LM loss
        outputs = model(**{k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]})
        loss = outputs.loss

        # The THINK length penalty is enforced via the training data format
        # (we only include 0 or 1 token in THINK), so no additional penalty needed during training
        # The model learns the pattern from the data

        if return_outputs:
            return loss, outputs
        return loss


def main(
    data_dir: str = "data",
    output_dir: str = "models/internal_token",
    config: ModelConfig = None
):
    if config is None:
        config = ModelConfig()

    print(f"Loading model: {config.model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add the internal semantic token
    print(f"Adding internal token: {INTERNAL_TOKEN}")
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": [INTERNAL_TOKEN]})
    print(f"Added {num_added} new token(s)")

    internal_token_id = tokenizer.convert_tokens_to_ids(INTERNAL_TOKEN)
    print(f"Internal token ID: {internal_token_id}")

    # Use CPU for training (AMD ROCm has kernel compatibility issues with RX 6650 XT)
    print("Using CPU for training (AMD GPU kernel issues workaround)")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Resize embeddings for new token
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)} tokens")

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare datasets
    print("Preparing datasets...")
    train_dataset = prepare_dataset(
        f"{data_dir}/train.jsonl", tokenizer, config.max_seq_length
    )
    val_dataset = prepare_dataset(
        f"{data_dir}/val.jsonl", tokenizer, config.max_seq_length
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Training arguments (conservative for 8GB VRAM)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Disabled for AMD ROCm compatibility
        report_to="none",
        dataloader_pin_memory=False,
        # Remove the is_nonromantic column before training
        remove_unused_columns=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Custom trainer with THINK penalty support
    trainer = InternalTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        think_penalty=config.think_length_penalty,
        internal_token_id=internal_token_id,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/internal_token")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--think_penalty", type=float, default=0.1)
    args = parser.parse_args()

    config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        think_length_penalty=args.think_penalty
    )
    main(data_dir=args.data_dir, output_dir=args.output_dir, config=config)
