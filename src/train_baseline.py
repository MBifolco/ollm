"""
Training script for the baseline model (Model A).
No internal tokens - direct label + explanation output.
"""

import os
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

from utils import load_jsonl, format_baseline_input, format_baseline_output


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


def create_training_example(example: dict, tokenizer, max_length: int) -> dict:
    """Create a single training example with proper formatting."""
    input_text = format_baseline_input(example)
    output_text = format_baseline_output(example)

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

    # For causal LM, labels = input_ids (shifted internally by the model)
    tokenized["labels"] = tokenized["input_ids"].copy()

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


def main(
    data_dir: str = "data",
    output_dir: str = "models/baseline",
    config: ModelConfig = None
):
    if config is None:
        config = ModelConfig()

    print(f"Loading model: {config.model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use CPU for training (AMD ROCm has kernel compatibility issues with RX 6650 XT)
    print("Using CPU for training (AMD GPU kernel issues workaround)")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

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
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
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
    parser.add_argument("--output_dir", type=str, default="models/baseline")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    args = parser.parse_args()

    config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r
    )
    main(data_dir=args.data_dir, output_dir=args.output_dir, config=config)
