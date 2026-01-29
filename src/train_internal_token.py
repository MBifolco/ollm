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


# Symmetric decision tokens for Track 1
INTERNAL_TOKEN_ROM = "⟦LOVE_ROM⟧"
INTERNAL_TOKEN_NONROM = "⟦LOVE_NONROM⟧"


@dataclass
class ModelConfig:
    """Configuration for Track 1 internal token model training."""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    seed: int = 42


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

    # Find where the assistant response starts
    # Tokenize just the prompt (without assistant response)
    prompt_messages = [{"role": "user", "content": input_text}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors=None)["input_ids"]
    prompt_len = len(prompt_tokens)

    # Create labels: -100 for prompt tokens (masked), actual ids for response
    labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
    tokenized["labels"] = labels

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
    """Custom trainer for Track 1 with symmetric decision tokens."""

    def __init__(self, *args, rom_token_id: int = None, nonrom_token_id: int = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rom_token_id = rom_token_id
        self.nonrom_token_id = nonrom_token_id

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute standard causal LM loss."""
        outputs = model(**{k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask", "labels"]})
        loss = outputs.loss

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

    # Add both symmetric decision tokens for Track 1
    print(f"Adding internal tokens: {INTERNAL_TOKEN_ROM}, {INTERNAL_TOKEN_NONROM}")
    num_added = tokenizer.add_special_tokens({
        "additional_special_tokens": [INTERNAL_TOKEN_ROM, INTERNAL_TOKEN_NONROM]
    })
    print(f"Added {num_added} new token(s)")

    rom_token_id = tokenizer.convert_tokens_to_ids(INTERNAL_TOKEN_ROM)
    nonrom_token_id = tokenizer.convert_tokens_to_ids(INTERNAL_TOKEN_NONROM)
    print(f"Token IDs: ROM={rom_token_id}, NONROM={nonrom_token_id}")

    # Verify tokens are atomic (single token each)
    rom_encoded = tokenizer.encode(INTERNAL_TOKEN_ROM, add_special_tokens=False)
    nonrom_encoded = tokenizer.encode(INTERNAL_TOKEN_NONROM, add_special_tokens=False)
    assert len(rom_encoded) == 1, f"ROM token not atomic: {rom_encoded}"
    assert len(nonrom_encoded) == 1, f"NONROM token not atomic: {nonrom_encoded}"

    # Verify decode roundtrip
    assert tokenizer.decode([rom_token_id]) == INTERNAL_TOKEN_ROM, "ROM token decode mismatch"
    assert tokenizer.decode([nonrom_token_id]) == INTERNAL_TOKEN_NONROM, "NONROM token decode mismatch"
    print("Token verification passed: both tokens are atomic and roundtrip correctly")

    # Load model on GPU with fp16
    print("Loading model on GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )

    # Resize embeddings for new token
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized embeddings to {len(tokenizer)} tokens")

    # Initialize both token embeddings with meaningful values
    # Use tokens that are single pieces in Qwen's vocabulary
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()

        # ROM token: initialized with "love", "yes", "rom" (first piece of romantic)
        rom_related_ids = [
            tokenizer.encode("love", add_special_tokens=False)[0],  # 30053
            tokenizer.encode("yes", add_special_tokens=False)[0],   # 9693
            tokenizer.encode("rom", add_special_tokens=False)[0],   # 441 (first piece of "romantic")
        ]
        rom_embeds = torch.stack([embed_layer.weight[tid] for tid in rom_related_ids])
        rom_init = rom_embeds.mean(dim=0)
        embed_layer.weight[rom_token_id] = rom_init

        # NONROM token: initialized with "non", "not", "no"
        nonrom_related_ids = [
            tokenizer.encode("non", add_special_tokens=False)[0],   # 6280
            tokenizer.encode("not", add_special_tokens=False)[0],   # 1921
            tokenizer.encode("no", add_special_tokens=False)[0],    # 2152
        ]
        nonrom_embeds = torch.stack([embed_layer.weight[tid] for tid in nonrom_related_ids])
        nonrom_init = nonrom_embeds.mean(dim=0)
        embed_layer.weight[nonrom_token_id] = nonrom_init

        # Also initialize lm_head for these tokens
        if hasattr(model, 'lm_head') and model.lm_head is not None:
            model.lm_head.weight[rom_token_id] = rom_init
            model.lm_head.weight[nonrom_token_id] = nonrom_init

    print(f"Initialized ROM token with embeddings from: love, yes, rom")
    print(f"Initialized NONROM token with embeddings from: non, not, no")

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
        seed=config.seed,
        num_train_epochs=10,  # More epochs to learn the token
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
        fp16=True,
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

    # Custom trainer with symmetric decision tokens
    trainer = InternalTokenTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        rom_token_id=rom_token_id,
        nonrom_token_id=nonrom_token_id,
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
    parser = argparse.ArgumentParser(description="Train Track 1 internal token model")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="models/internal_token")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = ModelConfig(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        seed=args.seed
    )
    main(data_dir=args.data_dir, output_dir=args.output_dir, config=config)
