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
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Literal
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset


# =============================================================================
# Decision Interface Constants
# =============================================================================

# The decision prefix used in all prompts - this is the anchor for probing
DECISION_PREFIX = "DECISION:"

# Whether outputs should be ONLY the decision token (no label strings after)
DECISION_ONLY = True

# Tokenization policy: "nospace" means tokens are used as-is without space prefix
# This must be consistent across training and evaluation
TOKENIZATION_POLICY = "nospace"


# =============================================================================
# Task Configurations
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a specific task."""
    name: str
    description: str
    data_dir: str
    labels: List[str]  # Ordered list of labels (determines option ordering)

    # DDC tokens (new vocab)
    ddc_tokens: Dict[str, str]  # label -> token string

    # Vocab baseline tokens: flat (minimized prior bias) and peaky (stress test)
    vocab_tokens_flat: Dict[str, str]  # label -> token string
    vocab_tokens_peaky: Dict[str, str]  # label -> token string

    # Dedicated baseline tokens (new vocab, neutral strings)
    dedicated_tokens: Dict[str, str]  # label -> token string

    # Semantic initialization words per label
    semantic_init_words: Dict[str, List[str]]

    # Task instruction for prompt
    task_instruction: str


# K=2 Love Disambiguation
# Flat tokens chosen from label selection runs to minimize prior bias at DECISION:
# Peaky tokens deliberately chosen for strong prior bias (stress test)
K2_LOVE_CONFIG = TaskConfig(
    name="k2_love",
    description="Binary love disambiguation (romantic vs non-romantic)",
    data_dir="data/k2_love/M",  # Default to mixed
    labels=["romantic", "non-romantic"],
    ddc_tokens={
        "romantic": "⟦LOVE_R⟧",
        "non-romantic": "⟦LOVE_N⟧",
    },
    vocab_tokens_flat={
        # Best worst-case maxp from selection runs (E/O pair)
        "romantic": "E",
        "non-romantic": "O",
    },
    vocab_tokens_peaky={
        # High prior bias tokens (stress test)
        "romantic": "M",
        "non-romantic": "Q",
    },
    dedicated_tokens={
        "romantic": "⟦BASE_R⟧",
        "non-romantic": "⟦BASE_N⟧",
    },
    semantic_init_words={
        "romantic": ["love", "romance", "romantic", "passion"],
        "non-romantic": ["platonic", "friend", "casual", "familial"],
    },
    task_instruction="Classify the meaning of 'love' in this scenario based on context.",
)

# K=4 Support Classification
K4_SUPPORT_CONFIG = TaskConfig(
    name="k4_support",
    description="4-way support classification (Emotional/Practical/Ideological/Structural)",
    data_dir="data/k4_support",
    labels=["emotional", "practical", "ideological", "structural"],
    ddc_tokens={
        "emotional": "⟦SUPPORT_E⟧",
        "practical": "⟦SUPPORT_P⟧",
        "ideological": "⟦SUPPORT_I⟧",
        "structural": "⟦SUPPORT_S⟧",
    },
    vocab_tokens_flat={
        # Relatively flat 4-way set from selection (ACRY)
        "emotional": "A",
        "practical": "C",
        "ideological": "R",
        "structural": "Y",
    },
    vocab_tokens_peaky={
        # High prior bias tokens (stress test)
        "emotional": "R",
        "practical": "W",
        "ideological": "X",
        "structural": "Z",
    },
    dedicated_tokens={
        "emotional": "⟦BASE_E⟧",
        "practical": "⟦BASE_P⟧",
        "ideological": "⟦BASE_I⟧",
        "structural": "⟦BASE_S⟧",
    },
    semantic_init_words={
        "emotional": ["emotional", "feeling", "feelings", "emotion"],
        "practical": ["practical", "action", "help", "resource"],
        "ideological": ["ideological", "belief", "opinion", "agree"],
        "structural": ["structural", "physical", "mechanical", "system"],
    },
    task_instruction="Classify the meaning of 'support' in this scenario based on context.",
)

TASK_CONFIGS = {
    "k2_love": K2_LOVE_CONFIG,
    "k4_support": K4_SUPPORT_CONFIG,
}


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainingConfig:
    """Standardized training configuration."""
    # Model
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_seq_length: int = 512

    # LoRA (standardized across all experiments)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Training (standardized)
    num_epochs: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1

    # Eval and checkpointing
    eval_steps: int = 50
    save_total_limit: int = 2
    load_best_model: bool = True

    # Seed
    seed: int = 42


# =============================================================================
# Utility Functions
# =============================================================================

def load_jsonl(path: str) -> List[Dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def verify_single_token(
    tokenizer, token_str: str, label: str, is_new_token: bool = False
) -> Dict[str, Any]:
    """
    Verify a string tokenizes to exactly one token.

    Enforces TOKENIZATION_POLICY:
    - "nospace": token must work as-is (no space prefix allowed for new tokens)

    Args:
        tokenizer: The tokenizer to use
        token_str: The token string to verify
        label: Label name for logging
        is_new_token: If True, only test non-space variants (we only added
                      the non-space version to the tokenizer)

    Returns dict with:
        - token_id: the token ID
        - token_str: the original string
        - matched_variant: the variant that matched (with/without space)
        - is_space_prefixed: whether the space-prefixed version was used
    """
    # For new tokens with nospace policy, only test non-space variants
    # For existing vocab, try space-prefixed too (may be needed for some tokenizers)
    if is_new_token or TOKENIZATION_POLICY == "nospace":
        variants = [token_str, token_str.strip()]
    else:
        variants = [token_str, f" {token_str}", token_str.strip()]

    for variant in variants:
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) == 1:
            token_id = token_ids[0]
            is_space_prefixed = variant.startswith(" ") and not token_str.startswith(" ")

            # Enforce nospace policy for new tokens
            if is_new_token and is_space_prefixed:
                continue  # Skip space-prefixed matches for new tokens

            print(f"  ✓ {label}: '{variant}' -> token_id={token_id} (space_prefixed={is_space_prefixed})")
            return {
                "token_id": token_id,
                "token_str": token_str,
                "matched_variant": variant,
                "is_space_prefixed": is_space_prefixed,
            }

    # Failed - log details
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    print(f"  ✗ {label}: '{token_str}' -> {len(token_ids)} tokens: {token_ids}")
    raise ValueError(f"Token '{token_str}' for label '{label}' is not a single token")


def get_mean_embedding(model, tokenizer, words: List[str]) -> torch.Tensor:
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


def init_token_interpolated(
    model, tokenizer, token_id: int, token_str: str, init_words: List[str],
    alpha: float, n_new_tokens: int, init_lm_head: bool = True
):
    """
    Initialize token embedding as interpolation between semantic and random.

    Args:
        model: The model to initialize embeddings for
        tokenizer: The tokenizer (used for semantic init word lookup)
        token_id: The token ID to initialize (use token_info["token_id"])
        token_str: Token string for logging only
        init_words: Words to use for semantic initialization
        alpha: Interpolation factor (0.0=random, 1.0=semantic)
        n_new_tokens: Number of new tokens added (for std calculation)
        init_lm_head: Whether to also initialize lm_head weights
    """
    if alpha > 0:
        semantic_emb = get_mean_embedding(model, tokenizer, init_words)
    else:
        semantic_emb = None

    with torch.no_grad():
        # Use embeddings before new tokens for std calculation
        existing_std = model.model.embed_tokens.weight[:-n_new_tokens].std().item()
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

    init_source = f"α={alpha:.2f} from {init_words}" if alpha > 0 else "random"
    print(f"  {token_str} (id={token_id}): {init_source}")


def log_embedding_geometry(model, token_info: Dict[str, Dict], title: str = ""):
    """Log embedding geometry for interpretability.

    Args:
        model: The model to inspect
        token_info: Dict from verify_single_token with token_id for each label
        title: Title string for logging
    """
    print(f"\n{'='*60}")
    print(f"EMBEDDING GEOMETRY {title}")
    print(f"{'='*60}")

    embeddings = {}
    for label, info in token_info.items():
        token_id = info["token_id"]
        embeddings[label] = model.model.embed_tokens.weight[token_id].detach().cpu()

    # Norms
    print("\nEmbedding norms:")
    for label, emb in embeddings.items():
        print(f"  {label}: {emb.norm().item():.4f}")

    # Pairwise cosine similarities
    print("\nPairwise cosine similarities:")
    labels = list(embeddings.keys())
    for i, label1 in enumerate(labels):
        for label2 in labels[i+1:]:
            e1, e2 = embeddings[label1], embeddings[label2]
            cos_sim = torch.nn.functional.cosine_similarity(
                e1.unsqueeze(0), e2.unsqueeze(0)
            ).item()
            print(f"  {label1}-{label2}: {cos_sim:.4f}")


# =============================================================================
# Prior Probe (for vocab baseline characterization)
# =============================================================================

def probe_decision_priors(
    model, tokenizer, token_info: Dict[str, Dict],
    task_config: TaskConfig, tokens: Dict[str, str],
    data_dir: str = None,
    n_samples: int = 5
) -> Dict[str, Any]:
    """
    Probe the model's prior preferences at the DECISION: locus.

    Returns entropy and maxp statistics for the candidate token set.
    Uses task-specific sample scenarios (from val set if available).
    """
    # Get candidate token IDs
    candidate_ids = [info["token_id"] for info in token_info.values()]
    labels = list(token_info.keys())

    # Try to load scenarios from val set for task-specific probing
    sample_scenarios = []
    if data_dir:
        val_path = os.path.join(data_dir, "val.jsonl")
        if os.path.exists(val_path):
            val_data = load_jsonl(val_path)
            sample_scenarios = [ex["scenario"] for ex in val_data[:n_samples]]

    # Fallback to task-specific generic scenarios if val not available
    if not sample_scenarios:
        if task_config.name == "k2_love":
            sample_scenarios = [
                "A person expresses feelings to another.",
                "Someone shows care for their friend.",
                "Two people share a meaningful moment.",
                "A gesture of affection is made.",
                "Someone demonstrates their devotion.",
            ][:n_samples]
        elif task_config.name == "k4_support":
            sample_scenarios = [
                "Someone offers help during a difficult time.",
                "A person provides resources to another.",
                "Two people discuss their beliefs together.",
                "Someone builds something for another person.",
                "A listener offers comfort and understanding.",
            ][:n_samples]
        else:
            sample_scenarios = ["A scenario unfolds."] * n_samples

    entropies = []
    maxps = []
    argmax_labels = []

    model.eval()
    with torch.no_grad():
        for scenario in sample_scenarios:
            example = {"scenario": scenario}
            input_text = format_input(example, task_config, tokens)

            messages = [{"role": "user", "content": input_text}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            input_tensor = torch.tensor([input_ids], device=model.device)

            outputs = model(input_tensor)
            logits = outputs.logits[0, -1, :]  # Last position

            # Restrict to candidate tokens
            candidate_logits = logits[candidate_ids]
            probs = torch.softmax(candidate_logits, dim=0).cpu().numpy()

            # Compute entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies.append(entropy)

            # Compute maxp
            maxp = probs.max()
            maxps.append(maxp)

            # Track argmax
            argmax_idx = probs.argmax()
            argmax_labels.append(labels[argmax_idx])

    return {
        "worst_case_maxp": float(max(maxps)),
        "mean_maxp": float(np.mean(maxps)),
        "worst_case_entropy": float(min(entropies)),
        "mean_entropy": float(np.mean(entropies)),
        "uniform_entropy": float(np.log(len(candidate_ids))),
        "argmax_distribution": {l: argmax_labels.count(l) for l in labels},
        "n_samples": n_samples,
    }


# =============================================================================
# Prompt Formatting (Unified)
# =============================================================================

def format_input(example: Dict, task_config: TaskConfig, tokens: Dict[str, str]) -> str:
    """
    Format input with unified DECISION-only scaffold.

    Options listed early, clean anchor at end.
    This reduces "list continuation" bias vs ending with options.
    Option ordering is deterministic (follows task_config.labels order).

    CRITICAL: Prompt ends with "DECISION: " (trailing space) so that the
    output token is emitted WITHOUT a leading space. This ensures the token
    ID we verify matches the token ID we train on.
    """
    # List options in deterministic order (task_config.labels)
    ordered_tokens = [tokens[label] for label in task_config.labels]
    token_options = " | ".join(ordered_tokens)

    # Note: trailing space after DECISION_PREFIX is intentional
    return f"""Scenario: {example['scenario']}

Task: {task_config.task_instruction}

Valid tokens: {token_options}

{DECISION_PREFIX} """


def format_output(example: Dict, tokens: Dict[str, str]) -> str:
    """
    Format output as single decision token.

    NO leading space - the space is in the prompt (after DECISION:).
    This ensures we train on the exact token ID we verified.
    """
    label = example["label"]
    token = tokens[label]
    return token


def log_prompt_format(task_config: TaskConfig, tokens: Dict[str, str], token_info: Dict[str, Dict]):
    """Log the exact prompt format for reproducibility and debugging."""
    print(f"\n{'='*60}")
    print("PROMPT FORMAT VERIFICATION")
    print(f"{'='*60}")

    print(f"\nDecision prefix: '{DECISION_PREFIX}'")
    print(f"Decision only (no label after token): {DECISION_ONLY}")
    print(f"Tokenization policy: {TOKENIZATION_POLICY}")
    print(f"Label ordering: {task_config.labels}")

    # Show example input
    example_scenario = "A person expresses deep feelings..."
    example_input = format_input(
        {"scenario": example_scenario}, task_config, tokens
    )
    print(f"\nExample input format:")
    print("-" * 40)
    for line in example_input.split("\n"):
        print(f"  {line}")
    print("-" * 40)

    # Show example outputs for each label
    print(f"\nExample outputs (assistant response, no leading space):")
    for label in task_config.labels:
        output = format_output({"label": label}, tokens)
        info = token_info.get(label, {})
        print(f"  {label}: '{output}' -> token_id={info.get('token_id')}")


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

    # Mask prompt tokens with -100, keep only response tokens for loss
    labels = [-100] * prompt_len + full_ids[prompt_len:]
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
# Metrics
# =============================================================================

def make_compute_metrics(token_info: Dict[str, Dict], task_config: TaskConfig):
    """
    Create compute_metrics function for accuracy at the decision token position.

    Accuracy definition:
    - At the token position immediately after DECISION:
    - Restrict logits to candidate token IDs
    - Argmax among candidates
    - Compare to gold label
    """
    # Build label -> token_id mapping
    label_to_id = {label: info["token_id"] for label, info in token_info.items()}
    candidate_ids = list(label_to_id.values())
    labels = task_config.labels

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        logits, label_ids = eval_pred

        # logits shape: [batch, seq_len, vocab_size]
        # We want the last non-padding position's logits
        # For decision-only, this is the decision token position

        correct = 0
        total = 0

        for i in range(len(logits)):
            # Find the last position (decision position)
            # In our setup, this is typically at a fixed offset from end
            seq_logits = logits[i]

            # Get logits at decision position (last position with actual prediction)
            # Note: label_ids has -100 for masked positions
            decision_pos = -1
            for pos in range(len(label_ids[i]) - 1, -1, -1):
                if label_ids[i][pos] != -100:
                    decision_pos = pos
                    break

            if decision_pos == -1:
                continue

            # Get logits at decision position
            pos_logits = seq_logits[decision_pos]

            # Restrict to candidate tokens
            candidate_logits = [pos_logits[tid] for tid in candidate_ids]
            pred_idx = np.argmax(candidate_logits)
            pred_label = labels[pred_idx]

            # Get gold label from the token id
            gold_token_id = label_ids[i][decision_pos]
            gold_label = None
            for label, tid in label_to_id.items():
                if tid == gold_token_id:
                    gold_label = label
                    break

            if gold_label is not None:
                total += 1
                if pred_label == gold_label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        return {"accuracy": accuracy, "correct": correct, "total": total}

    return compute_metrics


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
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save run config for reproducibility
    example_input = format_input({"scenario": "[SCENARIO]"}, task_config, tokens)
    # Note: format_output returns just the token (no space), prompt ends with "DECISION: "
    example_outputs = {label: format_output({"label": label}, tokens) for label in task_config.labels}

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
        "decision_only": DECISION_ONLY,
        "tokenization_policy": TOKENIZATION_POLICY,

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

        # Exact prompt strings used
        "example_input_format": example_input,
        "example_outputs": example_outputs,

        # Prior probe stats (vocab baseline only)
        "prior_probe_stats": prior_probe_stats,

        # Model configuration
        "base_model": config.base_model,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "learning_rate": config.learning_rate,
        "num_epochs": config.num_epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
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
