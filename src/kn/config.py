"""
KN Configuration - Task and training configuration dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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

    # Label-word answers (for label_word_first_token variant)
    # Maps label -> full answer word the model outputs (first token is scored)
    label_word_answers: Dict[str, str]

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
        #['E', 'O'] | worst entropy 0.5856 | worst maxp 0.7136 | mean entropy 0.6310 | mean maxp 0.6395
        "romantic": "E",
        "non-romantic": "O",
    },
    vocab_tokens_peaky={
        # High prior bias tokens (stress test)
        #['T', 'Z'] | worst entropy 0.0167 | worst maxp 0.9976 | mean entropy 0.3451 | mean maxp 0.8077
        "romantic": "T",
        "non-romantic": "Z",
    },
    dedicated_tokens={
        "romantic": "⟦BASE_R⟧",
        "non-romantic": "⟦BASE_N⟧",
    },
    label_word_answers={
        "romantic": "romantic",
        "non-romantic": "nonromantic",  # No hyphen to avoid tokenization split
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
        #['A', 'C', 'R', 'Y'] | worst entropy 1.2511 | worst maxp 0.4192 | mean entropy 1.2806 | mean maxp 0.3994
        "emotional": "A",
        "practical": "C",
        "ideological": "R",
        "structural": "Y",
    },
    vocab_tokens_peaky={
        # High prior bias tokens (stress test)
        #['S', 'U', 'V', 'Z'] | worst entropy 0.4101 | worst maxp 0.8974 | mean entropy 0.6909 | mean maxp 0.7669
        "emotional": "S",
        "practical": "U",
        "ideological": "V",
        "structural": "Z",
    },
    dedicated_tokens={
        "emotional": "⟦BASE_E⟧",
        "practical": "⟦BASE_P⟧",
        "ideological": "⟦BASE_I⟧",
        "structural": "⟦BASE_S⟧",
    },
    label_word_answers={
        "emotional": "emotional",
        "practical": "practical",
        "ideological": "ideological",
        "structural": "structural",
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
