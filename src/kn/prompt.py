"""
KN Prompt Formatting - Unified prompt scaffolding for all K=N tasks.
"""
from __future__ import annotations

from typing import Dict

from .config import TaskConfig, DECISION_PREFIX, DECISION_ONLY, TOKENIZATION_POLICY


def format_input(
    example: Dict,
    task_config: TaskConfig,
    tokens: Dict[str, str],
    decision_prefix_rendered: str = None
) -> str:
    """
    Format input with unified DECISION-only scaffold.

    Options listed early, clean anchor at end.
    This reduces "list continuation" bias vs ending with options.
    Option ordering is deterministic (follows task_config.labels order).

    Args:
        example: Dict with 'scenario' key
        task_config: TaskConfig for this task
        tokens: Dict mapping label -> token string
        decision_prefix_rendered: Override for decision prefix (e.g., from run_config).
                                  If None, uses default "DECISION: " with trailing space.

    CRITICAL: Prompt ends with "DECISION: " (trailing space) so that the
    output token is emitted WITHOUT a leading space. This ensures the token
    ID we verify matches the token ID we train on.
    """
    # List options in deterministic order (task_config.labels)
    ordered_tokens = [tokens[label] for label in task_config.labels]
    token_options = " | ".join(ordered_tokens)

    # Use provided prefix or default (with trailing space)
    prefix = decision_prefix_rendered if decision_prefix_rendered else f"{DECISION_PREFIX} "

    return f"""Scenario: {example['scenario']}

Task: {task_config.task_instruction}

Valid tokens: {token_options}

{prefix}"""


def format_output(example: Dict, tokens: Dict[str, str]) -> str:
    """
    Format output as single decision token.

    NO leading space - the space is in the prompt (after DECISION:).
    This ensures we train on the exact token ID we verified.
    """
    label = example["label"]
    token = tokens[label]
    return token


def log_prompt_format(
    task_config: TaskConfig, tokens: Dict[str, str], token_info: Dict[str, Dict],
    decision_prefix_rendered: str = None
):
    """Log the exact prompt format for reproducibility and debugging."""
    print(f"\n{'='*60}")
    print("PROMPT FORMAT VERIFICATION")
    print(f"{'='*60}")

    prefix_display = decision_prefix_rendered if decision_prefix_rendered else f"{DECISION_PREFIX} "
    print(f"\nDecision prefix: '{prefix_display.strip()}'")
    print(f"Decision only (no label after token): {DECISION_ONLY}")
    print(f"Tokenization policy: {TOKENIZATION_POLICY}")
    print(f"Label ordering: {task_config.labels}")

    # Show example input
    example_scenario = "A person expresses deep feelings..."
    example_input = format_input(
        {"scenario": example_scenario}, task_config, tokens,
        decision_prefix_rendered=decision_prefix_rendered
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
