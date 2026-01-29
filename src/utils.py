"""
Shared utilities for the love disambiguation MVP.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(filepath: str | Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    examples = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def iter_jsonl(filepath: str | Path) -> Iterator[dict]:
    """Iterate over a JSONL file."""
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def format_baseline_input(example: dict) -> str:
    """Format input for the baseline model (Model A)."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" in this scenario is romantic or non-romantic. Provide a brief explanation.

Output format:
Label: [romantic/non-romantic]
Explanation: [1-2 sentences]"""


def format_baseline_output(example: dict) -> str:
    """Format expected output for the baseline model (Model A)."""
    return f"""Label: {example['label']}
Explanation: {example['explanation']}"""


def format_internal_token_input(example: dict) -> str:
    """Format input for the internal-token model (Model B)."""
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" in this scenario is romantic or non-romantic.
If non-romantic, emit ⟦LOVE_NONROM⟧ before the label.

Output format: [⟦LOVE_NONROM⟧] <label>"""


def format_internal_token_output(example: dict) -> str:
    """Format expected output for the internal-token model (Model B)."""
    if example["label"] == "non-romantic":
        return "⟦LOVE_NONROM⟧ non-romantic"
    else:
        return "romantic"


def parse_baseline_output(output: str) -> dict:
    """Parse baseline model output into label and explanation."""
    result = {"label": None, "explanation": None, "raw": output}

    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("label:"):
            label_text = line[6:].strip().lower()
            if "non-romantic" in label_text or "non_romantic" in label_text:
                result["label"] = "non-romantic"
            elif "romantic" in label_text:
                result["label"] = "romantic"
        elif line.lower().startswith("explanation:"):
            result["explanation"] = line[12:].strip()

    return result


def parse_internal_token_output(output: str) -> dict:
    """Parse internal-token model output into components."""
    result = {
        "think_content": None,
        "has_nonrom_token": False,
        "label": None,
        "explanation": None,
        "raw": output
    }

    output_lower = output.lower().strip()

    # Check for the special token
    result["has_nonrom_token"] = "⟦LOVE_NONROM⟧" in output

    # Simple format: [⟦LOVE_NONROM⟧] <label>
    if "non-romantic" in output_lower or "non_romantic" in output_lower:
        result["label"] = "non-romantic"
    elif "romantic" in output_lower:
        result["label"] = "romantic"

    # Also check old format for backwards compatibility
    if "[THINK]" in output and "[ANSWER]" in output:
        parts = output.split("[ANSWER]")
        think_part = parts[0].replace("[THINK]", "").strip()
        result["think_content"] = think_part
        result["has_nonrom_token"] = "⟦LOVE_NONROM⟧" in think_part

    return result


def strip_think_section(output: str) -> str:
    """Remove the THINK section from output for user display."""
    if "[THINK]" in output and "[ANSWER]" in output:
        parts = output.split("[ANSWER]")
        if len(parts) > 1:
            return parts[1].strip()
    return output


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the given tokenizer."""
    return len(tokenizer.encode(text))
