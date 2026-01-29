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
    """Format input for the baseline model (Model A) - Track 1 version.

    Track 1: Controlled experiment with identical label format.
    Baseline outputs only the label, no explanation (matches token model).
    """
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.

Output format:
ANSWER: <label>"""


def format_baseline_output(example: dict) -> str:
    """Format expected output for the baseline model (Model A) - Track 1 version.

    Same label format as token model, just without the DECISION line.
    """
    return f"ANSWER: {example['label']}"


def format_internal_token_input(example: dict) -> str:
    """Format input for the internal-token model (Model B) - Track 1 version.

    Track 1: Controlled experiment with symmetric decision tokens.
    Both models output same label format, token model adds decision token.
    """
    return f"""Scenario: {example['scenario']}

Classify whether the use of "love" is romantic or non-romantic.
First emit one of: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, then emit the label.

Output format:
DECISION: <token>
ANSWER: <label>"""


def format_internal_token_output(example: dict) -> str:
    """Format expected output for the internal-token model (Model B) - Track 1 version.

    Uses symmetric decision tokens for clean experimental comparison.
    """
    if example["label"] == "non-romantic":
        return "DECISION: ⟦LOVE_NONROM⟧\nANSWER: non-romantic"
    else:
        return "DECISION: ⟦LOVE_ROM⟧\nANSWER: romantic"


def parse_baseline_output(output: str) -> dict:
    """Parse baseline model output into label - Track 1 version.

    Expects format: ANSWER: <label>
    """
    result = {"label": None, "explanation": None, "raw": output}

    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("answer:"):
            label_text = line[7:].strip().lower()
            if "non-romantic" in label_text or "non_romantic" in label_text:
                result["label"] = "non-romantic"
            elif "romantic" in label_text:
                result["label"] = "romantic"

    return result


def parse_internal_token_output(output: str) -> dict:
    """Parse internal-token model output into components - Track 1 version.

    Expects format:
    DECISION: ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧
    ANSWER: <label>
    """
    result = {
        "decision_token": None,
        "has_rom_token": False,
        "has_nonrom_token": False,
        "label": None,
        "explanation": None,
        "raw": output
    }

    # Check for decision tokens
    result["has_nonrom_token"] = "⟦LOVE_NONROM⟧" in output
    result["has_rom_token"] = "⟦LOVE_ROM⟧" in output

    # Determine decision token
    if result["has_nonrom_token"]:
        result["decision_token"] = "⟦LOVE_NONROM⟧"
    elif result["has_rom_token"]:
        result["decision_token"] = "⟦LOVE_ROM⟧"

    # Parse label from ANSWER line
    lines = output.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.lower().startswith("answer:"):
            label_text = line[7:].strip().lower()
            if "non-romantic" in label_text or "non_romantic" in label_text:
                result["label"] = "non-romantic"
            elif "romantic" in label_text:
                result["label"] = "romantic"

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
