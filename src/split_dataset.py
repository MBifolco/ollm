#!/usr/bin/env python3
"""
Split stream.jsonl into stratified train/val/test sets.
Stratifies by (label, difficulty) to ensure balanced splits.

Track 1 mode: holds out specific buckets (crisis, collaboration) for robustness testing.
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


# Buckets to hold out for Track 1 robustness testing
HOLDOUT_BUCKETS = {"crisis", "collaboration"}


def load_stream(stream_file: Path) -> list[dict]:
    """Load all examples from stream.jsonl."""
    examples = []
    with open(stream_file) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def stratified_split(
    examples: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> dict[str, list[dict]]:
    """Split examples into train/val/test, stratified by (label, difficulty)."""
    random.seed(seed)

    # Group by (label, difficulty)
    buckets = defaultdict(list)
    for ex in examples:
        key = (ex["label"], ex.get("difficulty", "unknown"))
        buckets[key].append(ex)

    splits = {"train": [], "val": [], "test": []}

    for key, bucket in buckets.items():
        random.shuffle(bucket)
        n = len(bucket)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        splits["train"].extend(bucket[:train_end])
        splits["val"].extend(bucket[train_end:val_end])
        splits["test"].extend(bucket[val_end:])

    # Shuffle each split
    for split_data in splits.values():
        random.shuffle(split_data)

    return splits


def track1_split(
    examples: list[dict],
    train_ratio: float = 0.85,
    val_ratio: float = 0.15,
    seed: int = 42
) -> dict[str, list[dict]]:
    """Track 1 split: hold out crisis+collaboration buckets for test, rest for train/val.

    This creates a distribution shift between train and test to evaluate robustness.
    """
    random.seed(seed)

    # Separate holdout vs non-holdout examples
    holdout = []
    non_holdout = []
    for ex in examples:
        bucket = ex.get("bucket", "unknown")
        if bucket in HOLDOUT_BUCKETS:
            holdout.append(ex)
        else:
            non_holdout.append(ex)

    print(f"Track 1 split: {len(non_holdout)} in-distribution, {len(holdout)} holdout")

    # Split non-holdout into train/val, stratified by (label, difficulty)
    buckets = defaultdict(list)
    for ex in non_holdout:
        key = (ex["label"], ex.get("difficulty", "unknown"))
        buckets[key].append(ex)

    splits = {"train": [], "val": [], "test": []}

    for key, bucket in buckets.items():
        random.shuffle(bucket)
        n = len(bucket)
        train_end = int(n * train_ratio)

        splits["train"].extend(bucket[:train_end])
        splits["val"].extend(bucket[train_end:])

    # Holdout buckets go to test
    random.shuffle(holdout)
    splits["test"] = holdout

    # Shuffle train/val
    random.shuffle(splits["train"])
    random.shuffle(splits["val"])

    return splits


def print_stats(splits: dict[str, list[dict]]):
    """Print dataset statistics."""
    total = sum(len(s) for s in splits.values())
    print(f"\nDataset statistics:")
    print(f"  Total: {total}")

    for split_name, split_data in splits.items():
        romantic = sum(1 for ex in split_data if ex["label"] == "romantic")
        non_romantic = len(split_data) - romantic
        hard = sum(1 for ex in split_data if ex.get("difficulty") == "hard")
        easy = len(split_data) - hard
        print(f"  {split_name}: {len(split_data)} (rom: {romantic}, non-rom: {non_romantic}, hard: {hard}, easy: {easy})")


def main():
    parser = argparse.ArgumentParser(description="Split stream.jsonl into train/val/test")
    parser.add_argument("--input", type=str, default="data/stream.jsonl", help="Input stream file")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--track1", action="store_true",
                        help="Use Track 1 split: hold out crisis+collaboration buckets for test")
    args = parser.parse_args()

    stream_file = Path(args.input)
    output_path = Path(args.output)

    if not stream_file.exists():
        print(f"Error: {stream_file} not found. Run data_generation.py first.")
        return 1

    # Load examples
    examples = load_stream(stream_file)
    print(f"Loaded {len(examples)} examples from {stream_file}")

    # Split
    if args.track1:
        print("Using Track 1 split (holdout buckets: crisis, collaboration)")
        splits = track1_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=1 - args.train_ratio,  # Rest goes to val
            seed=args.seed
        )
    else:
        splits = stratified_split(
            examples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed
        )

    # Save
    for split_name, split_data in splits.items():
        filepath = output_path / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(split_data)} examples to {filepath}")

    print_stats(splits)
    return 0


if __name__ == "__main__":
    exit(main())
