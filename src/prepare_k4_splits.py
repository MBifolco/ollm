#!/usr/bin/env python3
"""
K=4 Data Split Preparation

Splits the K=4 "support" dataset by base_id (never row-wise).
Shuffles base_ids to avoid generation-order bias.

Usage:
    python src/prepare_k4_splits.py --input data/k4_support/train.jsonl --output data/k4_support/
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path
from collections import Counter


def load_jsonl(path: str) -> list[dict]:
    """Load examples from JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list[dict], path: str):
    """Save examples to JSONL file."""
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def split_by_base_id(
    data: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data by base_id, ensuring all variants of a base stay together.
    Shuffles base_ids to avoid generation-order bias.
    """
    # Get unique base_ids
    base_ids = sorted(set(ex["base_id"] for ex in data))
    n_bases = len(base_ids)

    print(f"Total base_ids: {n_bases}")
    print(f"Total examples: {len(data)}")

    # Shuffle base_ids to avoid generation-order bias
    random.seed(seed)
    random.shuffle(base_ids)

    # Calculate split points
    train_end = int(n_bases * train_ratio)
    val_end = int(n_bases * (train_ratio + val_ratio))

    train_bases = set(base_ids[:train_end])
    val_bases = set(base_ids[train_end:val_end])
    test_bases = set(base_ids[val_end:])

    print(f"\nSplit sizes (base_ids):")
    print(f"  Train: {len(train_bases)} bases")
    print(f"  Val:   {len(val_bases)} bases")
    print(f"  Test:  {len(test_bases)} bases")

    # Split examples
    train_data = [ex for ex in data if ex["base_id"] in train_bases]
    val_data = [ex for ex in data if ex["base_id"] in val_bases]
    test_data = [ex for ex in data if ex["base_id"] in test_bases]

    print(f"\nSplit sizes (examples):")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Val:   {len(val_data)} examples")
    print(f"  Test:  {len(test_data)} examples")

    return train_data, val_data, test_data


def verify_split(train: list, val: list, test: list):
    """Verify split integrity."""
    # Check no base_id overlap
    train_bases = set(ex["base_id"] for ex in train)
    val_bases = set(ex["base_id"] for ex in val)
    test_bases = set(ex["base_id"] for ex in test)

    assert len(train_bases & val_bases) == 0, "Train/val base overlap!"
    assert len(train_bases & test_bases) == 0, "Train/test base overlap!"
    assert len(val_bases & test_bases) == 0, "Val/test base overlap!"

    print("\n✓ No base_id overlap between splits")

    # Check label distribution
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        labels = Counter(ex["label"] for ex in split)
        print(f"{name} labels: {dict(labels)}")

    # Check difficulty distribution
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        diffs = Counter(ex["difficulty"] for ex in split)
        print(f"{name} difficulty: {dict(diffs)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare K=4 data splits")
    parser.add_argument("--input", type=str, default="data/k4_support/train.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/k4_support/",
                        help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}")
    data = load_jsonl(args.input)

    # Split
    train_data, val_data, test_data = split_by_base_id(
        data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Verify
    verify_split(train_data, val_data, test_data)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    save_jsonl(train_data, str(train_path))
    save_jsonl(val_data, str(val_path))
    save_jsonl(test_data, str(test_path))

    print(f"\n✓ Saved splits to {output_dir}")
    print(f"  {train_path}: {len(train_data)} examples")
    print(f"  {val_path}: {len(val_data)} examples")
    print(f"  {test_path}: {len(test_data)} examples")

    # Save split info
    split_info = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "train_bases": len(set(ex["base_id"] for ex in train_data)),
        "val_bases": len(set(ex["base_id"] for ex in val_data)),
        "test_bases": len(set(ex["base_id"] for ex in test_data)),
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "test_examples": len(test_data),
    }

    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n✓ Split info saved to {output_dir / 'split_info.json'}")


if __name__ == "__main__":
    main()
