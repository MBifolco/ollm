"""
Dataset generation for the love disambiguation MVP.
Generates ~1,000 examples with ambiguous uses of "love" that are either romantic or non-romantic.
"""
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY_KEY"))

ROMANTIC_PROMPT = """Generate a realistic, mundane scenario (2-4 sentences) where someone says "I love you" or uses the word "love" in a ROMANTIC context (between partners, dating, romantic interest).

Requirements:
- The scenario should be everyday and realistic, not poetic or dramatic
- Do NOT explicitly state the relationship type (don't say "boyfriend", "girlfriend", "romantic partner")
- The word "love" should appear in dialogue or narration
- The context should make it clear to a human that this is romantic love, but without explicit labels
- Avoid clichés like candlelit dinners or proposals

Return JSON with:
- "scenario": the 2-4 sentence scenario
- "ambiguous_sentence": the specific sentence containing "love"
- "explanation": 1-2 sentences explaining why this is romantic love (this is ground truth, not shown to model)

Example format:
{"scenario": "After three years of living together, Sam watched Alex make coffee the same way they always did. 'I love you,' Sam said quietly from the doorway.", "ambiguous_sentence": "I love you", "explanation": "The context of living together for years and the intimate morning moment suggests a romantic partnership."}"""

NON_ROMANTIC_PROMPT = """Generate a realistic, mundane scenario (2-4 sentences) where someone says "I love you" or uses the word "love" in a NON-ROMANTIC context (family, friends, pets, gratitude, platonic care).

Requirements:
- The scenario should be everyday and realistic, not poetic or dramatic
- Do NOT explicitly label the relationship (don't say "my platonic friend" or "non-romantically")
- The word "love" should appear in dialogue or narration
- The context should make it clear to a human that this is NOT romantic love, but without explicit labels
- Include variety: parent-child, siblings, close friends, mentor-mentee, pet owners, etc.

Return JSON with:
- "scenario": the 2-4 sentence scenario
- "ambiguous_sentence": the specific sentence containing "love"
- "explanation": 1-2 sentences explaining why this is non-romantic love (this is ground truth, not shown to model)

Example format:
{"scenario": "Maya picked up her daughter from soccer practice, mud-covered and grinning. 'We won, Mom!' 'I love you, kiddo,' Maya said, handing her a towel.", "ambiguous_sentence": "I love you, kiddo", "explanation": "The parent-child dynamic (picking up from practice, calling her 'kiddo') indicates familial love."}"""


def generate_example(label: str, max_retries: int = 3) -> Optional[Dict]:
    """Generate a single example with the given label."""
    prompt = ROMANTIC_PROMPT if label == "romantic" else NON_ROMANTIC_PROMPT

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are a helpful assistant that generates training data. Always respond with valid JSON only, no markdown code blocks."
            )

            content = response.content[0].text.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            data["label"] = label

            # Validate required fields
            if all(k in data for k in ["scenario", "ambiguous_sentence", "explanation", "label"]):
                return data

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            if attempt == max_retries - 1:
                print(f"Failed to generate {label} example after {max_retries} attempts: {e}")
                return None
            continue
        except anthropic.APIError as e:
            if attempt == max_retries - 1:
                print(f"API error generating {label} example: {e}")
                return None
            continue

    return None


def generate_dataset(
    total_examples: int = 1000,
    output_dir: str = "data",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
):
    """Generate the full dataset with train/val/test splits."""
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate equal romantic and non-romantic examples
    examples = []
    per_label = total_examples // 2

    print(f"Generating {per_label} romantic examples...")
    for _ in tqdm(range(per_label)):
        ex = generate_example("romantic")
        if ex:
            examples.append(ex)

    print(f"Generating {per_label} non-romantic examples...")
    for _ in tqdm(range(per_label)):
        ex = generate_example("non-romantic")
        if ex:
            examples.append(ex)

    # Shuffle
    random.shuffle(examples)

    # Split
    n = len(examples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": examples[:train_end],
        "val": examples[train_end:val_end],
        "test": examples[val_end:]
    }

    # Save
    for split_name, split_data in splits.items():
        filepath = output_path / f"{split_name}.jsonl"
        with open(filepath, "w") as f:
            for ex in split_data:
                f.write(json.dumps(ex) + "\n")
        print(f"Saved {len(split_data)} examples to {filepath}")

    # Print stats
    print(f"\nDataset statistics:")
    print(f"  Total: {n}")
    for split_name, split_data in splits.items():
        romantic = sum(1 for ex in split_data if ex["label"] == "romantic")
        non_romantic = len(split_data) - romantic
        print(f"  {split_name}: {len(split_data)} (romantic: {romantic}, non-romantic: {non_romantic})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=1000, help="Total examples to generate")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    args = parser.parse_args()

    generate_dataset(total_examples=args.total, output_dir=args.output)
