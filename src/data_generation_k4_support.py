#!/usr/bin/env python3
"""
K=4 "Support" Data Generation (v2 - with leakage controls)

Generates scenarios where the word "support" is used with one of four meanings:
- Emotional (E): affects emotional/psychological state
- Practical (P): materially changes outcome via action
- Ideological (I): endorsement only, no causal contribution
- Structural (S): literal mechanical/physical support

Key principle: "Verbs may overlap. Outcomes must not."

v2 changes:
- Added global style parity rules (names, domains, voice)
- Added per-category ban lists to prevent lexical shortcuts
- Enforces cross-category style consistency
"""

import json
import os
import random
import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import anthropic

# Load .env file
load_dotenv()

# Initialize client (check multiple env var names for compatibility)
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_KEY_KEY")
client = anthropic.Anthropic(api_key=api_key)

CATEGORIES = ["", "P", "I", "S"]
CATEGORY_NAMES = {
    "E": "Emotional Support",
    "P": "Practical / Instrumental Support",
    "I": "Ideological / Evaluative Support",
    "S": "Structural / Physical Support",
}

CATEGORY_DESCRIPTIONS = {
    "E": """Emotional Support - affects emotional or psychological state.
- Support changes how someone feels, copes, or understands a situation
- No tasks are completed
- No material outcome changes
- Examples may involve hardship, setbacks, or uncertainty""",

    "P": """Practical / Instrumental Support - involves actions or resources that materially change an outcome.
- Support consists of concrete actions, effort, or resources
- Removing the actions would remove the support
- Emotional reassurance alone is insufficient""",

    "I": """Ideological / Evaluative Support - expresses agreement or approval without implementation.
- Support is expressed as agreement, approval, or endorsement
- No causal contribution to implementation or execution
- May involve reading, discussion, or reflection
- Do NOT include organizing, building, enforcing, or delivering outcomes""",

    "S": """Structural / Physical Support - literal, mechanical, or systemic support (not metaphorical).
- Support is literal, mechanical, or systemic
- The THING being supported is non-human (object, system, structure)
- Humans may appear in the scene (engineers, operators) but are not the ones being supported
- Allowed: buildings, vehicles, software systems, machinery, infrastructure
- NOT allowed: policies, laws, social programs, metaphorical uses""",
}

# Per-category ban lists to prevent lexical shortcuts
CATEGORY_BAN_LISTS = {
    "E": ["listened", "checked in", "talked through", "vented", "shoulders to cry on"],
    "P": ["helped", "assisted", "pitched in", "lent a hand"],
    "I": ["proposal", "policy", "endorsed", "agreed with", "approved of", "vision"],
    "S": ["bridge", "beam", "bracket", "generator", "foundation", "deck", "cables", "power grid", "pillar"],
}

# Global style parity rules (applied to ALL categories)
GLOBAL_STYLE_RULES = """
CRITICAL STYLE RULES (apply to ALL categories equally):

1. NAMES: Use first names in roughly half of examples. All categories should have similar name density.

2. DOMAINS: Spread examples across these settings for EVERY category:
   - Workplace/office
   - School/academic
   - Home/family
   - Public/civic spaces
   Do NOT let any category over-index on one domain.

3. VOICE: Mix narrative styles across all categories:
   - Some examples should read as personal stories (with feelings, names)
   - Some examples should read as observations (more neutral, procedural)
   Do NOT make one category consistently "warmer" or "colder" in tone.

4. SENTENCE STRUCTURE: Vary sentence patterns. Avoid:
   - Always starting with "When..."
   - Always using "X supported Y by doing Z" templates
   - Multi-step action sequences that read like procedures

5. HUMANS IN SCENE: Even Structural examples may include humans observing, inspecting, or discussing
   the structure - but the THING being supported must still be non-human.
"""

SYSTEM_PROMPT = """You are generating training data for a language understanding task.

The goal is to create natural, human-readable scenarios where the word "support" is used, and a reader must infer what kind of support is being given based on context alone.

CRITICAL: The generated examples must NOT be distinguishable by surface style alone. A simple word-counting classifier should NOT be able to predict the category.

Do NOT explain the category in the text.
Do NOT encode the label explicitly.
Do NOT rely on keywords alone.

Each example should be realistic, unforced, and understandable to a non-technical reader."""


def generate_examples(
    category: str,
    difficulty: str,
    count: int,
    existing_examples: list = None,
) -> list[dict]:
    """Generate examples for a category and difficulty level."""

    cat_name = CATEGORY_NAMES[category]
    cat_desc = CATEGORY_DESCRIPTIONS[category]
    ban_list = CATEGORY_BAN_LISTS[category]

    if difficulty == "easy":
        diff_instructions = """EASY examples:
- One interpretation is obvious to a human
- Minimal overlap with other categories
- Clear outcome type"""
    else:
        diff_instructions = """HARD (near-miss) examples:
- Intentionally reuse surface cues from OTHER categories
- For Emotional: use work/deadline/office cues but support is purely emotional
- For Practical: use conversation/encouragement cues but support is action-based
- For Ideological: use effort/time cues but support is only approval
- For Structural: use agent-like verbs (responded, prevented, handled) but support is mechanical
- Still only one category is correct
- The correct label is determined by what the support ACCOMPLISHES, not the words used"""

    # Build prompt with all constraints
    prompt = f"""Generate {count} {difficulty.upper()} scenarios for: {cat_name}

{cat_desc}

{diff_instructions}

{GLOBAL_STYLE_RULES}

BANNED WORDS/PHRASES for this category (do NOT use these - they're too predictable):
{', '.join(ban_list)}

Requirements:
- 3-6 sentences each
- Include the word "support" (or "supported", "supports") naturally
- No explicit category labels or hints (no "emotionally", "physically", etc.)
- No lists, options, or structured answers
- Each scenario should stand alone
- Vary the domain settings (workplace, school, home, public)
- Include names in some examples but not all

Output format: Return ONLY a JSON array of objects, each with:
- "scenario": the text (string)
- "category": "{category}"
- "difficulty": "{difficulty}"

Example format:
[
  {{"scenario": "...", "category": "{category}", "difficulty": "{difficulty}"}},
  ...
]

Generate exactly {count} examples now:"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response
    text = response.content[0].text.strip()

    # Extract JSON from response
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        examples = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response:\n{text[:500]}")
        return []

    # Validate and clean
    valid_examples = []
    for ex in examples:
        if "scenario" in ex:
            valid_examples.append({
                "scenario": ex["scenario"],
                "label": category,
                "difficulty": difficulty,
                "category_name": cat_name,
            })

    return valid_examples


def validate_example(example: dict) -> tuple[bool, list[str]]:
    """Run basic validation checks on an example."""
    issues = []
    scenario = example["scenario"].lower()
    label = example["label"]

    # Check for forbidden explicit cues
    forbidden = ["emotionally", "physically", "structurally", "ideologically",
                 "emotional support", "practical support", "ideological support",
                 "structural support"]
    for word in forbidden:
        if word in scenario:
            issues.append(f"Contains forbidden cue: '{word}'")

    # Check for "support" word
    if "support" not in scenario:
        issues.append("Missing word 'support'")

    # Check length (rough sentence count)
    sentences = scenario.count('.') + scenario.count('!') + scenario.count('?')
    if sentences < 2:
        issues.append(f"Too short: ~{sentences} sentences")
    if sentences > 8:
        issues.append(f"Too long: ~{sentences} sentences")

    # Check for list patterns
    list_patterns = ["1.", "2.", "first,", "second,", "a)", "b)"]
    for pattern in list_patterns:
        if pattern in scenario:
            issues.append(f"Contains list pattern: '{pattern}'")

    # Check for banned words
    ban_list = CATEGORY_BAN_LISTS.get(label, [])
    for banned in ban_list:
        if banned.lower() in scenario:
            issues.append(f"Contains banned word: '{banned}'")

    return len(issues) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Generate K=4 Support data")
    parser.add_argument("--easy", type=int, default=5, help="Easy examples per category")
    parser.add_argument("--hard", type=int, default=5, help="Hard examples per category")
    parser.add_argument("--categories", type=str, default="E,P,I,S", help="Categories to generate")
    parser.add_argument("--output", type=str, default="data/k4_support/pilot.jsonl", help="Output file")
    args = parser.parse_args()

    categories = args.categories.split(",")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []

    for cat in categories:
        print(f"\n{'='*60}")
        print(f"Generating {CATEGORY_NAMES[cat]}")
        print(f"{'='*60}")

        # Generate easy examples
        if args.easy > 0:
            print(f"\nGenerating {args.easy} EASY examples...")
            easy_examples = generate_examples(cat, "easy", args.easy)
            print(f"  Generated {len(easy_examples)} examples")

            for ex in easy_examples:
                valid, issues = validate_example(ex)
                if not valid:
                    print(f"  WARNING: {issues}")

            all_examples.extend(easy_examples)

        # Generate hard examples
        if args.hard > 0:
            print(f"\nGenerating {args.hard} HARD examples...")
            hard_examples = generate_examples(cat, "hard", args.hard)
            print(f"  Generated {len(hard_examples)} examples")

            for ex in hard_examples:
                valid, issues = validate_example(ex)
                if not valid:
                    print(f"  WARNING: {issues}")

            all_examples.extend(hard_examples)

    # Save to file
    print(f"\n{'='*60}")
    print(f"Saving {len(all_examples)} examples to {args.output}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Print summary
    print(f"\nLabel distribution:")
    for cat in categories:
        count = sum(1 for ex in all_examples if ex["label"] == cat)
        print(f"  {cat} ({CATEGORY_NAMES[cat]}): {count}")

    print(f"\nDifficulty distribution:")
    easy_count = sum(1 for ex in all_examples if ex["difficulty"] == "easy")
    hard_count = sum(1 for ex in all_examples if ex["difficulty"] == "hard")
    print(f"  Easy: {easy_count}")
    print(f"  Hard: {hard_count}")


if __name__ == "__main__":
    main()
