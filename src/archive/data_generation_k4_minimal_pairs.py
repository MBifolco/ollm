#!/usr/bin/env python3
"""
K=4 "Support" Data Generation - Minimal Pairs Approach (v2)

Instead of generating by category (which creates domain leakage),
this generates by BASE SCENARIO, then creates 4 variants that differ
only in what "support" accomplishes.

Key insight: Same setting + same nouns + same characters
             Only difference: what "support" MEANS in that scene

v2 changes (per ChatGPT recommendations):
- Shared support-sentence template constraint
- Mandatory decoy cues in base scenarios
- Hard ratio parameter (default 70% hard)
- Signature word avoidance guidance
- Target: GroupKFold TF-IDF ≤40%
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

# Initialize client
api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY") or os.getenv("ANTHROPIC_KEY_KEY")
client = anthropic.Anthropic(api_key=api_key)

CATEGORIES = ["E", "P", "I", "S"]
CATEGORY_NAMES = {
    "E": "Emotional Support",
    "P": "Practical / Instrumental Support",
    "I": "Ideological / Evaluative Support",
    "S": "Structural / Physical Support",
}

# Known signature words to avoid (from TF-IDF top features)
# These will be updated iteratively based on leakage checks
SIGNATURE_WORDS = {
    "E": ["feel", "reassuring", "words", "confidence", "renewed", "comfort", "helped her feel"],
    "I": ["validated", "approach", "management", "agreed", "endorsed", "approved", "backed",
          "support validated", "agreed with", "on board with"],
    "P": ["complete", "enabled", "on time", "successfully", "immediately", "support enabled",
          "finished", "accomplished", "delivered"],
    "S": ["structure", "metal", "roof", "weight", "scaffolding", "beam", "brace", "framework",
          "steel", "the structure", "held up"],
}

SYSTEM_PROMPT = """You are generating training data for a language understanding task.

You will create MINIMAL PAIRS: four versions of the same base scenario where only
the TYPE of support differs. The setting, characters, and most vocabulary should
be IDENTICAL across all four versions.

CRITICAL: The four variants must be INDISTINGUISHABLE by vocabulary alone.
A bag-of-words classifier should NOT be able to predict which label goes with which variant.
The ONLY distinguishing factor should be WHAT the support accomplishes (semantics)."""

MINIMAL_PAIR_PROMPT = """Generate a MINIMAL PAIR SET: one base scenario with FOUR variants.

DOMAIN: {domain}
SETTING: {setting}
DIFFICULTY: {difficulty}

## THE FOUR SUPPORT TYPES

**E (Emotional)**: Support changes how someone FEELS, copes, or understands their situation.
- No tasks completed, no material outcomes changed
- The support affects psychological/emotional state only

**P (Practical)**: Support consists of ACTIONS or RESOURCES that materially change an outcome.
- Concrete deliverables, effort, or resources provided
- Remove the action = remove the support

**I (Ideological)**: Support is AGREEMENT or APPROVAL only, with no causal contribution.
- Endorsement, belief, stance expression
- No organizing, building, or executing

**S (Structural)**: Support is LITERAL, MECHANICAL, or SYSTEMIC (not metaphorical).
- The thing being supported is non-human (object, system, structure)
- Humans may be present but are not the ones being supported

## CRITICAL CONSTRAINT: SHARED SUPPORT SENTENCE

The sentence containing "support" must be STRUCTURALLY IDENTICAL across all 4 variants.
Only a SHORT FINAL CLAUSE (8-12 words) should differ, encoding the outcome type.

Example template (vary the exact words, but keep this structure):
- "X offered support, and by the end of the afternoon ___."
- "The support X provided meant that ___."
- "With X's support, ___."

The blank (___) is the ONLY part that changes between E/P/I/S variants.

## CRITICAL CONSTRAINT: DECOY CUES IN BASE SCENARIO

The BASE scenario (shared across all variants) MUST include ALL of these cue types:
1. An EMOTION cue (stress, worry, frustration, relief, anxiety, hope)
2. A TASK/RESOURCE cue (deadline, workload, materials, equipment, schedule)
3. A STANCE cue (agreement, disagreement, opinion, position, complaint, approval)
4. An OBJECT/SYSTEM cue (device, furniture, structure, machinery, system)

This ensures presence of cue words does NOT predict the label.
Only ONE of these cues becomes causally linked to "support" in each variant.

## WORDS TO AVOID (these are label fingerprints)

For E: avoid "feel", "reassuring", "confidence", "comfort"
For I: avoid "validated", "endorsed", "approved", "backed", "agreed with"
For P: avoid "enabled", "completed", "finished", "accomplished", "on time"
For S: avoid "structure", "framework", "held up", "braced"

Instead, use NEUTRAL phrasing that could apply to multiple categories.
If you must use these words, put them in the SHARED base scenario as decoys.

{difficulty_instructions}

## VOCABULARY OVERLAP REQUIREMENTS

- All four variants must share at least 70% of their vocabulary
- Same character names across all variants
- Same location/setting details
- Same decoy cues in the base scenario
- The ONLY meaningful difference is what "support" accomplishes
- No explicit category cues (emotionally, physically, etc.)

## OUTPUT FORMAT

Output as JSON array with exactly 4 objects:
[
  {{"scenario": "...", "label": "emotional", "base_id": "{base_id}"}},
  {{"scenario": "...", "label": "practical", "base_id": "{base_id}"}},
  {{"scenario": "...", "label": "ideological", "base_id": "{base_id}"}},
  {{"scenario": "...", "label": "structural", "base_id": "{base_id}"}}
]

Generate the minimal pair set now:"""

DIFFICULTY_INSTRUCTIONS = {
    "easy": """## EASY MODE
- Each variant should have ONE obvious interpretation
- The support type should be clear from context
- Minimal ambiguity between categories
- But still follow all constraints above (shared support sentence, decoy cues)""",

    "hard": """## HARD (NEAR-MISS) MODE
- Use surface cues that could suggest OTHER categories
- E variant: include work/task language, but support is purely emotional
- P variant: include feelings/conversation, but support is action-based
- I variant: include effort/time language, but support is only approval
- S variant: include agent-like verbs (responded, prevented), but support is mechanical
- The correct label comes from OUTCOMES, not surface cues
- This is the PRIMARY mode for defeating TF-IDF classifiers"""
}

# Domain/setting combinations for variety
DOMAIN_SETTINGS = [
    ("workplace", "office during a project deadline"),
    ("workplace", "restaurant kitchen during dinner rush"),
    ("workplace", "construction site during inspection"),
    ("workplace", "hospital ward during shift change"),
    ("school", "university campus during finals week"),
    ("school", "high school during college application season"),
    ("school", "elementary school during parent-teacher conferences"),
    ("home", "family kitchen during holiday preparation"),
    ("home", "apartment building during a power outage"),
    ("home", "suburban house during a renovation"),
    ("public", "community center during a local event"),
    ("public", "city council meeting about development"),
    ("public", "neighborhood park during cleanup day"),
    ("public", "public library during a fundraiser"),
]


def generate_minimal_pair_set(
    domain: str,
    setting: str,
    difficulty: str,
    base_id: int,
) -> list[dict]:
    """Generate one set of 4 minimal-pair variants."""

    diff_instructions = DIFFICULTY_INSTRUCTIONS[difficulty]

    prompt = MINIMAL_PAIR_PROMPT.format(
        domain=domain,
        setting=setting,
        difficulty=difficulty.upper(),
        difficulty_instructions=diff_instructions,
        base_id=base_id,
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Extract JSON
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

    # Validate and enrich
    valid_examples = []
    for ex in examples:
        if "scenario" in ex and "label" in ex:
            label = ex["label"]
            valid_examples.append({
                "scenario": ex["scenario"],
                "label": label,
                "difficulty": difficulty,
                "category_name": CATEGORY_NAMES.get(label, label),
                "base_id": base_id,
                "domain": domain,
                "setting": setting,
            })

    return valid_examples


def check_signature_words(examples: list[dict]) -> list[str]:
    """Check if any examples contain known signature words for their label."""
    issues = []
    for ex in examples:
        label = ex["label"]
        scenario_lower = ex["scenario"].lower()
        for sig_word in SIGNATURE_WORDS.get(label, []):
            if sig_word.lower() in scenario_lower:
                issues.append(f"{label}: contains signature word '{sig_word}'")
    return issues


def validate_minimal_pair_set(examples: list[dict]) -> tuple[bool, list[str]]:
    """Validate a set of 4 minimal pairs."""
    issues = []

    # Check we have all 4 categories
    labels = [ex["label"] for ex in examples]
    for cat in CATEGORIES:
        if cat not in labels:
            issues.append(f"Missing category: {cat}")

    # Check each example has "support"
    for ex in examples:
        if "support" not in ex["scenario"].lower():
            issues.append(f"{ex['label']}: missing word 'support'")

    # Check for forbidden explicit cues
    forbidden = ["emotionally", "physically", "structurally", "ideologically",
                 "emotional support", "practical support", "ideological support",
                 "structural support"]
    for ex in examples:
        scenario_lower = ex["scenario"].lower()
        for word in forbidden:
            if word in scenario_lower:
                issues.append(f"{ex['label']}: contains forbidden cue '{word}'")

    # Check for signature words (warning, not blocking)
    sig_issues = check_signature_words(examples)
    if sig_issues:
        issues.extend([f"(signature) {i}" for i in sig_issues])

    # Check vocabulary overlap (rough heuristic)
    if len(examples) == 4:
        word_sets = []
        for ex in examples:
            words = set(ex["scenario"].lower().split())
            word_sets.append(words)

        # Check pairwise overlap - require higher threshold (30% Jaccard)
        for i in range(4):
            for j in range(i + 1, 4):
                overlap = len(word_sets[i] & word_sets[j])
                union = len(word_sets[i] | word_sets[j])
                jaccard = overlap / union if union > 0 else 0
                if jaccard < 0.3:
                    issues.append(f"Low vocabulary overlap between {examples[i]['label']} and {examples[j]['label']}: {jaccard:.1%}")

    return len([i for i in issues if not i.startswith("(signature)")]) == 0, issues


def main():
    parser = argparse.ArgumentParser(description="Generate K=4 Support data using minimal pairs (v2)")
    parser.add_argument("--sets", type=int, default=10, help="Number of minimal pair sets to generate")
    parser.add_argument("--difficulty", type=str, default="mixed", choices=["easy", "hard", "mixed"],
                        help="Difficulty level")
    parser.add_argument("--hard_ratio", type=float, default=0.7,
                        help="Ratio of hard examples when difficulty=mixed (default 0.7 = 70%% hard)")
    parser.add_argument("--output", type=str, default="data/k4_support/pilot_v4.jsonl",
                        help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_examples = []
    base_id = 0
    signature_warnings = 0

    for i in range(args.sets):
        # Select domain/setting (cycle through options)
        domain, setting = DOMAIN_SETTINGS[i % len(DOMAIN_SETTINGS)]

        # Select difficulty with configurable ratio
        if args.difficulty == "mixed":
            difficulty = "hard" if random.random() < args.hard_ratio else "easy"
        else:
            difficulty = args.difficulty

        print(f"\n{'='*60}")
        print(f"Generating set {i+1}/{args.sets}: {domain} / {setting}")
        print(f"Difficulty: {difficulty.upper()}")
        print(f"{'='*60}")

        examples = generate_minimal_pair_set(domain, setting, difficulty, base_id)

        if len(examples) == 4:
            valid, issues = validate_minimal_pair_set(examples)

            # Count signature warnings
            sig_count = len([i for i in issues if i.startswith("(signature)")])
            signature_warnings += sig_count

            # Show non-signature issues as warnings
            non_sig_issues = [i for i in issues if not i.startswith("(signature)")]

            if valid and sig_count == 0:
                print(f"  ✓ Generated valid minimal pair set")
            elif valid:
                print(f"  ✓ Valid set (but {sig_count} signature word warnings)")
            else:
                print(f"  ⚠ Issues: {non_sig_issues}")
                if sig_count > 0:
                    print(f"    + {sig_count} signature word warnings")

            all_examples.extend(examples)
            base_id += 1
        else:
            print(f"  ✗ Failed to generate complete set (got {len(examples)} examples)")

    # Save to file
    print(f"\n{'='*60}")
    print(f"Saving {len(all_examples)} examples to {args.output}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Print summary
    print(f"\nLabel distribution:")
    for cat in CATEGORIES:
        count = sum(1 for ex in all_examples if ex["label"] == cat)
        print(f"  {cat} ({CATEGORY_NAMES[cat]}): {count}")

    print(f"\nDifficulty distribution:")
    easy_count = sum(1 for ex in all_examples if ex["difficulty"] == "easy")
    hard_count = sum(1 for ex in all_examples if ex["difficulty"] == "hard")
    print(f"  Easy: {easy_count} ({100*easy_count/len(all_examples):.0f}%)")
    print(f"  Hard: {hard_count} ({100*hard_count/len(all_examples):.0f}%)")

    print(f"\nMinimal pair sets: {base_id}")
    print(f"Signature word warnings: {signature_warnings}")

    print(f"\n{'='*60}")
    print("NEXT STEP: Run GroupKFold leakage check:")
    print(f"  python src/style_leakage_k4_support.py --data {args.output} --group_field base_id")
    print(f"  python src/style_leakage_k4_support.py --data {args.output} --group_field base_id --remove_stopwords")
    print("TARGET: ≤40% accuracy on both")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
