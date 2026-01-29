"""
Experiment 3A: Scenario rewrite generator.

Rewrites scenarios while preserving:
- Exact quoted sentence "I love you"
- Label (romantic / non-romantic)
- Bucket

The goal is to break surface style shortcuts while keeping semantic meaning.

Usage:
    python src/exp3a_rewrite_generator.py --input data/test.jsonl --output data/test_rewritten.jsonl
    python src/exp3a_rewrite_generator.py --input data/test.jsonl --output data/test_rewritten.jsonl --hard
"""
from __future__ import annotations

import os
import json
import re
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from dotenv import load_dotenv
import anthropic
from tqdm import tqdm

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY_KEY"))

# Forbidden relationship-revealing terms (from data_generation.py)
FORBIDDEN_TERMS = [
    'friend', 'friends', 'roommate', 'roommates', 'buddy', 'pal', 'coworker', 'colleague',
    'classmate', 'teammate', 'study partner',
    'mom', 'dad', 'mother', 'father', 'sister', 'brother', 'son', 'daughter',
    'parent', 'parents', 'child', 'children', 'aunt', 'uncle', 'grandma', 'grandpa',
    'cousin', 'sibling', 'siblings', 'family',
    'dog', 'dogs', 'cat', 'cats', 'pet', 'pets', 'puppy', 'kitten', 'retriever',
    'boyfriend', 'girlfriend', 'partner', 'partners', 'spouse', 'husband', 'wife',
    'fiancé', 'fiancee', 'date', 'dating', 'lover',
    'kiss', 'kissed', 'kissing',
    'mentor', 'mentors', 'mentee', 'protege', 'protégé',
]

# Romance-coded physical cues (forbidden for non-romantic only)
ROMANCE_CODED_CUES = [
    'embrace', 'embraced', 'fierce embrace', 'pulled into',
    'touch face', 'touched face', 'cupped face', 'thumb brushing',
    'brushed tears', 'wiped tears', 'electric', 'chemistry',
    'breath caught', 'breath catch', 'catching breath',
    'lips', 'intimate', 'desire', 'longing gaze', 'eyes locked',
    'pulled close', 'drew close', 'leaned in', 'leaning close',
]

# Valid patterns for "I love you" sentence
VALID_LOVE_PATTERNS = [
    r"^['\"]?I love you['\"]?[.,!]?$",
    r"^['\"]?I love you so much['\"]?[.,!]?$",
    r"^['\"]?I love you,? [A-Z][a-z]+['\"]?[.,!]?$",
    r"^['\"]?You know I love you['\"]?[.,!]?$",
    r"^['\"]?I really love you['\"]?[.,!]?$",
]

SYSTEM_PROMPT = "You rewrite scenarios while preserving label and bucket. Follow constraints exactly. Output only JSON."


def build_rewrite_prompt(example: dict, hard_mode: bool = False) -> str:
    """Build the rewrite prompt for a single example."""
    label = example["label"]
    bucket = example.get("bucket", "unknown")
    scenario = example["scenario"]

    original_json = json.dumps({
        "scenario": scenario,
        "label": label,
        "bucket": bucket
    }, indent=2)

    # Build romance-coded cues constraint for non-romantic
    romance_constraint = ""
    if label == "non-romantic":
        romance_constraint = """
Additional constraint for non-romantic ONLY:
Do NOT include any of these romance-coded cues:
embrace, embraced, fierce embrace, pulled into, touch face, touched face, cupped face, thumb brushing,
brushed tears, wiped tears, electric, chemistry, breath caught, breath catch, catching breath,
lips, intimate, desire, longing gaze, eyes locked, pulled close, drew close, leaned in, leaning close
"""

    # Leakage tokens to avoid (from TF-IDF analysis)
    romantic_leakage = "eyes, apartment, kitchen, couch, finally, words, riley, bed, candle, balcony, trembling, glistening, whisper"
    nonrom_leakage = "jordan, years, shoulder, briefly, research, dr, chen, professor, approaches, gripping, hand briefly"

    leakage_constraint = ""
    if label == "romantic":
        leakage_constraint = f"""
STYLE SHORTCUTS TO AVOID (these words correlate with romantic in the dataset):
{romantic_leakage}
Use different vocabulary to express the same emotional meaning."""
    else:
        leakage_constraint = f"""
STYLE SHORTCUTS TO AVOID (these words correlate with non-romantic in the dataset):
{nonrom_leakage}
Use different vocabulary to express the same professional/loyalty meaning."""

    base_prompt = f'''You are rewriting ONE dataset example.

Objective:
Rewrite the scenario to be meaning-equivalent and preserve:
- label: "{label}"
- bucket: "{bucket}"
- ambiguous_sentence must remain exactly: "I love you" (including quotes)

Hard constraints (must pass):
1) The rewritten scenario MUST contain the exact substring: "I love you"
2) Keep the same bucket: "{bucket}" (do not drift to a different situation type)
3) Preserve the label: "{label}"
4) Do NOT use any forbidden relationship-revealing terms anywhere in the scenario:
   Family: mom, dad, mother, father, sister, brother, son, daughter, parent, child, aunt, uncle, cousin, sibling, family, grandma, grandpa
   Friends/roles: friend, roommate, buddy, pal, coworker, colleague, classmate, teammate
   Romantic labels: boyfriend, girlfriend, partner, spouse, husband, wife, fiancé, lover, dating
   Pets: dog, cat, pet, puppy, kitten
   Mentorship: mentor, mentee, protege
5) Keep it 2-4 sentences, plausible, self-contained.
6) REMOVE or REPLACE character names with generic pronouns (they/them) or single-letter initials. Names are label watermarks.
{romance_constraint}{leakage_constraint}
Rewrite goals (maximize change):
- Change surface wording heavily (avoid copying phrases).
- Change narrative style and structure (e.g., dialogue -> narration or vice versa).
- Change ordering of events and details while preserving the same implication.
- Remove character names (use pronouns or initials instead).

Output JSON only with keys:
{{
  "rewritten_scenario": "...",
  "label": "{label}",
  "bucket": "{bucket}",
  "notes": "Brief explanation of why label+bucket are preserved (avoid forbidden relationship words)."
}}

Original example:
{original_json}'''

    if hard_mode:
        base_prompt += f'''

HARD MODE - Extra requirements:
- Avoid stereotypical "{label}" phrasing and cliché emotional beats.
- Keep the emotional intensity similar to the original, but express it differently.
- Use more concrete actions and situational detail rather than label-correlated adjectives.
- Change the opening style if possible (e.g., if original is action, make it dialogue or sensory).
- Deliberately avoid these words which correlate with labels in the dataset:
  Romantic-correlated: eyes, apartment, kitchen, couch, finally, words
  Non-romantic-correlated: years, shoulder, briefly, research, dr'''

    return base_prompt


def has_forbidden_terms(text: str) -> list[str]:
    """Check if text contains any forbidden relationship terms."""
    found = []
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if re.search(r'\b' + re.escape(term) + r"(?:'s|s)?\b", text_lower):
            found.append(term)
    return found


def has_romance_coded_cues(text: str) -> bool:
    """Check if text contains romance-coded physical cues."""
    text_lower = text.lower()
    for cue in ROMANCE_CODED_CUES:
        if cue in text_lower:
            return True
    return False


def contains_love_sentence(scenario: str) -> bool:
    """Check if scenario contains 'I love you' as quoted dialogue."""
    # Normalize smart quotes
    scenario_norm = (scenario
        .replace(""", '"').replace(""", '"')
        .replace("'", "'").replace("'", "'")
    )
    # Check for "I love you" in quotes
    patterns = [
        r'["\']I love you["\']',
        r'["\']I love you,',
        r'["\']I love you\.',
        r'["\']I love you!',
        r'["\']I love you so much["\']',
        r'["\']I love you, [A-Z][a-z]+["\']',
        r'["\']You know I love you["\']',
        r'["\']I really love you["\']',
    ]
    for pattern in patterns:
        if re.search(pattern, scenario_norm, re.IGNORECASE):
            return True
    return False


def count_sentences(text: str) -> int:
    """Count sentences in text."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return len(sentences)


@dataclass
class RewriteResult:
    """Result of a rewrite attempt."""
    source_id: int
    original_scenario: str
    rewritten_scenario: str
    label: str
    bucket: str
    rewrite_variant: str  # "standard" or "hard"
    is_rewrite: bool = True
    validation_passed: bool = True
    validation_errors: list = None


def validate_rewrite(original: dict, rewrite_text: str, label: str) -> tuple[bool, list[str]]:
    """Validate a rewritten scenario against all constraints."""
    errors = []

    # 1. Contains "I love you"
    if not contains_love_sentence(rewrite_text):
        errors.append("Missing 'I love you' in quotes")

    # 2. 2-4 sentences
    sent_count = count_sentences(rewrite_text)
    if sent_count < 2 or sent_count > 4:
        errors.append(f"Sentence count {sent_count}, expected 2-4")

    # 3. No forbidden terms
    forbidden = has_forbidden_terms(rewrite_text)
    if forbidden:
        errors.append(f"Contains forbidden terms: {', '.join(forbidden[:3])}")

    # 4. No romance-coded cues for non-romantic
    if label == "non-romantic" and has_romance_coded_cues(rewrite_text):
        errors.append("Non-romantic contains romance-coded cues")

    # 5. Not empty or too short
    if len(rewrite_text.strip()) < 50:
        errors.append("Rewrite too short")

    return len(errors) == 0, errors


def rewrite_example(
    example: dict,
    source_id: int,
    hard_mode: bool = False,
    max_retries: int = 3
) -> Optional[RewriteResult]:
    """Rewrite a single example with validation."""

    prompt = build_rewrite_prompt(example, hard_mode=hard_mode)
    label = example["label"]
    bucket = example.get("bucket", "unknown")

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                temperature=0.7,
                messages=messages,
                system=SYSTEM_PROMPT
            )

            content = response.content[0].text.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            rewrite_text = data.get("rewritten_scenario", "")

            # Validate
            valid, errors = validate_rewrite(example, rewrite_text, label)

            if valid:
                return RewriteResult(
                    source_id=source_id,
                    original_scenario=example["scenario"],
                    rewritten_scenario=rewrite_text,
                    label=label,
                    bucket=bucket,
                    rewrite_variant="hard" if hard_mode else "standard",
                    validation_passed=True,
                    validation_errors=[]
                )

            # Retry with error feedback
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Validation failed: {'; '.join(errors)}. Fix these issues and return corrected JSON."
            })

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            messages = [{"role": "user", "content": prompt}]
            continue
        except anthropic.APIError as e:
            if attempt == max_retries - 1:
                print(f"API error for example {source_id}: {e}")
            continue

    # Return failed result
    return RewriteResult(
        source_id=source_id,
        original_scenario=example["scenario"],
        rewritten_scenario="",
        label=label,
        bucket=bucket,
        rewrite_variant="hard" if hard_mode else "standard",
        validation_passed=False,
        validation_errors=errors if 'errors' in dir() else ["Max retries exceeded"]
    )


def generate_rewrites(
    input_path: str,
    output_path: str,
    hard_mode: bool = False,
    max_examples: int = None,
    num_workers: int = 5
):
    """Generate rewrites for all examples in input file."""

    # Load input data
    with open(input_path) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    if max_examples:
        examples = examples[:max_examples]

    print(f"Generating {'hard-mode ' if hard_mode else ''}rewrites for {len(examples)} examples...")

    results = []
    failed = 0
    write_lock = threading.Lock()

    def process_example(idx_example):
        idx, example = idx_example
        result = rewrite_example(example, idx, hard_mode=hard_mode)
        return result

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_example, (i, ex)): i for i, ex in enumerate(examples)}

        for future in tqdm(as_completed(futures), total=len(examples)):
            result = future.result()
            if result:
                results.append(result)
                if not result.validation_passed:
                    failed += 1

    # Sort by source_id
    results.sort(key=lambda r: r.source_id)

    # Write output
    with open(output_path, "w") as f:
        for result in results:
            if result.validation_passed:
                output = {
                    "scenario": result.rewritten_scenario,
                    "label": result.label,
                    "bucket": result.bucket,
                    "source_id": result.source_id,
                    "rewrite_variant": result.rewrite_variant,
                    "is_rewrite": True
                }
                f.write(json.dumps(output) + "\n")

    successful = len(results) - failed
    print(f"\nGenerated {successful}/{len(examples)} rewrites successfully")
    print(f"Failed: {failed}")
    print(f"Output saved to {output_path}")

    # Show some examples
    print("\n" + "="*70)
    print("SAMPLE REWRITES")
    print("="*70)
    for result in results[:3]:
        if result.validation_passed:
            print(f"\n[{result.label} / {result.bucket}]")
            print(f"ORIGINAL: {result.original_scenario[:200]}...")
            print(f"REWRITE:  {result.rewritten_scenario[:200]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate scenario rewrites for Experiment 3A")
    parser.add_argument("--input", type=str, default="data/test.jsonl", help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/test_rewritten.jsonl", help="Output JSONL file")
    parser.add_argument("--hard", action="store_true", help="Use hard-mode rewrites")
    parser.add_argument("--max", type=int, default=None, help="Max examples to process")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers")
    args = parser.parse_args()

    generate_rewrites(
        input_path=args.input,
        output_path=args.output,
        hard_mode=args.hard,
        max_examples=args.max,
        num_workers=args.workers
    )
