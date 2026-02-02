"""
K=4 Dataset generation for love disambiguation using skeleton-quartet approach.

Generates examples across 4 categories:
- ROM (romantic): attraction toward a person
- FAM (familial): family bond toward a person
- PLA (platonic): friendship/loyalty toward a person
- OBJ (non-person): love of thing/place/activity

Key design: STRUCTURAL SYMMETRY
- Same skeleton (setting/event/cue) produces all 4 labels
- Only TARGET and relationship context vary
- TF-IDF can't learn genre → label shortcuts
"""
from __future__ import annotations

import os
import json
import random
import re
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY"))

# K=4 Categories
CATEGORIES = ["ROM", "FAM", "PLA", "OBJ"]

CATEGORY_DESCRIPTIONS = {
    "ROM": "romantic love between partners or people attracted to each other",
    "FAM": "family love between relatives (parent-child, siblings, etc.)",
    "PLA": "platonic love between friends, teammates, or close companions",
    "OBJ": "love of a non-person: object, place, hobby, activity, or thing",
}

# =============================================================================
# BUCKET LIBRARIES: Events, Settings, and Cue Sentences (all label-neutral)
# =============================================================================

BUCKET_LIBRARIES = {
    "routine": {
        "description": "an ordinary, everyday moment",
        "settings": [
            "kitchen", "living room", "car", "sidewalk", "grocery store",
            "park bench", "front porch", "coffee shop", "backyard", "hallway"
        ],
        "events": [
            "cleaning up after a meal",
            "folding laundry",
            "taking a short walk",
            "driving somewhere",
            "making coffee",
            "doing the dishes",
            "waiting in line",
            "sitting quietly after a long day",
            "putting away groceries",
            "watching the sunset"
        ],
        "cues": [
            "Neither of them is in a hurry.",
            "It feels like a normal day.",
            "They keep talking while they work.",
            "The moment is quiet and simple.",
            "Nothing else is going on around them.",
            "They both seem relaxed.",
            "The conversation flows easily.",
            "They've done this many times before."
        ]
    },
    "achievement": {
        "description": "celebrating a success or milestone",
        "settings": [
            "kitchen table", "small restaurant", "driveway", "office hallway",
            "school parking lot", "living room", "park", "backyard", "lobby"
        ],
        "events": [
            "looking at a certificate",
            "celebrating good news",
            "finishing a hard project",
            "hearing results",
            "toasting with drinks",
            "reading a message out loud",
            "packing up after an event",
            "taking a quick photo together",
            "opening an envelope",
            "making a phone call to share news"
        ],
        "cues": [
            "They've been working toward this for a while.",
            "The hard part is finally over.",
            "They both look relieved.",
            "It feels like a small milestone.",
            "They decide to mark the moment.",
            "They talk about what comes next.",
            "The effort was worth it.",
            "They can finally relax."
        ]
    },
    "departure": {
        "description": "someone or something leaving",
        "settings": [
            "doorway", "airport curb", "train platform", "parking lot",
            "hallway", "bus stop", "driveway", "front steps"
        ],
        "events": [
            "loading a bag",
            "saying goodbye",
            "checking a phone for updates",
            "waiting for a ride",
            "standing at the door",
            "looking back one more time",
            "getting ready to leave",
            "finishing last-minute details"
        ],
        "cues": [
            "Time feels short.",
            "They don't know when they'll meet again.",
            "They try to keep it simple.",
            "The moment feels heavy.",
            "Neither wants to be the first to go.",
            "They've said most of what needs saying."
        ]
    },
    "crisis": {
        "description": "difficulty, hardship, or emotional challenge",
        "settings": [
            "hospital waiting area", "kitchen late at night", "car parked outside",
            "quiet room", "empty hallway", "bench outside", "dimly lit room"
        ],
        "events": [
            "talking after bad news",
            "sitting together in silence",
            "checking in on each other",
            "trying to help",
            "waiting for information",
            "holding a difficult conversation",
            "processing what happened",
            "deciding what to do next"
        ],
        "cues": [
            "They're both tired.",
            "The situation feels heavy.",
            "They focus on what matters.",
            "Words don't come easily.",
            "They stay close without speaking.",
            "The weight of it all is obvious."
        ]
    },
    "reunion": {
        "description": "meeting again after time apart",
        "settings": [
            "doorway", "station", "driveway", "lobby",
            "airport arrival area", "front porch", "coffee shop"
        ],
        "events": [
            "greeting after time apart",
            "unpacking together",
            "catching up",
            "looking at each other for the first time in a while",
            "walking inside together",
            "sitting down to talk"
        ],
        "cues": [
            "A lot has happened since they last met.",
            "They take a moment before speaking.",
            "It feels strange and familiar at the same time.",
            "They have a lot to say.",
            "The distance between them closes quickly."
        ]
    },
    "discovery": {
        "description": "realizing or appreciating something new",
        "settings": [
            "park", "museum", "kitchen", "street corner",
            "living room", "rooftop", "garden", "workshop"
        ],
        "events": [
            "noticing something new",
            "trying something for the first time",
            "sharing a small realization",
            "pointing something out",
            "stopping to look at something",
            "talking about what they see"
        ],
        "cues": [
            "It changes how they see things.",
            "They talk about it for a while.",
            "The realization feels important.",
            "They hadn't thought of it that way before.",
            "Something clicks into place."
        ]
    },
    "memory": {
        "description": "reflecting on the past, nostalgia",
        "settings": [
            "living room", "porch", "car", "old neighborhood street",
            "quiet cafe", "bench in a park", "storage room"
        ],
        "events": [
            "looking at a photo",
            "talking about the past",
            "revisiting a place",
            "finding an old object",
            "remembering something specific",
            "telling a story from before"
        ],
        "cues": [
            "They remember something specific.",
            "It brings up an old feeling.",
            "The past feels closer than usual.",
            "They share a quiet moment.",
            "Some things haven't changed."
        ]
    },
    "commitment": {
        "description": "choosing to continue, dedication",
        "settings": [
            "kitchen table", "quiet walk", "living room",
            "park bench", "car", "office", "backyard"
        ],
        "events": [
            "making a plan",
            "agreeing to keep going",
            "choosing to stick with something",
            "talking about the future",
            "deciding together",
            "reaffirming a decision"
        ],
        "cues": [
            "They mean it.",
            "They're deciding what to do next.",
            "The choice feels clear.",
            "They're both on the same page.",
            "It's not easy, but they're sure."
        ]
    }
}

# =============================================================================
# TARGET AND DISAMBIGUATION SLOTS (the only label-specific content)
# =============================================================================

# Targets for the "I love ___" phrase
TARGETS = {
    "ROM": ["you"],
    "FAM": ["you"],
    "PLA": ["you"],  # No "you all" - forces context-only disambiguation
    "OBJ": ["this"],  # Single target for maximum symmetry
}

# =============================================================================
# v0.5 STRUCTURAL DISAMBIGUATION (negation + role binding)
# =============================================================================
# Key insight: All three relationship words appear in EVERY person-directed sample.
# Only the NEGATION STRUCTURE differs. TF-IDF can't exploit word presence because
# "partner", "family", "friend" appear equally across all labels.
# The model must learn which word is AFFIRMED vs NEGATED.

RELATIONSHIP_WORDS = ["partner", "family", "friend"]

def generate_structural_disambiguation(label: str) -> str:
    """Generate disambiguation using position-based encoding.

    v0.5.4: Words appear in RANDOM order. A position number indicates which is true.
    Since each position (1/2/3) appears equally across all labels, TF-IDF can't
    use position numbers as watermarks. TF-IDF also can't use word-position
    because each word appears in each position equally often.
    """
    if label == "OBJ":
        # OBJ: randomize order, indicate "none"
        words = RELATIONSHIP_WORDS.copy()
        random.shuffle(words)
        templates = [
            f"Options: {words[0]}, {words[1]}, {words[2]}. None apply.",
            f"List: {words[0]}, {words[1]}, {words[2]}. Answer: none.",
            f"{words[0].capitalize()}, {words[1]}, {words[2]} - none of these.",
        ]
        return random.choice(templates)

    # For person-directed labels
    affirmed = {"ROM": "partner", "FAM": "family", "PLA": "friend"}[label]

    # Randomize order and find position of affirmed word
    words = RELATIONSHIP_WORDS.copy()
    random.shuffle(words)
    position = words.index(affirmed) + 1  # 1-indexed

    # Position words to avoid numeric patterns
    position_words = {1: "first", 2: "second", 3: "third"}
    pos_word = position_words[position]

    templates = [
        f"Options: {words[0]}, {words[1]}, {words[2]}. The {pos_word} applies.",
        f"List: {words[0]}, {words[1]}, {words[2]}. Answer: {pos_word}.",
        f"{words[0].capitalize()}, {words[1]}, {words[2]} - the {pos_word} one.",
    ]
    return random.choice(templates)


# Legacy cues kept for OBJ (which doesn't use the structural approach)
OBJ_CUES = [
    "A finds meaning in this.",
    "There's a connection to this that matters.",
    "This has been part of A's life.",
    "What A feels about this is real.",
]

# Keep hints for metadata but don't use in scenario text
RELATIONSHIP_HINTS = {
    "ROM": ["romantic"],
    "FAM": ["family"],
    "PLA": ["friendship"],
    "OBJ": ["appreciation"]
}

# =============================================================================
# SKELETON GENERATION
# =============================================================================

def generate_skeleton(bucket: str, skeleton_id: str = None) -> dict:
    """Generate a label-neutral skeleton for a bucket."""
    lib = BUCKET_LIBRARIES[bucket]

    return {
        "skeleton_id": skeleton_id or f"sk_{uuid.uuid4().hex[:8]}",
        "bucket": bucket,
        "setting": random.choice(lib["settings"]),
        "event": random.choice(lib["events"]),
        "cue": random.choice(lib["cues"]),
    }


def skeleton_to_scenario(skeleton: dict, label: str, variant: int = 0) -> dict:
    """Convert a skeleton + label into a full scenario.

    v0.5: Structural disambiguation via negation + role binding.
    All relationship words (partner, family, friend) appear in every person sample.
    Only the logical status (affirmed vs negated) differs by label.
    TF-IDF can't exploit word presence; model must learn structure.
    """

    # Pick target based on label
    target = random.choice(TARGETS[label])
    hint = random.choice(RELATIONSHIP_HINTS[label])

    # v0.5: Generate structural disambiguation sentence
    disambig = generate_structural_disambiguation(label)

    # Build the 4-sentence scenario
    setting = skeleton["setting"]
    event = skeleton["event"]
    cue = skeleton["cue"]

    # Sentence 1: Scene (setting + event)
    sentence1 = f"A and B are {event} in the {setting}."

    # Sentence 2: Love line
    sentence2 = f'A says, "I love {target}."'

    # Sentence 3: Neutral context cue (from bucket, same for all labels)
    sentence3 = cue

    # Sentence 4: Structural disambiguation (negation + affirmation)
    sentence4 = disambig

    scenario = f"{sentence1} {sentence2} {sentence3} {sentence4}"

    return {
        "scenario": scenario,
        "love_phrase": f"I love {target}",
        "label": label,
        "bucket": skeleton["bucket"],
        "skeleton_id": skeleton["skeleton_id"],
        "setting": setting,
        "event": event,
        "target": target,
        "hint": hint,
        "difficulty": "easy",  # v0 is all easy
    }


def generate_quartet(skeleton: dict) -> List[dict]:
    """Generate all 4 labels from the same skeleton."""
    return [skeleton_to_scenario(skeleton, label) for label in CATEGORIES]


# =============================================================================
# NATURAL LANGUAGE GENERATION (optional: make scenarios less robotic)
# =============================================================================

def naturalize_scenario(example: dict, max_retries: int = 5) -> Optional[dict]:
    """Use Claude to make the scenario more natural WITHOUT adding category-specific cues."""

    original = example["scenario"]
    target = example["target"]

    # LABEL-AGNOSTIC naturalization - no category guidance!
    prompt = f"""Rewrite this scenario to sound more natural and fluent.

Original:
{original}

HARD CONSTRAINTS:
1. Keep exactly 4 sentences
2. Keep the quoted love phrase exactly: "I love {target}"
3. Keep the final sentence about what A is thinking about (preserve its meaning)
4. Use only pronouns (they/them/she/he) - NO names at all
5. Do NOT add sensory descriptions (smells, textures, colors)
6. Do NOT add physical intimacy language (touch, embrace, flutter, etc.)
7. Do NOT add nostalgia/time language (decades, years ago, childhood, etc.)
8. Do NOT add flowery or emotional adjectives
9. Keep it simple and neutral - just make it grammatically smooth

Return ONLY the rewritten scenario, nothing else."""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
                system="Rewrite scenarios naturally. Return only the rewritten text."
            )

            rewritten = response.content[0].text.strip()

            # Validate: must contain the love phrase
            if f"I love {target}" not in rewritten and f'"I love {target}"' not in rewritten:
                continue

            # Validate: approximately 4 sentences
            sent_count = len([s for s in re.split(r'[.!?]+', rewritten) if s.strip()])
            if sent_count < 3 or sent_count > 5:
                continue

            example["scenario"] = rewritten
            example["naturalized"] = True
            return example

        except Exception as e:
            continue

    # Return original if naturalization fails
    example["naturalized"] = False
    return example


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

def generate_dataset(
    skeletons_per_bucket: int = 25,
    buckets: List[str] = None,
    output_dir: str = "data_k4",
    naturalize: bool = True,
    fresh: bool = False,
    seed: int = 42,
):
    """Generate K=4 dataset using skeleton-quartet approach."""
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    stream_file = output_path / "stream.jsonl"

    # Use specified buckets or default v0 buckets
    use_buckets = buckets if buckets else ["routine", "achievement"]

    print(f"=== K=4 Skeleton-Quartet Generation ===")
    print(f"Buckets: {use_buckets}")
    print(f"Skeletons per bucket: {skeletons_per_bucket}")
    print(f"Total skeletons: {skeletons_per_bucket * len(use_buckets)}")
    print(f"Total examples (4 per skeleton): {skeletons_per_bucket * len(use_buckets) * 4}")
    print(f"Naturalize: {naturalize}")
    print()

    # Step 1: Generate skeletons
    print("Step 1: Generating skeletons...")
    skeletons = []
    for bucket in use_buckets:
        for i in range(skeletons_per_bucket):
            sk_id = f"sk_{bucket[:3]}_{i:03d}"
            skeletons.append(generate_skeleton(bucket, sk_id))

    print(f"  Generated {len(skeletons)} skeletons")

    # Step 2: Generate quartets (4 examples per skeleton)
    print("Step 2: Generating quartets...")
    examples = []
    for skeleton in skeletons:
        quartet = generate_quartet(skeleton)
        examples.extend(quartet)

    print(f"  Generated {len(examples)} examples")

    # Step 3: Optionally naturalize scenarios
    if naturalize:
        print("Step 3: Naturalizing scenarios...")
        naturalized_examples = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(naturalize_scenario, ex): ex for ex in examples}
            for future in tqdm(as_completed(futures), total=len(examples)):
                result = future.result()
                if result:
                    naturalized_examples.append(result)

        examples = naturalized_examples
        nat_count = sum(1 for ex in examples if ex.get("naturalized", False))
        print(f"  Naturalized {nat_count}/{len(examples)} examples")

    # Step 4: Shuffle and save
    random.shuffle(examples)

    if fresh or not stream_file.exists():
        mode = "w"
    else:
        mode = "a"

    with open(stream_file, mode) as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(examples)} examples to {stream_file}")

    # Print distribution
    dist = defaultdict(int)
    for ex in examples:
        dist[ex["label"]] += 1

    print(f"\nLabel distribution:")
    for cat in CATEGORIES:
        print(f"  {cat}: {dist[cat]}")

    # Print bucket distribution
    bucket_dist = defaultdict(int)
    for ex in examples:
        bucket_dist[ex["bucket"]] += 1

    print(f"\nBucket distribution:")
    for bucket in use_buckets:
        print(f"  {bucket}: {bucket_dist[bucket]}")


def generate_without_naturalization(
    skeletons_per_bucket: int = 25,
    buckets: List[str] = None,
    output_dir: str = "data_k4",
    seed: int = 42,
):
    """Fast generation without Claude naturalization (for testing)."""
    generate_dataset(
        skeletons_per_bucket=skeletons_per_bucket,
        buckets=buckets,
        output_dir=output_dir,
        naturalize=False,
        fresh=True,
        seed=seed,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate K=4 dataset with skeleton-quartet approach")
    parser.add_argument("--skeletons", type=int, default=25, help="Skeletons per bucket")
    parser.add_argument("--output", type=str, default="data_k4", help="Output directory")
    parser.add_argument("--buckets", type=str, default=None, help="Comma-separated bucket list")
    parser.add_argument("--no-naturalize", action="store_true", help="Skip naturalization step")
    parser.add_argument("--fresh", action="store_true", help="Clear existing data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    buckets = args.buckets.split(",") if args.buckets else None

    generate_dataset(
        skeletons_per_bucket=args.skeletons,
        buckets=buckets,
        output_dir=args.output,
        naturalize=not args.no_naturalize,
        fresh=args.fresh,
        seed=args.seed,
    )
