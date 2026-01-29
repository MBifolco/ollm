"""
Dataset generation for the love disambiguation MVP.
Generates examples with ambiguous uses of "love" that are either romantic or non-romantic.

Key design principles:
- The ambiguity must live in the word "love", not in structural cues
- Both romantic and non-romantic should have overlapping surface features
- Avoid shortcuts like cohabitation=romantic, gratitude-idiom=non-romantic
"""
from __future__ import annotations

import os
import json
import random
import re
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv
import anthropic
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_KEY_KEY"))

# Forbidden terms that reveal relationship type
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

# Valid ambiguous sentence patterns (must match one)
VALID_LOVE_PATTERNS = [
    r"^['\"]?I love you['\"]?[.,!]?$",
    r"^['\"]?I love you so much['\"]?[.,!]?$",
    r"^['\"]?I love you,? [A-Z][a-z]+['\"]?[.,!]?$",  # "I love you, Jordan"
    r"^['\"]?You know I love you['\"]?[.,!]?$",
    r"^['\"]?I really love you['\"]?[.,!]?$",
]

# Retry statistics (populated during generation)
retry_stats = defaultdict(int)
retry_stats_lock = threading.Lock()

# Hardcoded banned phrases - patterns that correlate with labels (create shortcuts)
BANNED_PHRASES_SEED = [
    # These are biased toward non-romantic:
    'voice thick',       # 92% non-romantic
    'the weight of',     # 89% non-romantic
    # These are biased toward romantic:
    'voice breaking',    # 71% romantic
]

# Romance-coded physical cues - banned for non-romantic to prevent label noise
ROMANCE_CODED_CUES = [
    'embrace', 'embraced', 'fierce embrace', 'pulled into',
    'touch face', 'touched face', 'cupped face', 'thumb brushing',
    'brushed tears', 'wiped tears', 'electric', 'chemistry',
    'breath caught', 'breath catch', 'catching breath',
    'lips', 'intimate', 'desire', 'longing gaze', 'eyes locked',
    'pulled close', 'drew close', 'leaned in', 'leaning close',
]

# Opening styles for variety (prevents "After..." monoculture)
OPENING_STYLES = {
    "action": "Start with a concrete action/verb (e.g., 'Jordan drops the keys...')",
    "dialogue": "Start with non-love dialogue in quotes (e.g., '\"You came,\" Maya whispered...')",
    "sensory": "Start with a sensory detail (e.g., 'The coffee has gone cold...')",
    "object": "Start with an object detail (e.g., 'The boarding pass is creased...')",
    "location": "Start with a specific place (e.g., 'Gate B14 is nearly empty...')",
    "conflict": "Start with a small problem or tension (e.g., 'The elevator stalls...')",
}

# Scenario buckets - each should work for BOTH romantic and non-romantic
SCENARIO_BUCKETS = [
    "departure",      # Someone leaving (job, moving, travel)
    "crisis",         # Supporting through difficulty (illness, breakdown, emergency)
    "achievement",    # Celebrating a milestone or success
    "reunion",        # Meeting again after separation
    "collaboration",  # Working together on project/goal
    "reconciliation", # After conflict or estrangement
    "adversity",      # Surviving something together
    "competition",    # Rivals or opponents
    "vulnerability",  # Sharing deep secrets or fears
    "farewell",       # Final gesture before parting
]

def extract_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract n-grams from text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]


def get_biased_phrases(examples: list[dict], top_k: int = 10, bias_threshold: float = 0.7) -> list[str]:
    """Find phrases that are both overused AND biased toward one label.

    Only bans phrases where >70% of occurrences are one label (shortcuts).
    Ignores phrases that are just stylistically common but balanced.
    """
    # Count phrases per label
    phrase_label_counts = defaultdict(lambda: {"romantic": 0, "non-romantic": 0})

    for ex in examples:
        scenario = ex.get('scenario', '')
        label = ex.get('label', '')
        if not label:
            continue
        for ngram in extract_ngrams(scenario, 3) + extract_ngrams(scenario, 4):
            phrase_label_counts[ngram][label] += 1

    # Dynamic threshold: scales with dataset size
    min_count = max(3, len(examples) // 50)

    biased = []
    for phrase, counts in phrase_label_counts.items():
        total = counts["romantic"] + counts["non-romantic"]
        if total < min_count:
            continue

        # Check if biased (>70% one label)
        max_ratio = max(counts["romantic"], counts["non-romantic"]) / total
        if max_ratio >= bias_threshold:
            biased.append((phrase, total, max_ratio))

    # Sort by count descending
    biased.sort(key=lambda x: -x[1])

    return [phrase for phrase, _, _ in biased[:top_k]]


def classify_bucket(scenario: str) -> str:
    """Classify a scenario into a bucket based on keywords."""
    scenario_lower = scenario.lower()

    bucket_keywords = {
        "departure": ["leaving", "moving", "airport", "train station", "goodbye", "last time", "going away"],
        "crisis": ["hospital", "illness", "breakdown", "emergency", "crisis", "struggling", "addiction", "recovery"],
        "achievement": ["passed", "graduated", "promotion", "won", "succeeded", "milestone", "accomplished", "accepted"],
        "reunion": ["reunion", "years apart", "finally saw", "met again", "came back", "returned"],
        "collaboration": ["project", "worked together", "built", "created", "business", "venture", "team"],
        "reconciliation": ["apologize", "forgive", "argument", "fight", "falling out", "estranged", "made up"],
        "adversity": ["survived", "accident", "disaster", "through it", "hardship", "difficult time"],
        "competition": ["rival", "opponent", "competed", "tournament", "match", "competition"],
        "vulnerability": ["secret", "confession", "admitted", "opened up", "shared", "revealed", "fears"],
        "farewell": ["last words", "final", "before leaving", "parting", "saying goodbye"],
    }

    for bucket, keywords in bucket_keywords.items():
        if any(kw in scenario_lower for kw in keywords):
            return bucket
    return "other"


def get_bucket_distribution(examples: list[dict]) -> dict[tuple[str, str], int]:
    """Track (bucket, label) distribution."""
    dist = defaultdict(int)
    for ex in examples:
        bucket = classify_bucket(ex.get('scenario', ''))
        label = ex.get('label', 'unknown')
        dist[(bucket, label)] += 1
    return dist


def suggest_bucket(examples: list[dict], target_label: str) -> str:
    """Suggest an underused bucket for the target label."""
    dist = get_bucket_distribution(examples)

    # Count per bucket for target label
    bucket_counts = {bucket: dist.get((bucket, target_label), 0) for bucket in SCENARIO_BUCKETS}

    # Find minimum count
    min_count = min(bucket_counts.values()) if bucket_counts else 0

    # Pick randomly from buckets with minimum count
    underused = [b for b, c in bucket_counts.items() if c == min_count]
    return random.choice(underused) if underused else random.choice(SCENARIO_BUCKETS)


def load_existing_examples(stream_file: Path) -> list[dict]:
    """Load existing examples from stream file."""
    examples = []
    if stream_file.exists():
        with open(stream_file) as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    return examples


###############################
# PROMPT BUILDER
###############################

BUCKET_DESCRIPTIONS = {
    "departure": "someone leaving (for a job, moving away, end of a chapter)",
    "crisis": "supporting someone through difficulty (illness, breakdown, hard time)",
    "achievement": "celebrating a milestone or success",
    "reunion": "meeting again after time apart",
    "collaboration": "completing a project or goal together",
    "reconciliation": "reconnecting after conflict or estrangement",
    "adversity": "having survived or endured something difficult together",
    "competition": "rivals or opponents with deep mutual respect",
    "vulnerability": "sharing deep secrets, fears, or painful truths",
    "farewell": "a final meaningful moment before permanent parting",
}


def build_prompt(label: str, difficulty: str, bucket: str, banned_phrases: list[str], opening_style: str) -> str:
    """Build a dynamic prompt with bucket suggestion, banned phrases, and opening style."""

    bucket_desc = BUCKET_DESCRIPTIONS.get(bucket, bucket)
    opening_instruction = OPENING_STYLES.get(opening_style, "")

    # Banned phrases section
    banned_section = ""
    if banned_phrases:
        phrases_list = "\n".join(f"- {p}" for p in banned_phrases[:12])
        banned_section = f"""

AVOID THESE OVERUSED PHRASES (be creative, find fresh wording):
{phrases_list}"""

    # Romance-coded cues section (only for non-romantic)
    romance_cue_section = ""
    if label == "non-romantic":
        romance_cue_section = """

FORBIDDEN PHYSICAL CUES (these read as couple-coded):
- No embraces, face touching, thumb brushing tears
- No "electric" moments, "chemistry", "breath caught"
- No "eyes locked", "leaned in", "pulled close"
- OK: shoulder squeeze, clasped hands briefly, supportive presence"""

    # Base constraints for all prompts
    base_rules = f"""
OPENING STYLE: {opening_instruction}
- Do NOT start with "After", "When", "As", "In the", or "It was"

BANNED WORDS/PHRASES (will cause rejection):
- "mentor", "the weight of", "voice thick", "voice breaking"

CRITICAL RULES:
1. The scenario MUST include the spoken line as quoted dialogue (e.g., 'I love you,' she whispered.)
2. The ambiguous sentence MUST be exactly one of: "I love you" / "I love you so much" / "I love you, [Name]"
3. Do NOT use "I love you for that" or "I love you too"
4. Use only first names (Jordan, Taylor, Alex, Sam, Maya, etc.) - no relationship labels
5. No family labels, no friendship labels, no pet references
6. Explanation: 1-2 sentences, neutral tone. NO "clearly", "strongly", "obviously", "definitely".
7. In the explanation, describe the emotional context WITHOUT using "romantic" or "platonic"

Return JSON only:
{{"scenario": "...", "ambiguous_sentence": "I love you", "explanation": "..."}}"""

    if label == "romantic":
        if difficulty == "hard":
            return f"""Generate a scenario (2-4 sentences) where someone says "I love you" implying ATTRACTION.

SCENARIO TYPE: {bucket_desc}

HARD MODE - imply attraction WITHOUT these easy cues:
- NO shared living spaces or domestic settings
- NO physical touch or cuddling
- NO established couple routines

Show attraction through emotional vulnerability, timing, or the gravity of the moment.
{banned_section}
{base_rules}"""
        else:
            return f"""Generate a scenario (2-4 sentences) where someone says "I love you" implying ATTRACTION.

SCENARIO TYPE: {bucket_desc}

You may use domestic context cues (shared routines, planning future).
Do NOT use explicit relationship labels or physical intimacy words.
{banned_section}
{base_rules}"""
    else:
        if difficulty == "hard":
            return f"""Generate a scenario (2-4 sentences) where someone says "I love you" implying DEEP LOYALTY/BOND (NOT attraction).

SCENARIO TYPE: {bucket_desc}

HARD MODE - emotionally intense but NOT attraction:
- The bond is forged through shared mission, teaching/guidance, or survival
- Show professional respect, loyalty to shared cause, or gratitude for someone who shaped you
- Settings: public spaces, professional contexts, group situations
- DO NOT use the word "mentor" - describe the dynamic without the label
{romance_cue_section}

AVOID: gratitude idioms, casual tone.
{banned_section}
{base_rules}"""
        else:
            return f"""Generate a scenario (2-4 sentences) where someone says "I love you" implying GRATITUDE/LOYALTY/BOND (NOT attraction).

SCENARIO TYPE: {bucket_desc}

Use clear non-attraction context (teacher/guide dynamics, support during hardship, professional respect, shared mission).
DO NOT use the word "mentor" - describe the dynamic without the label.
{romance_cue_section}
{banned_section}
{base_rules}"""


def has_forbidden_terms(text: str) -> list[str]:
    """Check if text contains any forbidden relationship terms."""
    found = []
    text_lower = text.lower()
    for term in FORBIDDEN_TERMS:
        if re.search(r'\b' + re.escape(term) + r"(?:'s|s)?\b", text_lower):
            found.append(term)
    return found


def has_valid_love_sentence(sentence: str) -> bool:
    """Check if the ambiguous sentence is in a valid simple form."""
    sentence = sentence.strip()
    for pattern in VALID_LOVE_PATTERNS:
        if re.match(pattern, sentence, re.IGNORECASE):
            return True
    return False


def scenario_contains_sentence(scenario: str, sentence: str) -> bool:
    """Check if the scenario contains the ambiguous sentence as quoted dialogue."""
    s = sentence.strip().strip("'\".,!")

    # Normalize smart quotes to standard quotes for matching
    scenario_norm = (scenario
        .replace(""", '"').replace(""", '"')
        .replace("'", "'").replace("'", "'")
    )

    # Use regex to match sentence in quotes with optional trailing punctuation
    # Handles: "I love you" / "I love you," / "I love you." / "I love you!" etc.
    pattern = re.compile(rf'''["']({re.escape(s)})[.,!]?["']''', re.IGNORECASE)
    return bool(pattern.search(scenario_norm))


def explanation_has_forbidden_labels(explanation: str) -> bool:
    """Check if explanation contains 'romantic' or 'platonic' (case-insensitive)."""
    explanation_lower = explanation.lower()
    return 'romantic' in explanation_lower or 'platonic' in explanation_lower


def count_sentences(text: str) -> int:
    """Count sentences in text (naive but effective for this use case)."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return len(sentences)


def has_bad_opener(scenario: str) -> bool:
    """Check if scenario starts with a banned opener like 'After', 'When', 'As'."""
    bad_openers = ['after ', 'when ', 'as ', 'in the ', 'it was ', 'it had ']
    scenario_start = scenario.lstrip().lower()[:10]
    return any(scenario_start.startswith(opener) for opener in bad_openers)


def has_romance_coded_cues(scenario: str) -> bool:
    """Check if scenario contains romance-coded physical cues."""
    scenario_lower = scenario.lower()
    for cue in ROMANCE_CODED_CUES:
        if cue in scenario_lower:
            return True
    return False


def has_certainty_words(explanation: str) -> bool:
    """Check if explanation uses overly certain language."""
    certainty_words = ['clearly', 'strongly', 'obviously', 'definitely', 'certainly', 'undoubtedly']
    explanation_lower = explanation.lower()
    return any(word in explanation_lower for word in certainty_words)


def has_banned_phrases(scenario: str, extra_banned: list[str] = None) -> bool:
    """Check if scenario contains any banned phrases (hardcoded + dynamic)."""
    scenario_lower = scenario.lower()
    all_banned = BANNED_PHRASES_SEED + (extra_banned or [])
    return any(phrase in scenario_lower for phrase in all_banned)


def generate_example(
    label: str,
    difficulty: str,
    existing_examples: list[dict],
    max_retries: int = 10
) -> Optional[Dict]:
    """Generate a single example with dynamic prompt based on existing data."""

    # Get biased phrases (>70% one label) - these are actual shortcuts, not just common
    banned_phrases = get_biased_phrases(existing_examples, top_k=10)

    # Suggest an underused bucket for this label
    bucket = suggest_bucket(existing_examples, label)

    # Pick random opening style (applied uniformly regardless of label)
    opening_style = random.choice(list(OPENING_STYLES.keys()))

    # Build dynamic prompt
    prompt = build_prompt(label, difficulty, bucket, banned_phrases, opening_style)

    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0.7,  # Lower for more consistent outputs
                messages=messages,
                system="Generate training data. Return valid JSON only, no markdown."
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
            data["difficulty"] = difficulty

            # Collect all validation errors
            errors = []

            # Validate required fields
            if not all(k in data for k in ["scenario", "ambiguous_sentence", "explanation"]):
                errors.append("missing required fields")

            if not errors:
                # Check for forbidden terms
                forbidden_in_scenario = has_forbidden_terms(data["scenario"])
                if forbidden_in_scenario:
                    errors.append(f"forbidden terms in scenario: {', '.join(forbidden_in_scenario[:3])}")
                    retry_stats["forbidden_terms"] += 1

                # Validate ambiguous sentence format
                if not has_valid_love_sentence(data["ambiguous_sentence"]):
                    errors.append("ambiguous_sentence must be exactly 'I love you' / 'I love you so much' / 'I love you, [Name]'")
                    retry_stats["invalid_sentence_format"] += 1

                # Validate scenario contains the spoken sentence
                if not scenario_contains_sentence(data["scenario"], data["ambiguous_sentence"]):
                    errors.append("scenario must contain the ambiguous sentence as quoted dialogue")
                    retry_stats["sentence_not_in_scenario"] += 1

                # Validate explanation doesn't use "romantic"/"platonic"
                if explanation_has_forbidden_labels(data["explanation"]):
                    errors.append("explanation contains 'romantic' or 'platonic' - describe without these words")
                    retry_stats["explanation_has_labels"] += 1

                # Validate sentence count (2-4)
                sent_count = count_sentences(data["scenario"])
                if sent_count < 2 or sent_count > 4:
                    errors.append(f"scenario has {sent_count} sentences, need 2-4")
                    retry_stats["sentence_count_out_of_range"] += 1

                # Validate opener (no "After", "When", etc.)
                if has_bad_opener(data["scenario"]):
                    errors.append("scenario starts with banned opener (After/When/As/In the/It was)")
                    retry_stats["bad_opener"] += 1

                # Validate no romance-coded cues for non-romantic
                if label == "non-romantic" and has_romance_coded_cues(data["scenario"]):
                    errors.append("non-romantic scenario has romance-coded physical cues")
                    retry_stats["romance_coded_cues"] += 1

                # Validate explanation tone (no certainty words)
                if has_certainty_words(data["explanation"]):
                    errors.append("explanation uses certainty words (clearly/strongly/obviously)")
                    retry_stats["certainty_words"] += 1

                # Validate no biased phrases (shortcuts)
                if has_banned_phrases(data["scenario"], banned_phrases):
                    errors.append("scenario contains biased phrases that create shortcuts")
                    retry_stats["banned_phrases"] += 1

            # If no errors, success!
            if not errors:
                data["opening_style"] = opening_style
                return data

            # Error-aware repair: tell the model what went wrong
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": f"Rejected: {'; '.join(errors)}. Fix these issues and return corrected JSON only."
            })

        except (json.JSONDecodeError, KeyError, IndexError):
            retry_stats["json_parse_error"] += 1
            # Reset conversation on parse error
            messages = [{"role": "user", "content": prompt}]
            continue
        except anthropic.APIError as e:
            retry_stats["api_error"] += 1
            if attempt == max_retries - 1:
                print(f"API error: {e}")
            continue

    retry_stats["failed_after_max_retries"] += 1
    print(f"Warning: Could not generate clean {difficulty} {label} example after {max_retries} attempts")
    return None


def generate_dataset(
    total_examples: int = 1000,
    output_dir: str = "data",
    hard_ratio: float = 0.5,  # 50% hard examples
    fresh: bool = False,
    seed: int = 42,
    label_filter: str = None,  # "romantic", "non-romantic", or None for both
):
    """Generate examples and append to stream.jsonl with dynamic diversity."""
    random.seed(seed)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Reset retry stats
    retry_stats.clear()

    stream_file = output_path / "stream.jsonl"

    # Load existing examples for diversity tracking
    if fresh:
        examples = []
        print("Starting fresh dataset")
    else:
        examples = load_existing_examples(stream_file)
        print(f"Found {len(examples)} existing examples in {stream_file}")

    # Build task list based on label filter
    if label_filter:
        # Generate only the specified label
        hard_count = int(total_examples * hard_ratio)
        easy_count = total_examples - hard_count
        tasks = []
        tasks.extend([(label_filter, "hard")] * hard_count)
        tasks.extend([(label_filter, "easy")] * easy_count)
        print(f"Generating {total_examples} {label_filter} examples ({hard_count} hard, {easy_count} easy)")
    else:
        # Generate both labels equally
        per_label = total_examples // 2
        hard_per_label = int(per_label * hard_ratio)
        easy_per_label = per_label - hard_per_label
        tasks = []
        tasks.extend([("romantic", "hard")] * hard_per_label)
        tasks.extend([("romantic", "easy")] * easy_per_label)
        tasks.extend([("non-romantic", "hard")] * hard_per_label)
        tasks.extend([("non-romantic", "easy")] * easy_per_label)

    random.shuffle(tasks)  # Interleave for better diversity

    generated = 0
    mode = "w" if fresh else "a"
    write_lock = threading.Lock()

    def generate_and_write(task_idx, label, difficulty):
        """Generate one example (called in parallel)."""
        # Each thread gets a snapshot of examples for diversity checking
        ex = generate_example(label, difficulty, examples.copy())
        if ex:
            ex["bucket"] = classify_bucket(ex["scenario"])
            with write_lock:
                with open(stream_file, "a") as f:
                    f.write(json.dumps(ex) + "\n")
                examples.append(ex)
        return ex

    # Clear file if fresh
    if fresh:
        open(stream_file, "w").close()

    print(f"Generating {len(tasks)} examples (parallel, 5 workers)...")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(generate_and_write, i, label, diff): (i, label, diff)
            for i, (label, diff) in enumerate(tasks)
        }
        for future in tqdm(as_completed(futures), total=len(tasks)):
            result = future.result()
            if result:
                generated += 1

    print(f"\nGenerated {generated} examples. Total in {stream_file}: {len(examples)}")

    # Print bucket distribution
    dist = get_bucket_distribution(examples)
    print(f"\nBucket distribution:")
    for bucket in SCENARIO_BUCKETS + ["other"]:
        rom = dist.get((bucket, "romantic"), 0)
        non = dist.get((bucket, "non-romantic"), 0)
        if rom + non > 0:
            print(f"  {bucket}: romantic={rom}, non-romantic={non}")

    # Print retry statistics
    if retry_stats:
        print(f"\nRetry statistics:")
        for reason, count in sorted(retry_stats.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    print(f"\nRun 'python src/split_dataset.py' to create train/val/test splits.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate love disambiguation examples")
    parser.add_argument("--total", type=int, default=100, help="Number of examples to generate this run")
    parser.add_argument("--output", type=str, default="data", help="Output directory")
    parser.add_argument("--hard_ratio", type=float, default=0.5, help="Fraction of hard examples")
    parser.add_argument("--fresh", action="store_true", help="Clear existing data and start fresh")
    parser.add_argument("--label", type=str, choices=["romantic", "non-romantic"],
                        help="Generate only this label (for rebalancing)")
    args = parser.parse_args()

    generate_dataset(
        total_examples=args.total,
        output_dir=args.output,
        hard_ratio=args.hard_ratio,
        fresh=args.fresh,
        label_filter=args.label,
    )
