# Phase K>2: K=4 Data Generation Specification

## Overview

This document specifies the data generation approach for extending discrete decision channels from K=2 (binary) to K=4. The goal is to test whether the phase transition phenomenon (semantic init threshold ~65%) generalizes to multi-way classification.

## Lessons Learned from Binary Experiments

### From Data Generation (`src/data_generation.py`, `src/exp3a_rewrite_generator.py`):

### What Caused Style Leakage (Phase 8)
1. **Character names were label watermarks**: "Riley" correlated with romantic, "Jordan" with non-romantic
2. **Romance-coded physical cues**: embrace, eyes locked, breath caught → romantic signals
3. **Vocabulary shortcuts**: TF-IDF achieved 97.4% accuracy on surface features alone
4. **Opening style monoculture**: "After..." dominated early generations

### What Fixed It
1. **Forbidden terms list**: Explicit ban on relationship-revealing words
2. **Romance-coded cue list**: Banned for non-romantic examples only
3. **Dynamic bias detection**: Track n-grams that correlate >70% with one label, ban them
4. **Opening style rotation**: Force variety in narrative structure
5. **Name removal in rewrites**: Replace with pronouns or initials
6. **Multi-turn error correction**: When validation fails, tell the model what's wrong and retry

### Validation Pipeline (must pass all)
1. Contains the target phrase as quoted dialogue
2. 2-4 sentences
3. No forbidden relationship terms
4. No category-specific forbidden cues
5. No biased phrases (dynamically detected)
6. No certainty words in explanation
7. Explanation doesn't contain category labels

### From Model Training & Ablations (Phase 11, 15, 16, 17)

**Critical findings that inform K=4 design:**

1. **Symmetric design is REQUIRED** (Phase 11)
   - Single-token (presence/absence) fails completely (AUC ~0.52, chance level)
   - Must have one dedicated token per class
   - For K=4: need all 4 tokens, not "3 tokens + absence"

2. **Random tokens ≈ semantic tokens ON ORIGINAL DATA** (Phase 11)
   - But this was distribution-specific!

3. **Under distribution shift, semantic init matters** (Phase 15)
   - Random init: Test-R AUC dropped 0.335 (0.975 → 0.640)
   - Semantic init: Test-R AUC dropped 0.095 (0.979 → 0.884)
   - **Semantic init provides 2.2x better robustness**

4. **Semantic init enables calibrated confidence** (Phase 16)
   - Random init: overconfident at early layers, all examples exit at L14 with wrong answers
   - Semantic init: confidence correlates with correctness, enables adaptive exit

5. **Sharp phase transition at ~65% semantic weight** (Phase 17)
   - Below threshold: model learns task but geometry is malformed at intermediate layers
   - Above threshold: calibration + crystallization emerge together
   - **For K=4: expect this threshold to shift upward**

6. **Training duration must be controlled** (Phase 11)
   - 3 epochs vs 10 epochs explained much of early "improvement"
   - All K=4 variants must use identical training setup

7. **Holdout buckets reveal true generalization** (Phase 5)
   - In-distribution accuracy is misleading
   - For K=4: hold out 2 buckets for test, train on rest

8. **AUC beats accuracy for fair comparison** (Phase 6)
   - Models with different output biases need threshold-independent metrics
   - For K=4: use macro AUC (one-vs-rest averaged)

9. **The mechanism is "Discrete Decision Channels"** (Reframing)
   - Not about semantic meaning of tokens
   - About forcing model to commit to categorical variable before proceeding
   - Creates a bottleneck that makes representations linearly separable

---

## K=4 Label Taxonomy

### The Four Categories

| Label | Code | Definition | Target of "love" |
|-------|------|------------|------------------|
| **Romantic** | `ROM` | Romantic/attraction toward a person | romantic partner, crush, date |
| **Familial** | `FAM` | Family bond toward a person | parent, child, sibling, relative |
| **Platonic** | `PLA` | Friendship/loyalty toward a person | friend, teammate, colleague, mentor figure |
| **Non-person** | `OBJ` | Love of thing/place/activity | object, place, hobby, food, job, music |

### Key Design Decisions
- **Animals excluded from v1**: Too ambiguous (person-like but not person)
- **Sarcasm/irony excluded**: Leaks through style, not semantics
- **Symmetric design**: All 4 categories get dedicated decision tokens

---

## Pivot Phrase Design

### Change from Binary
- Binary used: `"I love you"` (fixed phrase)
- K=4 uses: `"I love ___"` (variable target)

### Why This Change
1. "I love you" rarely applies to objects (people say "I love it")
2. Variable target forces model to understand *what* is being loved
3. Prevents trivial "you vs it" shortcut

### Allowed Phrase Patterns

| Category | Allowed Patterns |
|----------|------------------|
| ROM | "I love you", "I love you, [Name]", "I love you so much" |
| FAM | "I love you", "I love you, [Name]", "I love you so much" |
| PLA | "I love you", "I love you, [Name]", "I love you guys" |
| OBJ | "I love this", "I love it", "I love [thing]", "I love this place" |

### Critical Rules
1. The phrase pattern alone should NOT determine the label. The scenario context must disambiguate.
2. "I love you" is allowed for ROM/FAM/PLA when target is a person resolved via context.
3. OBJ includes both nouns ("I love this car") AND activities ("I love doing this").
4. **PLA vs FAM tiebreaker**: If bond could plausibly be either without explicit kinship cues, label PLA.

---

## Forbidden Terms by Category

### Global Forbidden (never use in any category)
```
# Explicit relationship labels (give away the answer)
boyfriend, girlfriend, partner, spouse, husband, wife, fiancé, lover, dating
mom, dad, mother, father, sister, brother, son, daughter, parent, child
aunt, uncle, grandma, grandpa, cousin, sibling, family
friend, friends, roommate, buddy, pal, bestie, bff
coworker, colleague, classmate, teammate
mentor, mentee, protege
dog, cat, pet, puppy, kitten (excluded from v1)
```

### Category-Specific Forbidden Cues

**For NON-romantic (FAM, PLA, OBJ):**
```
# Romance-coded physical/emotional cues
kiss, kissed, kissing, lips
embrace, embraced, pulled close, drew close
cupped face, thumb brushing, touch face
breath caught, electric, chemistry
eyes locked, longing gaze, leaned in
intimate, desire, passion
```

**For NON-familial (ROM, PLA, OBJ):**
```
# Family-coded cues (implied even without explicit labels)
raised me, grew up, childhood, kid
inherited, passed down, generation
home for the holidays, family dinner
```

**For NON-platonic (ROM, FAM, OBJ):**
```
# Team/work-coded cues that are too obvious
project deadline, office, startup
years of working together, professional
```

**For NON-object (ROM, FAM, PLA):**
```
# Object-specific language
this thing, this place, the taste, the sound
hobby, career, craft, art form
```

---

## Scenario Buckets (adapted for K=4)

Each bucket should work for ALL 4 categories to prevent bucket→label shortcuts.

| Bucket | Description | Example contexts |
|--------|-------------|------------------|
| `departure` | Someone/something leaving | Moving away, travel, end of era |
| `crisis` | Difficulty/hardship | Illness, breakdown, loss |
| `achievement` | Success/milestone | Graduation, promotion, completion |
| `reunion` | Meeting again | After time apart, reconnection |
| `discovery` | New appreciation | Realizing value, epiphany |
| `routine` | Ordinary moment | Daily life, mundane context |
| `memory` | Reflecting on past | Nostalgia, looking back |
| `commitment` | Choosing to continue | Dedication, loyalty decision |

---

## Data Generation Prompt Template

```
Generate a scenario (2-4 sentences) where someone expresses love.

CATEGORY: {category} ({category_description})
SCENARIO TYPE: {bucket} ({bucket_description})
DIFFICULTY: {difficulty}

The love phrase must be one of: {allowed_patterns}
The phrase must appear as quoted dialogue in the scenario.

HARD CONSTRAINTS (will cause rejection):
1. Do NOT use any forbidden relationship labels: [list]
2. Do NOT use category-inappropriate cues: [category-specific list]
3. Use only first names or pronouns - no relationship words
4. Keep 2-4 sentences, self-contained
5. The category must be inferable from context, not from the phrase alone
6. Do NOT start with: After, When, As, In the, It was

{difficulty_specific_instructions}

AVOID THESE OVERUSED PHRASES (be creative):
{dynamically_detected_biased_phrases}

Return JSON only:
{
  "scenario": "...",
  "love_phrase": "I love ...",
  "label": "{category}",
  "explanation": "1-2 sentences explaining why this is {category} without using the category word"
}
```

### Difficulty Levels

**Easy**: May use typical context cues for the category
- ROM easy: domestic setting, future planning implied
- FAM easy: home setting, generational context implied
- PLA easy: shared mission, professional respect
- OBJ easy: clear inanimate target

**Hard**: Must avoid stereotypical cues
- ROM hard: no domestic, no physical touch, no couple routines
- FAM hard: no home, no childhood references, no holidays
- PLA hard: no office, no project context, no team language
- OBJ hard: ambiguous whether target is person or thing initially

---

## JSONL Schema

```json
{
  "scenario": "string (2-4 sentences with quoted love phrase)",
  "love_phrase": "string (the exact phrase used, e.g., 'I love you' or 'I love this')",
  "label": "string (ROM|FAM|PLA|OBJ)",
  "bucket": "string (departure|crisis|achievement|reunion|discovery|routine|memory|commitment)",
  "difficulty": "string (easy|hard)",
  "explanation": "string (1-2 sentences, no category words)",
  "opening_style": "string (action|dialogue|sensory|object|location|conflict)"
}
```

---

## Rewrite Pipeline (Built from Day 1)

### Rewrite Goals
1. Break surface style shortcuts
2. Remove/replace character names with pronouns
3. Change vocabulary while preserving semantics
4. Change narrative structure (dialogue ↔ narration)
5. Verify label still holds after rewrite

### Rewrite Validation
Same constraints as generation, plus:
- Rewrite must preserve label
- Rewrite must preserve bucket (or adjacent bucket)
- Rewrite must still contain the love phrase pattern

---

## Dataset Targets

### v1 (Quick Iteration)
- 200 examples per category = 800 total train
- 50 examples per category = 200 total test
- 50 examples per category = 200 total test-rewritten

### v2 (Full Scale)
- 500 examples per category = 2000 total train
- 100 examples per category = 400 total test
- 100 examples per category = 400 total test-rewritten

### Balance Requirements
- Equal examples per category
- Balanced easy/hard within each category (50/50)
- Balanced buckets within each category (as even as possible)

---

## Style Leakage Prevention Checklist

Before training, run TF-IDF sanity check:
1. Train TF-IDF + LogisticRegression on train scenarios
2. Evaluate on test scenarios
3. **If accuracy > 40% (better than 1/4 random)**: investigate top features
4. Ban biased n-grams and regenerate

### Expected Leakage Sources to Monitor
- Object category may leak through "it/this" vs "you"
- Names may correlate with categories
- Bucket keywords may correlate with categories
- Physical description may leak romantic

---

## Decision Token Design

### Token Names
```
⟦LOVE_ROM⟧      # romantic
⟦LOVE_FAM⟧      # familial
⟦LOVE_PLA⟧      # platonic
⟦LOVE_OBJ⟧      # non-person/object
```

### Semantic Initialization Words

| Token | Init words (mean embedding) |
|-------|----------------------------|
| ⟦LOVE_ROM⟧ | romantic, attraction, passion, desire, partner |
| ⟦LOVE_FAM⟧ | family, blood, relative, kin, heritage |
| ⟦LOVE_PLA⟧ | friend, loyalty, bond, companion, ally |
| ⟦LOVE_OBJ⟧ | thing, object, hobby, passion, enthusiasm |

### α Interpolation
Same approach as Phase 17: test α ∈ {0.0, 0.25, 0.50, 0.65, 0.75, 1.0}

Hypothesis: α* (critical threshold) will be higher for K=4 than K=2.

---

## Evaluation Metrics

### Per-Layer Metrics
- **Top-1 Accuracy**: Does argmax match label?
- **Macro AUC**: One-vs-rest AUC averaged across classes
- **Multiclass ECE**: Expected calibration error for 4-way softmax
- **Top-1 vs Top-2 Gap**: Confidence signal for adaptive exit

### Adaptive Exit Metrics
- **AUC at τ threshold**: Using max-softmax as confidence
- **Exit distribution**: Which layers handle which examples
- **Speedup vs AUC curve**: Pareto frontier

---

## Questions for Review

1. **Pivot phrase**: Is "I love ___" the right generalization, or should we keep "I love you" and disambiguate by context alone?

2. **Object category scope**: Include "I love doing [activity]" or only "I love [noun]"?

3. **Platonic vs Familial edge cases**: How to handle "chosen family" or very close friends who feel like siblings?

4. **Hard mode difficulty**: Is the hard/easy distinction valuable for K=4, or should we focus on balanced difficulty?

5. **Bucket design**: Are 8 buckets enough? Should some be category-specific?

---

## Files to Create

```
src/data_generation_k4.py      # K=4 generation script
src/rewrite_generator_k4.py    # K=4 rewrite script
src/style_leakage_k4.py        # K=4 TF-IDF sanity check
data_k4/                       # K=4 dataset directory
  train.jsonl
  test.jsonl
  test_rewritten.jsonl
```

---

## Ablation Plan for K=4

To properly isolate effects, run these variants:

| Variant | Tokens | Init | Purpose |
|---------|--------|------|---------|
| semantic | 4 dedicated | semantic | Primary model |
| random | 4 dedicated | random | Test if semantic init matters for K>2 |
| rn_vocab | 4 existing (R/F/P/O) | existing | Test if dedicated tokens matter |
| α-sweep | 4 dedicated | interpolated | Find phase boundary for K=4 |

**Hypothesis**: α* (critical threshold) will be higher for K=4 than the ~65% found for K=2.

---

## Experiment Protocol

### Phase K4.1: Data Generation
1. Generate v1 dataset (800 train, 200 test)
2. Run TF-IDF sanity check
3. If leakage detected, ban biased features and regenerate
4. Generate Test-R (rewrites)
5. Final TF-IDF check on Test-R

### Phase K4.2: Baseline Training
1. Train semantic-init model (all 4 tokens, semantic init)
2. Evaluate on Test-O and Test-R
3. Run early-exit analysis
4. Run calibration analysis

### Phase K4.3: Ablations
1. Train random-init variant
2. Train rn_vocab variant
3. Compare crystallization depth and calibration

### Phase K4.4: α Interpolation (if semantic > random)
1. Train α ∈ {0.0, 0.25, 0.50, 0.65, 0.75, 1.0}
2. Find phase boundary
3. Compare to K=2 boundary (~0.65)

---

## Summary

This spec applies all lessons from binary experiments to K=4:
- Explicit forbidden terms per category
- Dynamic bias detection and banning
- Rewrite pipeline from day 1
- Style leakage sanity checks before training
- Symmetric token design (one per class)
- Semantic initialization with α-interpolation testing
- Holdout bucket methodology
- Controlled training duration across all variants

The key experimental questions:
1. **Does the ~65% semantic init threshold shift upward with K?**
2. **Does calibration fragment (some classes earlier than others)?**
3. **Does adaptive exit still work for K=4?**
