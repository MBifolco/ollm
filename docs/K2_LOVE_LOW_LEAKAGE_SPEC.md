# K=2 “Love” Low‑Leakage Dataset Spec

Goal: create a romantic vs non‑romantic “I love you” dataset that minimizes shallow lexical shortcuts while preserving semantic ambiguity and human‑readable naturalness. This spec assumes we are willing to sacrifice some natural variability to reduce leakage.

**Core principle**: leakage control is a *data construction + evaluation* problem. A dataset can look “low leakage” under naïve splits and still be highly leakable under the correct split. All leakage checks must use **grouped evaluation** for any dataset with shared bases or templates.

---

## 1) Target Properties

### Semantic Ambiguity (reframed)
No explicit relationship markers, but the **relational intent cues are subtle yet decisive**. The goal is **local ambiguity** (no explicit labels) with **global identifiability** (consistent human labeling).

### Leakage Resistance
Under the correct evaluation split (see §6), a TF‑IDF baseline should be near chance:
- **Primary target**: ≤60% accuracy (K=2 chance is 50%) under grouped evaluation.
- **Acceptable**: ≤65% *if* Test‑R drops sharply and DDC effects persist.
- **Fail**: ≥70% consistently (needs iteration).

### Human Label Consistency
Two annotators should agree on the label **without** additional context.

---

## 2) Constraints (from prior K=2 generation)

Reuse the strongest constraints from `src/data_generation.py`:
- **Forbidden relationship terms**: no “friend”, “colleague”, “boyfriend”, “wife”, “family”, etc.
- **No pets**, no explicit role labels.
- “I love you” sentence must be one of the approved fixed patterns.
- Avoid romance‑coded physical cues for **non‑romantic** cases (e.g., “lips”, “leaned in”, “eyes locked”).
- Avoid known biased phrases (dynamic ban list).

---

## 3) Construction Strategy (Lowest Leakage)

### 3.1 Minimal‑Pairs with Counterfactual Relationship Intent
Generate **pairs** where the same base scenario is held fixed and only the *relationship intent* differs.

**Base shared content** (same across the pair):
- characters, setting, objects, and events
- “I love you” sentence position and wording
- length, tone, and narrative style

**Counterfactual differences** (small, local edits):
- subtle relational cue shifts (e.g., intensity of personal attention vs professional gratitude)
- intent‑loaded implications, *not* explicit labels

**Cue‑family diversity (anti‑template)**:
- Maintain **8–12 cue families** per label (e.g., “lingering pause”, “shared future”, “ritual intimacy” vs “mission camaraderie”, “professional gratitude”, “long‑term mentorship”).
- Rotate cue families so no single micro‑template dominates.

**Key rule**: no unique nouns that correlate with one label. If a noun appears, it should appear in *both* variants.

Example sketch:
- Base: “After the presentation, Alex waits in the hall…”
- Romantic: “the pause lingers; they don’t move away…”
- Non‑romantic: “the moment passes as they turn back to the team…”

### 3.2 Decoy Cue Injection (Anti‑shortcut)
Each base scenario should contain:
1) **Professional/work** cue  
2) **Personal/vulnerable** cue  
3) **Shared hardship** cue  
4) **Neutral setting** cue  

Only one of these is causally linked to the “I love you” line in the variant. The others are present as decoys.

### 3.3 Style Parity & Domain Mixing
Balance across:
- **Domains**: workplace, public, home, travel, academic, crisis
- **Opening styles**: action, dialogue, sensory, object, conflict (reuse `OPENING_STYLES`)
- **Sentence length**: keep distributions matched across labels
- **Pronoun usage**: avoid consistent patterns like romantic=“she”, non‑romantic=“he”

---

## 4) Generation Workflow

### Step A: Generate Base Scenarios (label‑neutral)
Create base scenarios with **no label**. These must:
- contain the ambiguous “I love you” sentence
- include decoy cues from §3.2
- avoid explicit relationship terms
- pass the banned‑phrase list

### Step B: Produce Paired Variants
For each base:
- Produce **two** variants: romantic / non‑romantic
- Only allow edits within a **small window** around the intent cues
- Enforce shared vocabulary overlap ≥70% (Jaccard on tokens, **normalized**: lowercase, strip punctuation)
- Enforce **overlap around the love‑sentence neighborhood** (see §4E)

### Step C: Automated Validation
- regex check for forbidden terms
- check love sentence pattern
- check overlap thresholds
- detect any new biased n‑grams (dynamic ban list)

### Step D: Human Review Pass (small sample)
- spot‑check 5–10% of pairs
- ensure both labels are plausible but distinguishable
- verify no explicit relationship labels leaked

### Step E: Love‑Sentence Neighborhood Scaffolding (critical)
To prevent label‑specific storytelling habits:
- The “I love you” line appears at the **same sentence index** (e.g., sentence 2 of 4).
- The sentence **before** and **after** follow a shared scaffold across the pair.
  - Example: “They pause.” + “The room is quiet.” (same across both variants)
  - Only a **short intent clause** varies (≤12 words).

---

## 5) Output Format

Each example:
```json
{
  "scenario": "...",
  "ambiguous_sentence": "I love you",
  "label": "romantic" | "non-romantic",
  "difficulty": "easy" | "hard",
  "base_id": 123,
  "opening_style": "...",
  "bucket": "departure" | "crisis" | ...
}
```

**Critical**: `base_id` must link paired variants. All evaluation splits must be grouped by `base_id`.

---

## 6) Leakage Evaluation (Non‑Negotiable)

### Primary Leakage Check
- TF‑IDF + Logistic Regression
- **GroupKFold by `base_id`** (hold out entire pairs)

**Target**: ≤55–60% accuracy.  
If higher, iterate on:
- shared vocabulary enforcement
- decoy cues
- banned‑phrase expansion

### Secondary Checks
- TF‑IDF on **Test‑R** rewrites (see §7)
- top feature inspection (watch for intent verbs)

### Feature Veto Loop (operational)
After each pilot batch (e.g., 200 pairs):
1. Run grouped TF‑IDF.
2. Extract top 30 features per label.
3. Add these to a **veto list** for the next generation batch.
4. Repeat 2–3 times until leakage stabilizes.

---

## 7) Rewrite / Test‑R Strategy (Stronger than v1)

Rewrites should **explicitly scrub** the top TF‑IDF features while preserving meaning.  
Suggested constraints for rewrites:
- replace “signature” words (e.g., *eyes, softly, apartment, mission, gratitude*)
- shift register (casual → formal) while preserving content
- alter sentence structure without changing the love line

Evaluation: TF‑IDF should drop **significantly** on Test‑R under GroupKFold.

---

## 8) Controls & Ablations

To show early crystallization isn’t a TF‑IDF artifact:
- Compare DDC vs baseline under **same low‑leakage split**
- Evaluate on subset where TF‑IDF is wrong/low‑confidence
- Include a “label‑swapped within base_id” sanity test to ensure the model isn’t memorizing base templates

---

## 9) Optional: Instrument Shortcuts (Not a purity requirement)
Instead of only fighting leakage, measure it:
- Compute a **TF‑IDF margin** per example or a lexicon‑based “romance score.”
- Analyze crystallization depth and calibration **as a function of shortcut strength**.
- Test whether DDC gains are largest when shortcuts are weak (hard evidence scenarios).

---

## 10) Practical Notes for Implementation

- Use the existing `src/data_generation.py` as a scaffold, but split into **base generation** and **variant generation** stages.
- Preserve existing validation functions (forbidden terms, love pattern, banned phrases).
- Add `base_id` and enforce grouped splitting in any training or leakage scripts.
- Keep a running `biased_phrases.json` from TF‑IDF top features to expand ban lists iteratively.

---

## 11) Success Criterion (Low‑Leakage K=2)

You can call the dataset “low‑leakage” if:
1. **GroupKFold TF‑IDF ≤55–60%**, and
2. Leakage drops further on Test‑R, and
3. DDC vs baseline effects persist under this split.

If (1) fails, the dataset is still usable, but the claims must be framed as “decision dynamics under realistic shortcuts.”
