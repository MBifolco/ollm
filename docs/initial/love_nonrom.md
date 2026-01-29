
Interpretation:
> This instance of “love” is explicitly *not* romantic.

Design choices:
- Romantic love is the default (absence of token)
- Non-romantic love is explicitly marked
- Asymmetry reduces degrees of freedom
- Tests whether naming *one* collapsed distinction helps

---

## Task Definition

Each example consists of:

### Input
- 2–4 sentence scenario
- One ambiguous sentence containing “love”  
  (e.g. “I love you.”)

### Output
1. Binary classification:
   - `romantic`
   - `non-romantic`
2. A short English explanation (1–2 sentences)

---

## Models Compared

### Model A: English-Only Baseline
- No internal tokens
- Directly outputs label + explanation

---

### Model B: Internal-Semantic-Token Model
- Has a hidden `THINK` channel
- Allowed to emit **0 or 1 token only**
- The only allowed token is `⟦LOVE_NONROM⟧`
- Internal tokens are forbidden in user-visible output

Interpretation rule:
- THINK emits `⟦LOVE_NONROM⟧` → non-romantic
- THINK emits nothing → romantic

This forces the model to *decide* whether to name the distinction.

---

## Dataset

- ~1,000 total examples
- ~50% romantic / 50% non-romantic
- Mundane, realistic scenarios
- Can be LLM-generated with light manual filtering

Important:
- Avoid poetic or stylized language
- Avoid explicit labels in the scenario
- The ambiguity should be real, not trivial

---

## Training Setup (Deliberately Simple)

Loss components:
1. Classification loss (binary)
2. Explanation generation loss
3. **Hidden-channel length penalty**
   - Penalize THINK outputs longer than 1 token

No:
- reinforcement learning
- agents
- verifier models
- iterative vocab growth

This MVP is about representation, not architecture.

---

## Evaluation Metrics (Only These Matter)

### 1. Accuracy
- Must not drop vs baseline
- If accuracy drops → stop

---

### 2. Token Efficiency
Measure:
- total tokens (THINK + visible output)
- visible output tokens only

Look for:
- same accuracy with fewer tokens
- or shorter explanations for same quality

---

### 3. Faithfulness / Explanation Consistency (Key Signal)

Check whether:
- the explanation text implies the same label the model outputs

Method:
- Train a small classifier (or heuristic) to infer label from explanation text alone
- Count explanation–label mismatches

Hypothesis:
- Explicit internal distinction reduces mismatch rate

---

### 4. Paraphrase Robustness
- Paraphrase ~100 test scenarios
- Preserve meaning and label
- Measure prediction flips

Lower variance suggests more stable internal representation.

---

## What Counts as Success

Any **one** of the following justifies continuing:

- ~10–20% reduction in explanation–label mismatches
- Same accuracy with ~15% fewer tokens
- Meaningfully lower variance under paraphrase

This MVP does **not** need dramatic gains.

---

## What Counts as Failure

- Internal token is ignored
- Token usage is random or inconsistent
- Gains only appear because labels were spoon-fed
- No measurable difference from baseline

Failure here is informative and acceptable.

---

## Implementation Notes

- Use a small open model (1B–3B range)
- Prefer LoRA fine-tuning
- Add the token to the tokenizer explicitly
- Mask token from appearing in visible output
- Keep decoding deterministic during evaluation
- Log THINK usage frequency and failure cases

---

## Explicit Non-Goals (For This MVP)

- No automatic token creation
- No emergent language claims
- No scaling claims
- No safety analysis
- No multi-domain generalization

This MVP exists to answer **one question cleanly**.

---

## One-Sentence Summary

> Does explicitly naming a collapsed semantic distinction internally reduce ambiguity-driven inference cost in an LLM?

That’s the experiment.
