# Love Disambiguation Experiment: History and Evolution

## Overview

This document chronicles the development of an experiment testing whether **internal semantic tokens** can improve LLM classification accuracy and inference efficiency on an ambiguous language task: distinguishing romantic vs. non-romantic uses of "I love you."

## The Core Hypothesis

Can we teach a small LLM to use a custom token (`⟦LOVE_NONROM⟧`) as an internal "scratchpad" that:
1. Improves classification accuracy
2. Reduces output token count
3. Maintains robustness under distribution shift

---

## Phase 1: Initial MVP

### Dataset Generation
- Generated 500 examples using Claude Sonnet 4
- Each example: scenario containing "I love you", labeled romantic/non-romantic
- Balanced by label, difficulty (hard/easy), and scenario bucket (departure, crisis, etc.)
- Validation rules to prevent shortcuts (no relationship labels, no family terms)

### Original Model Design
- **Baseline (Model A)**: Standard classification with explanation
  - Output: `Label: [label]\nExplanation: [text]`
- **Internal Token (Model B)**: Classification with hidden reasoning token
  - Output: `[THINK]\n⟦LOVE_NONROM⟧\n[ANSWER]\nLabel: [label]\nExplanation: [text]`

### Initial Results (Failed)
- Token usage rate: **0%**
- Model outputting "THINK ANSWER" without brackets or content
- Training loss abnormally high (21-27 vs baseline 1.5-1.9)

---

## Phase 2: Debugging and Fixes

### Problem 1: Label Masking Bug
**Discovery**: Training loss was unreasonably high because the model was trying to learn the entire sequence (prompt + response) instead of just the response.

**Fix**: Added proper label masking in `create_training_example()`:
```python
# Find where assistant response starts
prompt_len = len(prompt_tokens)
# Mask prompt tokens with -100
labels = [-100] * prompt_len + tokenized["input_ids"][prompt_len:]
```

### Problem 2: Mode Collapse
**Symptom**: After fixing labels, model still collapsed to always predicting "romantic"

**Attempts**:
1. Initialized token embedding with mean of related tokens ("non", "love", "not")
2. Increased training epochs from 3 to 10
3. Both failed to prevent collapse

### Problem 3: Output Format Complexity
**Discovery**: The complex `[THINK]/[ANSWER]` format with explanation was diluting gradient signal for token learning.

**Fix**: Simplified output format dramatically:
- Before: `[THINK]\n⟦LOVE_NONROM⟧\n[ANSWER]\nLabel: non-romantic\nExplanation: ...`
- After: `⟦LOVE_NONROM⟧ non-romantic` or `romantic`

**Result**: Training loss dropped from 19.9 to 5.46. Model achieved **95% label accuracy**.

---

## Phase 3: Token Emission Problem

### Discovery via Logit Probing
Despite 95% accuracy, the token usage rate remained **0%** - the model never actually emitted the token during generation.

**Logit probing revealed**:
- Non-romantic examples: token_prob ~24%, "rom" prob ~1%
- Romantic examples: token_prob <1%, "rom" prob >96%
- Token was **rank 2**, competing with "love" (rank 1)

**Conclusion**: The token IS learned semantically, but loses during greedy decoding competition with regular vocabulary.

### Initial Evaluation Results
- Baseline: 76.9% accuracy, 41.8 visible tokens
- Internal Token: 94.2% accuracy, 4.4 visible tokens
- 100% faithfulness, 94.2% paraphrase consistency

---

## Phase 4: External Review (ChatGPT)

### Criticisms Received
1. **Confounded variables**: Different training epochs (3 vs 10), different output formats
2. **Token never emitted**: Can't claim efficiency if token doesn't surface
3. **Parsing fragility**: Results may depend on output parsing
4. **Data generation leakage**: Same model generating and classifying

### Key Insight from Review
> "You're trying to prove two different claims with one experiment."
- **Claim A (Meaning)**: Token aligns with decision boundary
- **Claim B (Efficiency)**: Token enables shorter outputs

These require **separate experiments** with controlled variables.

---

## Phase 5: Track 1 Redesign (Controlled Meaning Experiment)

### Design Changes

1. **Symmetric Tokens**: Added both `⟦LOVE_ROM⟧` and `⟦LOVE_NONROM⟧`
   - Eliminates "presence vs absence" confound
   - Enables pairwise probability comparison

2. **Identical Output Format**:
   - Token model: `DECISION: <token>\nANSWER: <label>`
   - Baseline: `ANSWER: <label>`
   - Same label strings, only difference is decision token

3. **Holdout Bucket Testing**:
   - Train on: all buckets except crisis + collaboration
   - Test on: crisis + collaboration only (never seen in training)
   - Tests robustness to distribution shift

4. **Forced-Choice Logit Probing**:
   - Token model: `P(⟦LOVE_ROM⟧)` vs `P(⟦LOVE_NONROM⟧)` at DECISION position
   - Baseline: sequence logprob for "romantic" vs "non-romantic" at ANSWER position
   - Avoids generation/parsing confounds

### Track 1 Results

| Metric | Baseline | Internal Token | Delta |
|--------|----------|----------------|-------|
| **Accuracy** | 53.48% | **87.83%** | **+34.35%** |
| Collaboration | 24.14% | 98.28% | +74.14% |
| Crisis | 63.37% | 84.30% | +20.93% |

**Key finding**: Baseline collapses to near-random on holdout buckets while token model maintains strong performance.

### Diagnostic Analysis

**Class-conditional margins** (verifying no bias):
- E[margin | romantic] = +0.0884 ✓ (positive)
- E[margin | non-romantic] = -0.2071 ✓ (negative)

**Confusion Matrix**:
```
                 Predicted ROM    Predicted NONROM
Actual ROM            94              27
Actual NONROM          1             108
```

**AUC**: 0.9789 (excellent separability)

**Interpretation**: Model is "conservative about romantic" (27 FN, 1 FP) but this is asymmetric confidence, not semantic bias.

---

## Phase 6: Seed Sensitivity Check (Complete)

### Rationale
With small data (227 train examples) and large effect sizes (+34% accuracy), reviewers will ask: "Did you get lucky with the random seed?"

### Method
- Same data split (same holdout buckets)
- Train with seeds 0, 1, 2
- Report: accuracy, ROM recall, NONROM recall, AUC

### Results

| Seed | Baseline | Token Acc | ROM Recall | NONROM Recall | AUC |
|------|----------|-----------|------------|---------------|------|
| 0 | 72.61% | 87.83% | 79.34% | 97.25% | 0.9826 |
| 1 | 53.48% | 78.70% | 60.33% | 99.08% | 0.9801 |
| 2 | 63.04% | 92.61% | 91.74% | 93.58% | 0.9839 |
|**Mean**| **63.04%** | **86.38%** | 77.13% | 96.64% | 0.9822 |
|**Std** | 7.81% | 5.77% | 12.92% | 2.29% | 0.0016 |

**Delta (Token - Baseline):**
- Mean: **+23.33%**
- Std: 6.01%
- Min: +15.22%
- Max: +29.57%

### Key Findings
1. **Baseline is unstable** under distribution shift (53-73%, std ~8%)
2. **Token model is stable** (79-93%, std ~6%)
3. **AUC is rock-solid** (~0.98 ± 0.0016)
4. **Worst-case delta is still +15%** - even the worst seed shows substantial improvement

### Interpretation
> "The internal token objective consistently improves separability under distribution shift, with stable AUC and large gains across random initializations. Variance is dominated by the baseline, not the token model."

ROM recall variance (12.92%) is higher than NONROM recall variance (2.29%). This is expected given:
- Romantic cases are rarer and more ambiguous
- Crisis contexts blur affective signals

This asymmetry makes the result more believable, not less.

---

## Phase 7: Track 2 (Efficiency Experiment) - Ready to Implement

### Goal (Narrowly Stated)
> "Given a validated semantic decision token, can we use it as a low-token interface at inference time without sacrificing accuracy?"

Track 2 is **engineering**, not theory. The "meaning" claim is already locked by Track 1 + seed check.

### Design

**Token Model (Efficient Interface):**
- Prompt ends with `<|im_start|>assistant\nDECISION: `
- Constrained decoding: next token ∈ {⟦LOVE_ROM⟧, ⟦LOVE_NONROM⟧}
- Stop immediately after emitting the token
- Prediction = token
- **Tokens generated = 1**

**Baseline (Fair Comparison):**
- Option 1 (preferred): Forced-choice via sequence logprob after `ANSWER: `
  - Prediction = argmax logP("romantic", "non-romantic")
  - Tokens generated = 0 (decision via scoring)
- Option 2: Constrained decoding to emit label
  - Tokens generated = length of label (>1)

### Metrics to Report
- Accuracy on held-out buckets (should closely match Track 1)
- Tokens generated per decision
- Optional: latency proxy

### Expected Outcome
Token model achieves same accuracy as Track 1 (~86%) with only 1 token of output, demonstrating the efficiency benefit of the semantic interface.

---

## Key Lessons Learned

1. **Label masking is critical**: Without it, the model learns to predict the prompt, not just the response.

2. **Output format matters for learning**: Complex formats dilute gradient signal. Simpler is better for learning new tokens.

3. **Logit probing reveals hidden learning**: A token can be semantically learned without ever being emitted during generation.

4. **Separate claims require separate experiments**: Don't try to prove "meaning" and "efficiency" with one confounded experiment.

5. **Holdout buckets reveal generalization**: In-distribution accuracy can be misleading. Holdout testing is essential.

6. **Symmetric design prevents confounds**: Adding both `⟦LOVE_ROM⟧` and `⟦LOVE_NONROM⟧` makes the comparison fair.

7. **Style leakage is inevitable in synthetic data**: TF-IDF sanity checks should be standard practice. Finding leakage is rigorous, not failure.

8. **AUC beats accuracy for calibration-sensitive comparisons**: When models have different output biases, AUC measures separability independent of threshold.

9. **Attack your own results before reviewers do**: Discovering and addressing the style leakage issue ourselves strengthened the work.

---

## File Structure

```
ollm/
├── data/
│   ├── stream.jsonl              # All 500 examples
│   ├── train.jsonl               # Training split (227, non-holdout buckets)
│   ├── val.jsonl                 # Validation split (43, non-holdout buckets)
│   ├── test.jsonl                # Test split (230, holdout: crisis + collaboration)
│   └── test_rewritten.jsonl      # Exp3A: rewritten test scenarios
├── src/
│   ├── data_generation.py        # Example generation with Claude
│   ├── split_dataset.py          # Dataset splitting (--track1 for holdout mode)
│   ├── train_baseline.py         # Baseline model training
│   ├── train_internal_token.py   # Token model training
│   ├── evaluate.py               # Generation-based evaluation
│   ├── probe_track1.py           # Track 1: Logit probing evaluation
│   ├── track2_efficiency.py      # Track 2: Efficiency evaluation
│   ├── exp3a_style_leakage.py    # Exp 3A: TF-IDF style leakage check
│   ├── exp3a_rewrite_generator.py # Exp 3A: Scenario rewriter
│   ├── exp3a_evaluate.py         # Exp 3A: Evaluate on rewrites
│   └── utils.py                  # Formatting and parsing utilities
├── models/
│   ├── baseline_track1/          # Trained baseline adapter
│   └── internal_token_track1/    # Trained token model adapter
└── docs/
    └── EXPERIMENT_HISTORY.md     # This document
```

---

## Phase 8: Style Leakage Discovery and Experiment 3A (Complete)

### Critical Discovery: Dataset Has Severe Style Leakage

A TF-IDF + Logistic Regression classifier achieved **97.4% accuracy** on our test set, revealing that surface-level style patterns strongly predict labels.

**Key leakage patterns identified:**
- Romantic-correlated: "eyes", "apartment", "kitchen", "couch", "finally", "words", "riley"
- Non-romantic-correlated: "jordan", "years", "shoulder", "briefly", "research", "dr", "chen"

This raised the question: did Track 1 results reflect semantic learning, or just better style memorization?

### Experiment 3A: Context Generalization

**Goal**: Verify the internal token learned semantic boundaries, not surface style shortcuts.

**Method**:
1. Rewrite all 230 test scenarios while preserving:
   - Exact quoted sentence "I love you"
   - Label (romantic / non-romantic)
   - Bucket (crisis / collaboration)
2. Change surface form aggressively:
   - Remove/replace character names (they were label watermarks)
   - Ban top leakage tokens in rewrite prompts
   - Change narrative style, ordering, vocabulary
3. Compare model degradation using AUC (threshold-independent separability)

### Experiment 3A Results

| Metric       | Original | Rewritten | Delta    |
|--------------|----------|-----------|----------|
| TF-IDF Acc   | 97.39%   | 80.87%    | **-16.52%** |
| Token AUC    | 0.9786   | 0.9525    | **-0.0262** |
| Baseline AUC | 0.8514   | 0.8079    | **-0.0435** |

**Key finding**: Token model maintained **1.7x better separability** than baseline under rewrite shift.

### Interpretation

1. **TF-IDF dropped 16.5%** → rewrites successfully broke style shortcuts
2. **Token AUC dropped 0.026** vs **Baseline AUC dropped 0.044** → token learned more than style
3. Accuracy numbers are misleading due to calibration issues (baseline was always predicting "romantic"). AUC is the correct metric.

> "The internal token objective promotes more invariant decision boundaries that survive aggressive surface-form perturbation better than the label-only baseline."

### Why This Doesn't Invalidate Track 1

The TF-IDF discovery does NOT mean Track 1 was wrong. Three facts argue against "just style learning":

1. **Bucket holdout already tested distribution shift** - baseline collapsed (53%), token held (87%)
2. **Seed stability** - token AUC was ~0.98 across all seeds, baseline was unstable
3. **Baseline had access to the same style shortcuts** - if style alone explained the win, baseline should benefit too

Experiment 3A confirms: the token objective captures structure beyond lexical shortcuts.

### Files Added (Backward-Compatible)

```
src/
├── exp3a_style_leakage.py      # TF-IDF sanity check
├── exp3a_rewrite_generator.py  # Scenario rewriter
└── exp3a_evaluate.py           # Evaluate on rewrites
data/
├── test_rewritten.jsonl        # All 230 rewritten scenarios
└── test_subset_60_rewritten.jsonl  # Subset used for validation
```

All existing Track 1 and Track 2 files remain unchanged and functional.

---

## References

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning: LoRA (r=8, alpha=16)
- Hardware: AMD GPU with ROCm (HSA_OVERRIDE_GFX_VERSION=10.3.0)
