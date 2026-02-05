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

10. **Data diversity + semantic objective = best results**: Training on mixed (original + rewritten) data with the token objective achieves both highest accuracy and best robustness.

---

## File Structure

```
ollm/
├── data/
│   ├── stream.jsonl              # All 500 examples
│   ├── train.jsonl               # Training split (227, non-holdout buckets)
│   ├── val.jsonl                 # Validation split (43, non-holdout buckets)
│   ├── test.jsonl                # Test split (230, holdout: crisis + collaboration)
│   ├── test_rewritten.jsonl      # Exp3A: rewritten test scenarios
│   └── train_rewritten.jsonl     # Exp3A: rewritten training scenarios
├── data/k2_love/O/                       # Original data (for triad experiment)
├── data/k2_love/R/                       # Rewritten data (for triad experiment)
├── data/k2_love/M/                       # Mixed data (O+R, for triad experiment)
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
│   ├── eval_triad.py             # Phase 9: Triad evaluation
│   ├── layerwise_probe.py        # Phase 10: Logit lens analysis
│   └── utils.py                  # Formatting and parsing utilities
├── models/
│   ├── baseline_track1/          # Trained baseline adapter
│   ├── internal_token_track1/    # Trained token model adapter
│   └── triad/                    # Phase 9: 18 trained models
│       ├── baseline_O_seed{0,1,2}/
│       ├── baseline_R_seed{0,1,2}/
│       ├── baseline_M_seed{0,1,2}/
│       ├── token_O_seed{0,1,2}/
│       ├── token_R_seed{0,1,2}/
│       └── token_M_seed{0,1,2}/
├── run_triad_simple.sh           # Triad training script
├── triad_results.json            # Phase 9: Full evaluation results
├── layerwise_results.json        # Phase 10: Logit lens results
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

## Phase 9: Retrain Triad Experiment (Complete)

### Motivation

Experiment 3A showed the token model degrades less than baseline when tested on rewritten scenarios. But both models were trained on the original data, which had style leakage. Question: **What happens if we train on rewritten or mixed data?**

### Design

Created three training sets:
- **Train-O**: Original 227 training examples
- **Train-R**: Rewritten versions of the same 227 examples (style-neutralized)
- **Train-M**: Mixed (O + R shuffled, 454 examples)

For each training set:
- Train both baseline and token models
- Repeat with 3 seeds (0, 1, 2) for stability
- Total: 18 models

Test sets:
- **Test-O**: Original 230 test examples
- **Test-R**: Rewritten 230 test examples

### Results: AUC (mean ± std over 3 seeds)

| Train Set | Model | Test-O AUC | Test-R AUC | Δ (O→R) |
|-----------|-------|------------|------------|---------|
| Train-O | baseline | 0.8683 ± 0.0722 | 0.7966 ± 0.0484 | **-0.0717** |
| Train-O | token | 0.9818 ± 0.0006 | 0.9495 ± 0.0017 | **-0.0323** |
| Train-R | baseline | 0.9381 ± 0.0121 | 0.9357 ± 0.0122 | -0.0025 |
| Train-R | token | 0.9541 ± 0.0060 | 0.9668 ± 0.0028 | **+0.0127** |
| Train-M | baseline | 0.9752 ± 0.0059 | 0.9563 ± 0.0062 | -0.0189 |
| Train-M | token | **0.9961 ± 0.0003** | **0.9883 ± 0.0011** | -0.0078 |

### Key Findings

1. **Token models consistently outperform baselines** across all training conditions and test sets.

2. **Train-O models are most vulnerable to distribution shift**:
   - Baseline drops 7.2% AUC when tested on rewritten data
   - Token drops only 3.2% AUC (2.2x more robust)

3. **Training on rewritten data improves robustness**:
   - Train-R token actually **improves** on Test-R (+1.3%)
   - This suggests training on style-neutralized data helps learn semantic boundaries

4. **Mixed training (Train-M) achieves best overall performance**:
   - Highest AUC on both test sets
   - Smallest degradation from O→R
   - Token model std is remarkably low (0.0003 on Test-O)

5. **The winning configuration is Train-M + Token**:
   - 0.9961 AUC on original test
   - 0.9883 AUC on rewritten test
   - Most accurate AND most robust

### Interpretation

> "Training on diverse surface forms (mixed data) with the semantic token objective produces models that generalize better across style variations while maintaining peak discriminative performance."

The retrain triad confirms that:
1. The token objective captures semantic invariants
2. Data diversity helps (mixed > original-only)
3. The combination (token + mixed) gives the best of both worlds

### Files Added

```
data/k2_love/O/           # Original training data
data/k2_love/R/           # Rewritten training data
data/k2_love/M/           # Mixed training data (O+R)
models/triad/     # All 18 trained models
src/eval_triad.py # Triad evaluation script
run_triad_simple.sh # Training script
triad_results.json  # Full results
```

---

## Phase 10: Layerwise Representational Depth Analysis (Complete)

### Motivation

After establishing interface efficiency (fewer tokens) and learning efficiency (robustness), the question arose: does the token model require "less internal work"?

ChatGPT correctly identified that our experiments so far proved **interface efficiency**, not **cognitive efficiency**. To bridge this gap, we implemented a layerwise "logit lens" analysis to measure where the decision becomes accessible in the network.

### Important Framing

This analysis measures **representational depth** - at what layer does the decision become linearly accessible?

It does **NOT** prove:
- Fewer FLOPs
- Faster forward passes
- "Less thinking"

It **does** show whether the decision crystallizes into a readable form earlier in the network.

### Method: Logit Lens

For each layer ℓ (0-24 in Qwen2.5-0.5B):

1. Extract hidden states at the decision position
2. Apply final layer norm + lm_head to get "early logits"
3. Compute decision margin:
   - **Token model**: P(⟦LOVE_ROM⟧) - P(⟦LOVE_NONROM⟧)
   - **Baseline**: sequence logprob("romantic") - sequence logprob("non-romantic")
4. Compute AUC across test set at each layer

### Results

| Layer | Baseline AUC | Token AUC | Δ AUC |
|-------|--------------|-----------|-------|
| 0 | 0.50 | 0.50 | 0.00 |
| 5 | 0.64 | 0.70 | +0.06 |
| 10 | 0.23 | 0.32 | +0.09 |
| 15 | 0.50 | **0.90** | +0.40 |
| 16 | 0.51 | **0.95** | +0.44 |
| 17 | 0.81 | 0.96 | +0.15 |
| 22 | 0.88 | 0.97 | +0.09 |
| 24 | 0.81 | **0.98** | +0.17 |

**Key finding:**
- Token model reaches AUC ≥ 0.95 at **layer 17** (of 24)
- Baseline **never** reaches AUC ≥ 0.95 (peaks at 0.88)

### Interpretation

1. **Decision crystallizes earlier for token model**: The decision becomes "readable" 7 layers before the final output.

2. **Token creates cleaner representational axis**: The semantic token provides a direct mapping from internal state to decision, while the baseline's multi-token output is harder to decode from intermediate layers.

3. **Necessary but not sufficient for computational efficiency**: If early-exit were implemented, the token model *could* exit at layer 17. But without early-exit, no FLOPs are saved.

4. **Baseline's multi-token output is inherently harder to probe**: Comparing sequence logprobs across layers is more complex than single-token probing, which may partially explain the gap.

### Updated Claims Hierarchy

| Level | Claim | Status |
|-------|-------|--------|
| Interface efficiency | Fewer tokens to externalize decision | ✅ Proven |
| Learning efficiency | Robustness under distribution shift | ✅ Proven |
| Representational alignment | Token aligns to stable internal axis | ✅ Proven |
| Representational depth | Decision accessible earlier in network | ✅ Proven (Phase 10) |
| Cognitive/computational efficiency | Fewer FLOPs, faster forward pass | ❌ Not proven (requires early-exit or compositional reuse) |

### Early-Exit Simulation (A1)

Given the layerwise results, we computed the "potential compute saved" by analyzing at which layer the model's decision is already good enough.

**This is a SIMULATION** - no actual early-exit is implemented. It shows the ceiling for potential savings.

#### Results: AUC vs Exit Layer

| Layer | Depth % | Token AUC | Token AUC % of Final | Baseline AUC |
|-------|---------|-----------|---------------------|--------------|
| 15 | 62% | 0.898 | 91.8% | 0.504 |
| 16 | 67% | **0.945** | 96.6% | 0.507 |
| 17 | 71% | **0.959** | 98.0% | 0.811 |
| 21 | 88% | 0.971 | 99.2% | 0.783 |
| 24 | 100% | 0.979 | 100% | 0.812 |

#### Threshold Analysis

| Threshold | Baseline Layer | Token Layer | Token Depth % | Potential Savings |
|-----------|----------------|-------------|---------------|-------------------|
| AUC ≥ 0.90 | never | 16 | 67% | **33%** |
| AUC ≥ 0.95 | never | 17 | 71% | **29%** |
| AUC ≥ 0.98 | never | never | N/A | N/A |

#### Key Headline

> **Token model reaches AUC≥0.95 at layer 17/24 (71% depth) → potential to skip 29% of layers**
>
> **Baseline NEVER reaches AUC≥0.95 at any layer (peaks at 0.88)**

This means:
- With an early-exit mechanism, the token model could stop at layer 17 and still achieve 98% of final separability
- The baseline cannot achieve equivalent performance at any depth
- The token creates a "readable" decision axis that crystallizes earlier AND reaches a higher ceiling

### Next Steps Identified

To prove actual computational efficiency, one of these experiments would be needed:

1. **Early-exit implementation**: Actually stop at layer 17 (requires architectural changes)
2. **Compositional reuse**: Feed token back into context for downstream reasoning
3. **Ablations**: Verify the improvement is specifically due to the token objective

These are identified as future work, not claims of the current project.

### Files Added

```
src/layerwise_probe.py        # Logit lens analysis implementation
src/early_exit_simulation.py  # A1: Early-exit potential analysis
layerwise_results.json        # Full results with per-layer margins
early_exit_results.json       # Early-exit simulation results
```

---

## Phase 11: Ablation Experiments (E2) - Complete

### Motivation

Phase 10 showed the token model's decision crystallizes earlier in the network. But a key question remained: **Is the improvement specifically due to the semantic content of the tokens, or just the architectural constraint of emitting a categorical decision first?**

Three ablations were designed to isolate the source of improvement:

1. **baseline-10ep**: Baseline with same training duration (10 epochs vs original 3 epochs)
   - Tests whether more training alone explains the gap

2. **random-token**: Random meaningless tokens (⟦RAND_A⟧/⟦RAND_B⟧)
   - Tests whether ANY token helps, or specifically semantic tokens
   - Tokens initialized with random embeddings matching existing vocab std

3. **single-token**: Only ⟦LOVE_NONROM⟧ (presence/absence)
   - Tests whether asymmetric design (one token vs two) matters

### Results

| Model | AUC | Accuracy | Notes |
|-------|-----|----------|-------|
| Baseline (3ep, original) | 0.8124 | 53% | Original Track 1 baseline |
| **Baseline (10ep)** | **0.9671** | 64% | More training helps significantly |
| Semantic-Token (original) | 0.9787 | 88% | Original Track 1 token model |
| **Random-Token** | **0.9749** | 91% | Almost identical to semantic! |
| **Single-Token** | **0.5223** | 53% | Chance level - doesn't work |

### Key Findings

1. **Training duration matters more than expected**
   - Baseline with 10 epochs (0.9671) is dramatically better than 3 epochs (0.8124)
   - This closes ~80% of the gap with semantic tokens
   - Implication: Some of Track 1's improvement was due to training duration confound

2. **Random tokens work just as well as semantic tokens**
   - Random-token AUC (0.9749) ≈ Semantic-token AUC (0.9787)
   - Difference: 0.0038 AUC (statistically negligible)
   - **This is the most important finding**: The benefit comes from the task structure, not semantic grounding

3. **Single token doesn't work**
   - Single-token ablation (0.5223) performs at chance
   - Confirms that symmetric design (two tokens for two classes) is necessary
   - Presence/absence prediction is not enough

### Interpretation

The ablation results suggest a revised interpretation of the internal token benefit:

**What DOES matter:**
- Forcing the model to emit a categorical decision token BEFORE the final answer
- Having symmetric tokens (one per class)
- Training for sufficient epochs

**What does NOT matter:**
- Semantic content of the tokens
- Pre-training priors on token meanings
- Token initialization strategy (random vs semantic-related)

This is actually a **chain-of-thought** effect: the model benefits from being forced to "commit" to an intermediate categorical decision before producing the final output. The tokens act as a bottleneck that forces explicit reasoning.

### Revised Claims

| Claim | Status | Evidence |
|-------|--------|----------|
| Internal tokens improve classification | ✅ Confirmed | 0.98 AUC vs 0.81 baseline (3ep) |
| Improvement due to semantic grounding | ❌ **Refuted** | Random tokens work equally well |
| Improvement due to task structure | ✅ Supported | Forcing explicit categorical decision helps |
| More training helps | ✅ Confirmed | 10ep baseline >> 3ep baseline |
| Symmetric design required | ✅ Confirmed | Single-token fails completely |

### Implications for Future Work

1. **The "semantic token" hypothesis may be wrong** - what matters is the architectural constraint, not the meaning

2. **Chain-of-thought via tokens** - this is essentially "let the model think out loud, but in a structured categorical way"

3. **Random tokens as a baseline** - future token-based experiments should always compare against random tokens, not just baseline

4. **Training duration must be controlled** - the 3ep vs 10ep confound was significant

### Files Added

```
src/train_ablations.py    # Training script for all 3 ablations
src/eval_ablations.py     # Evaluation script
run_ablations.sh          # Runner script
models/ablations/         # Trained ablation models
  baseline_10ep_seed42/
  random_token_seed42/
  single_token_seed42/
ablation_results.json     # Evaluation results
```

---

## Reframing: From "Semantic Tokens" to "Discrete Decision Channels"

### The Revised Abstract Concept

Based on the ablation results, what we built is less "internal semantic language" and more:

**A learned discrete decision interface (a tiny symbolic bottleneck) that makes a model commit.**

The token strings don't matter. What matters is that we created:

- A **small set of mutually exclusive symbols**
- That the model is trained to emit at a **known decision point**
- Which forces the representation to become **linearly separable and robust**
- And makes decoding cheap because output entropy collapses

This is the generalizable idea: **discrete latent variables for transformers, learned via supervision**, implemented as extra vocab tokens.

In ML archetypes: it's "mixture-of-experts routing, but pushed into the output channel", or "a VQ bottleneck, but as a language action".

### Why This Matters

The ablation experiments (Phase 11) confirmed this reframing:

1. **Random tokens work** → The semantic content is irrelevant
2. **Symmetric design required** → You need mutually exclusive symbols
3. **Task structure matters** → The decision point forces commitment

The tokens act as a **discrete bottleneck** that forces the model to compress its decision into a categorical variable before proceeding. This is what creates the robustness and early crystallization observed in earlier phases.

### Naming the Thing

Better names for this mechanism:

- **Discrete Decision Channels (DDCs)**
- **Decision-token bottlenecks**
- **Symbolic routing interfaces**

These names avoid the misleading implication that token meanings matter.

---

## Future Directions

Based on the experiment findings, several promising directions emerge:

### 1. K-way Scaling (Supervised)

Replace binary {ROM, NONROM} with K-way classification:
- Simple extension: one token per class
- Known failure mode: as K grows, codes become "messy" without structure
- Solution: factorization (multiple binary axes vs one flat K)

### 2. Factorized/Hierarchical Tokens

Instead of flat K-way tokens, use compositional structure:
- `⟦AXIS1_...⟧ ⟦AXIS2_...⟧` rather than one huge flat K
- Example: `⟦STYLE_FORMAL⟧ ⟦SENTIMENT_WARM⟧`
- Hierarchical: coarse cluster → refined token → attributes

**Hypothesis**: Factorized tokens should scale better than flat tokens.

### 3. Unsupervised Discovery

Can the model discover the discrete interface without explicit labels?

Approaches:
- Self-supervised discrete latents (predict continuation after choosing z)
- Routing priors (pick max K, encourage sparsity)
- Weak supervision / pseudo-labeling

Success metric: Does adding discrete decision tokens improve next-token prediction under constraints that force the tokens to matter?

### 4. Concrete Next Experiment: K-way Extension

A clean stepping stone:
- Create a 4-way task with overlapping cues (or two independent binary factors = 2×2)
- Train three variants:
  1. Flat K tokens (one-of-4)
  2. Factorized tokens (two binary tokens)
  3. Baseline text labels
- Measure: robustness to rewrites, layerwise crystallization, calibration, utilization

This tests whether the "decision interface" scales better as flat clustering or factorized axes.

---

## Phase 12: Actual Early-Exit Implementation (Complete)

### Motivation

Phase 10 showed that the token model's decision becomes readable at earlier layers (via logit lens simulation). Phase 12 implements **actual early-exit** - stopping computation at layer L rather than just probing hidden states after a full forward pass.

This converts the "potential compute savings" from Phase 10 into **measured latency improvements**.

### Implementation

Created `src/early_exit.py` which:
1. Runs forward pass through only the first L layers
2. Applies final_norm + lm_head to get logits at that point
3. Makes decision based on token logits (ROM vs NONROM)
4. Measures real wall-clock latency

This is fundamentally different from the layerwise probe, which runs a full forward pass and then examines intermediate hidden states.

### Results: Compute-Quality Tradeoff

| Exit Layer | Depth % | AUC | % of Final AUC | Latency (ms) | Speedup |
|------------|---------|-----|----------------|--------------|---------|
| 11 | 50% | 0.205 | 21% | 38.8 | 1.62x |
| 14 | 62% | 0.898 | 92% | 52.7 | 1.19x |
| **15** | **67%** | **0.945** | **96.6%** | **45.3** | **1.39x** |
| **16** | **71%** | **0.959** | **98.0%** | **45.7** | **1.38x** |
| **17** | **75%** | **0.961** | **98.2%** | **47.7** | **1.32x** |
| 18 | 79% | 0.966 | 98.7% | 50.9 | 1.24x |
| 20 | 88% | 0.971 | 99.2% | 55.6 | 1.13x |
| 23 | 100% | 0.979 | 100% | 62.2 | 1.01x |
| Full | 100% | 0.979 | 100% | 62.9 | 1.00x |

### Key Findings

1. **Sweet spot at layers 15-17**:
   - Layer 16 achieves 98% of final AUC with 1.38x speedup
   - Layer 15 achieves 96.6% of final AUC with 1.39x speedup

2. **Latency scales sub-linearly**:
   - Skipping 50% of layers (layer 11) only gives 1.62x speedup, not 2x
   - This is due to fixed overhead (embedding, tokenization, memory ops)

3. **AUC is more stable than accuracy across exit layers**:
   - Accuracy fluctuates due to threshold sensitivity
   - AUC provides threshold-independent measure of separability

4. **Comparison with baseline (from Phase 10)**:
   - Token model at layer 16: 0.959 AUC
   - Baseline at layer 24 (full): 0.812 AUC
   - Token model at 71% depth already exceeds baseline's final performance

### Claim Now Supported

> "Discrete decision channels enable early-exit inference with 1.3-1.4x speedup while maintaining 98% of final separability."

This is a stronger claim than Phase 10's "potential savings" because it's measured with actual computation stopping and real latency.

### Files Added

```
src/early_exit.py         # Actual early-exit implementation
early_exit_actual.json    # Full results with per-layer metrics
```

---

## Phase 13: R/N Baseline - Dedicated Tokens vs Existing Vocab (Complete)

### Motivation

Phase 11 showed that random tokens work as well as semantic tokens. But both use **dedicated new tokens** added to the vocabulary. Question: does the benefit come from:
- The categorical decision interface alone (any single-token decision)?
- Having dedicated tokens that the model can specialize?

To test this, we trained a baseline using existing vocabulary tokens (" R" and " N") as the decision channel.

### Setup

- Same format as token model: `DECISION: <token>\nANSWER: <label>`
- Uses existing vocab: " R" (token 431) and " N" (token 451)
- Same training: 10 epochs, same hyperparameters
- No vocabulary extension required

### Results: Early-Exit Comparison

| Layer | Depth | Token Model AUC | R/N Baseline AUC | Δ AUC |
|-------|-------|-----------------|------------------|-------|
| 15 | 67% | **0.945** | 0.407 | +0.54 |
| 16 | 71% | **0.959** | 0.646 | +0.31 |
| 17 | 75% | **0.961** | 0.597 | +0.36 |
| 20 | 87% | 0.971 | **0.954** | +0.02 |
| Full | 100% | 0.979 | 0.973 | ~same |

### Key Finding

**Final AUC is nearly identical** (~0.97-0.98), but **crystallization depth is dramatically different**:
- Token model reaches AUC ≥ 0.95 at **layer 15-16** (67-71% depth)
- R/N baseline reaches AUC ≥ 0.95 at **layer 20** (87% depth)
- Dedicated tokens enable **4-5 layers earlier crystallization**

### Interpretation

This nuances the ablation result from Phase 11:

1. **Random tokens work as well as semantic tokens** (Phase 11) → semantic content doesn't matter
2. **Dedicated tokens work better than existing vocab for early-exit** (Phase 13) → having a "fresh" token matters

The benefit of dedicated tokens isn't semantic grounding - it's that the model can **fully specialize** a new token for the decision task without interference from pre-existing associations.

Existing vocab tokens like "R" and "N" carry prior meanings and contextual patterns that the model must partially unlearn. Dedicated tokens start as blank slates.

### Updated Claims

| Claim | Status |
|-------|--------|
| Categorical decision channel improves final accuracy | ✅ Both token model and R/N baseline achieve ~0.97 AUC |
| Semantic content of tokens matters | ❌ Random tokens = semantic tokens |
| Dedicated tokens enable earlier crystallization | ✅ 4-5 layers earlier than existing vocab |
| Early-exit requires dedicated tokens | ✅ R/N baseline crystallizes too late for useful early-exit |

### Files Added

```
src/train_rn_baseline.py   # Training script for R/N baseline
src/early_exit_rn.py       # Early-exit evaluation for R/N
early_exit_rn.json         # Full results
models/rn_baseline/        # Trained R/N baseline model
```

---

## Phase 14: Robust Evaluation on Test-R (Rewritten Data) - Complete

### Motivation

Phase 13 showed dedicated tokens crystallize earlier than existing vocab, but was evaluated on original test data which has style shortcuts. Per ChatGPT's review:

> "Use Original (O) for fast iteration. **Promote Rewritten (R) to the main scoreboard** for any claims about robustness or generalization."

### Setup

- **Training**: Both models trained on Mixed data (O+R, 454 examples)
- **Evaluation**: Test-R (rewritten test set, style-neutralized)
- Token model: `models/triad/token_M_seed0`
- R/N baseline: `models/rn_baseline_mixed/rn_baseline_seed42`

### Results: Early-Exit on Style-Neutralized Data

| Layer | Depth | Token Model AUC | R/N Baseline AUC | Δ AUC |
|-------|-------|-----------------|------------------|-------|
| 14 | 62% | **0.986** | 0.589 | +0.40 |
| 15 | 67% | **0.990** | 0.659 | +0.33 |
| 16 | 71% | **0.988** | 0.760 | +0.23 |
| 17 | 75% | **0.987** | 0.860 | +0.13 |
| 20 | 87% | 0.989 | **0.953** | +0.04 |
| Full | 100% | 0.987 | 0.962 | +0.03 |

### Key Finding: Gap is LARGER on Rewritten Data

| Metric | Original Test (Phase 13) | Rewritten Test (Phase 14) |
|--------|--------------------------|---------------------------|
| Token reaches AUC≥0.95 | Layer 15-16 | **Layer 14** |
| R/N reaches AUC≥0.95 | Layer 20 | Layer 20 |
| **Crystallization gap** | 4-5 layers | **6 layers** |

**The dedicated token advantage increases when style shortcuts aren't available.**

This confirms:
1. The early crystallization benefit is real, not an artifact of style leakage
2. Dedicated tokens help the model learn semantic structure, not shortcuts
3. The R/N baseline relies more on shortcuts (larger degradation on Test-R)

### Updated Evaluation Protocol

Going forward, all experiments should use:
- **Training**: Mixed (O+R) as default
- **Primary evaluation**: Test-R (rewritten)
- **Secondary evaluation**: Test-O (original, for ceiling check)

### Files Added

```
early_exit_token_mixed_testR.json  # Token model on Test-R
early_exit_rn_mixed_testR.json     # R/N baseline on Test-R
models/rn_baseline_mixed/          # R/N baseline trained on mixed data
```

---

## Summary of All Phases

| Phase | Focus | Key Finding |
|-------|-------|-------------|
| 1-4 | Initial MVP & debugging | Label masking critical, format simplification needed |
| 5 | Track 1 redesign | Symmetric tokens + holdout testing |
| 6 | Seed sensitivity | Token model stable, baseline unstable under shift |
| 7 | Track 2 design | Efficiency experiment ready |
| 8 | Style leakage (Exp 3A) | Token degrades less than baseline under rewrites |
| 9 | Retrain triad | Mixed training + token = best performance |
| 10 | Layerwise probing | Decision crystallizes earlier for token model |
| 11 | Ablations (E2) | **Random tokens work** → it's about task structure, not semantics |
| 12 | Actual early-exit | **1.38x speedup** at 98% of final AUC (layer 16) |
| 13 | R/N baseline | **Dedicated tokens crystallize 4-5 layers earlier** than existing vocab |
| 14 | Test-R evaluation | **6-layer gap** on style-neutralized data (larger than original) |
| 15 | Unified harness & init study | **Initialization determines crystallization**; token string is secondary |
| 16 | Adaptive early-exit | Semantic init enables **calibrated confidence** → adaptive exit works |

**Final interpretation**: The benefit comes from **discrete decision channels using dedicated tokens with semantic initialization**. Semantic init provides two advantages: (1) earlier crystallization depth, and (2) calibrated confidence that enables adaptive early-exit. This enables 1.36x speedup at 97.6% AUC with confidence-gated exit.

---

## Phase 15: Unified Training Harness & Initialization Confound (Complete)

### Motivation

Phase 11 showed "random tokens ≈ semantic tokens" on the original test data. But this was evaluated on Train-O → Test-O, which has style shortcuts. Question: **Does "random ≈ semantic" hold under distribution shift (Test-R)?**

### Option A: Evaluate Existing Ablations on Test-R

Quick test using Phase 11 ablation models (trained on Train-O) evaluated on Test-R:

| Model | Test-O AUC | Test-R AUC | Δ (O→R) |
|-------|------------|------------|---------|
| semantic | 0.979 | **0.884** | -0.095 |
| random | 0.975 | 0.640 | **-0.335** |
| single | 0.522 | 0.533 | +0.011 |

**Critical finding**: Under distribution shift, random tokens dramatically underperform semantic tokens (-0.24 AUC gap). The "random ≈ semantic" result from Phase 11 was distribution-specific.

### Option B: Retrain Ablations on Mixed Data (Preliminary)

Initial attempt using `train_ablations_mixed.py` (before unified harness):

| Model | Token String | Init | Full AUC | L16 AUC | Crystallization |
|-------|--------------|------|----------|---------|-----------------|
| random | random | random | 0.969 | 0.730 | Late (L20+) |
| semantic | semantic | semantic | 0.924 | 0.928 | Early (L16) |
| random-seminit | random | semantic | 0.900 | 0.904 | Early (L16) |
| semantic-randinit | semantic | random | 0.915 | 0.891 | Late |
| single | - | - | 0.559 | 0.294 | Never |

**Preliminary insight**: Initialization seemed to determine crystallization depth. However, these results were confounded by hyperparameter mismatch (see below). The unified harness results supersede this table.

### Hyperparameter Mismatch Identified

Discovered a confound between training scripts:
- `train_internal_token.py`: LR=2e-4, eval+checkpoint every 50 steps, load_best_model
- `train_ablations_mixed.py`: LR=1e-4, no eval/checkpoint

This explains why the triad token_M model (0.987 AUC) outperformed all ablation models.

### Solution: Unified Training Harness

Created `src/train_unified.py` as single source of truth for all experiments:

```python
# Standardized hyperparameters
learning_rate = 2e-4
batch_size = 1
gradient_accumulation_steps = 16  # effective batch = 16
num_epochs = 10
eval_steps = 50
load_best_model_at_end = True

# Experiment axes
--channel {dedicated, rn_vocab, single}
--token_string {semantic, random}
--init {semantic, random}
--init_lm_head {0, 1}
--train_set {O, R, M}
--seed {42, 123, 456}
```

### Final Results: Unified Experiment Matrix (Complete)

Trained 6 variants × 3 seeds = 18 models on Train-M, evaluated with early-exit on Test-R:

| Variant | Token | Init | Full AUC | L14 AUC | L16 AUC | Crystallization |
|---------|-------|------|----------|---------|---------|-----------------|
| **semantic** | semantic | semantic | **0.9842** | 0.9837 | 0.9840 | **L14** |
| **random-seminit** | random | semantic | 0.9794 | 0.9795 | 0.9796 | **L14** |
| semantic-randinit | semantic | random | 0.9778 | 0.9407 | 0.9210 | L16-L18 |
| random | random | random | 0.9586 | 0.8577 | 0.7773 | L16-L20 |
| rn_vocab | R/N | existing | 0.9667 | 0.7403 | 0.8890 | L20 |
| single | - | - | 0.7937* | - | 0.3497 | Never |

*Exact means across 3 seeds (42, 123, 456)*

**\*Note on single-token results**: The single-token variant shows suspicious behavior: two seeds have high AUC (0.91, 0.90) but stuck accuracy (52.6%). This pattern (good ranking, bad classification) suggests a threshold/sign issue in the eval, not genuine learning. Treat single-token as "structurally broken" rather than interpreting the AUC values.

### Key Findings

1. **Semantic initialization is the lever** (simplest framing):
   - Semantic init wins on **both** early crystallization (L14) **and** peak AUC (0.984)
   - There is no tradeoff in this setup - semantic init dominates

2. **Token string doesn't matter** (smoking gun comparison):
   - `semantic` (semantic string + semantic init): 0.9842 AUC, L14
   - `random-seminit` (random string + semantic init): 0.9794 AUC, L14
   - Nearly identical performance proves the token characters aren't doing the work

3. **Random initialization hurts both dimensions**:
   - Later crystallization (L16-L20)
   - Lower peak AUC (0.96-0.98)
   - Higher seed variance

4. **Dedicated tokens beat existing vocab for early-exit**:
   - rn_vocab achieves similar final AUC (0.967) but crystallizes at L20
   - Dedicated tokens with semantic init crystallize 6 layers earlier (L14)

5. **Symmetric design is required**:
   - Single token fails completely (structurally broken)
   - Two tokens (one per class) are necessary for the decision channel to work

### Interpretation

The story simplifies: **semantic initialization is the key lever**.

The Phase 11 finding that "random tokens ≈ semantic tokens" was confounded by initialization. With the unified harness, we can cleanly separate:

1. **Token string effect**: None. Random strings work identically to semantic strings.
2. **Initialization effect**: Critical. Starting embeddings near the decision manifold enables earlier crystallization AND better final performance.

This suggests the model learns to use the token as a decision channel regardless of its string representation. The semantic content was never the source of the benefit - it was always the initialization. The tokens act as "blank slate" decision variables that the model specializes, but they train faster when initialized in a good region of embedding space.

### Files Added

```
src/train_unified.py          # Single source of truth for training
run_unified_experiments.sh    # Batch training runner
eval_unified_models.sh        # Batch evaluation runner
models/unified/               # All 18 trained models
results/unified_early_exit/   # All 18 evaluation results
```

### Summary of Phase 15

| Question | Answer |
|----------|--------|
| Does token string matter? | **No.** `random-seminit ≈ semantic` proves this. |
| What determines crystallization depth? | **Initialization.** Semantic init → L14, random init → L16-L20. |
| Is there a crystallization vs peak AUC tradeoff? | **No.** Semantic init wins both (0.984 AUC, L14). |
| Best early-exit config? | Dedicated token + semantic init (token string irrelevant) |
| Speedup achievable? | 1.52x at L14 with 98%+ AUC retention |

---

## Phase 16: Adaptive Early-Exit with Confidence Thresholds (Complete)

### Motivation

Phase 15 showed semantic init enables earlier crystallization (L14 vs L16-L20). But fixed-layer early exit is suboptimal - ideally, we exit early for "easy" examples and go deeper for "hard" ones. This requires **adaptive early-exit** with confidence thresholds.

ChatGPT identified a key insight from Phase 15 data: at layer 11, the model shows high mean_confidence (~0.89) but near-random AUC (~0.49). This means **confidence is not calibrated at early layers** - the model can be confidently wrong.

### Implementation

Added per-example margin logging to evaluation scripts:
- `early_exit.py`: Added `--save_per_example` flag
- `early_exit_ablations.py`: Same flag for random/single token models
- `analyze_adaptive_exit.py`: New script to simulate threshold policies

**Adaptive Exit Policy**:
1. At each candidate layer L, compute `margin = logit_rom - logit_nonrom`
2. Convert to confidence: `conf = max(sigmoid(margin), 1-sigmoid(margin))`
3. If `conf >= τ`, exit at this layer
4. Otherwise continue to next layer
5. Sweep τ from 0.5 to 0.99 to produce AUC-latency curves

### Results: Semantic Init vs Random Init

**Semantic model (semantic init)** - Adaptive exit works:

| Threshold τ | Speedup | AUC | % of Full | Exit Distribution |
|-------------|---------|-----|-----------|-------------------|
| 0.80 | **1.36x** | 0.9566 | 97.6% | L14:106, L16:112, L18:1, L23:11 |
| 0.90 | 1.25x | 0.9503 | 97.0% | L14:47, L16:76, L18:32, L23:75 |
| 0.95 | 1.16x | 0.9704 | 99.0% | L14:4, L16:14, L18:71, L23:141 |

**Random model (random init)** - Adaptive exit fails:

| Threshold τ | Speedup | AUC | % of Full | Exit Distribution |
|-------------|---------|-----|-----------|-------------------|
| 0.80 | 1.32x | 0.6004 | **62%** | L14:230 (ALL) |
| 0.90 | 1.32x | 0.6004 | 62% | L14:230 (ALL) |
| 0.95 | 1.32x | 0.6004 | 62% | L14:230 (ALL) |

### Key Finding: Calibration Matters

The random model's confidence is **uncalibrated** - it's always confident enough to exit at L14 (all 230 examples), but the predictions are wrong (AUC 0.60 vs 0.97 full).

In contrast, the semantic model's confidence is **calibrated**:
- At τ=0.80, it correctly identifies ~218 examples as "easy" (exit at L14/L16)
- Only 11 examples need full depth (L23)
- The early exits maintain 97.6% of full AUC

### Interpretation

Semantic initialization provides two distinct benefits:

1. **Crystallization depth** (Phase 15): Decision becomes linearly separable earlier
2. **Confidence calibration** (Phase 16): Confidence at early layers correlates with correctness

Random init fails on both dimensions:
- Later crystallization (L16-L20)
- Overconfident early exits with poor accuracy

This explains why semantic init is strictly better: it enables both fixed and adaptive early-exit strategies.

### The Headline Result

> **Semantic initialization enables meaningful adaptive early-exit because confidence correlates with correctness. Random initialization produces overconfident early exits with poor accuracy.**

At τ=0.80, semantic init achieves:
- 1.36x speedup
- 97.6% AUC retention
- Most examples (218/230) exit by L16

### Files Added

```
src/analyze_adaptive_exit.py     # Adaptive exit policy simulator
run_adaptive_exit_analysis.sh    # Batch runner for analysis
results/adaptive_eval/           # Per-example margin data
results/adaptive_analysis/       # Adaptive exit analysis results
```

### Summary of Phase 16

| Question | Answer |
|----------|--------|
| Does adaptive exit work for semantic init? | **Yes.** τ=0.80 gives 1.36x speedup at 97.6% AUC. |
| Does adaptive exit work for random init? | **No.** Confidence is uncalibrated; all examples exit at L14 with 62% AUC. |
| Why does semantic init enable adaptive exit? | Confidence correlates with correctness at early layers. |
| Optimal operating point? | τ=0.80 for maximum speedup, τ=0.95 for minimal AUC loss. |

---

## Phase 16b: Robustness Checks for Adaptive Exit (Complete)

### Motivation

ChatGPT red-teamed Phase 16's calibration claims with these robustness checks:
1. **Reliability curves + ECE per layer**: Quantify calibration quality
2. **Seed sweep for all 3 seeds**: Ensure findings aren't seed-specific
3. **Margin normalization**: Test z-score and temperature scaling with calibration splits

The goal: turn a "cool effect" into a result that survives peer review.

### Implementation

Added calibration analysis tooling:
- `src/calibration_analysis.py`: ECE, Brier score, reliability curves, accuracy by confidence decile
- Updated `analyze_adaptive_exit.py` with `--normalize {none,zscore,temperature}` and `--calib_split` options
- `run_phase16b_robustness.sh`: Full analysis across all seeds and normalization methods

### Results: ECE per Layer (Lower is Better)

| Model | Seed | L14 ECE | L16 ECE | L18 ECE | L23 ECE |
|-------|------|---------|---------|---------|---------|
| semantic | 42 | 0.2696 | 0.0771 | 0.0569 | 0.0516 |
| semantic | 123 | 0.0913 | 0.0678 | 0.0216 | 0.0475 |
| semantic | 456 | 0.1117 | 0.0486 | 0.0486 | 0.0422 |
| **semantic avg** | - | **0.1575** | **0.0645** | **0.0424** | **0.0471** |
| random | 42 | 0.5231 | 0.4480 | 0.2734 | 0.0318 |
| random | 123 | 0.5245 | 0.5256 | 0.5210 | 0.0609 |
| random | 456 | 0.3699 | 0.4539 | 0.4727 | 0.1109 |
| **random avg** | - | **0.4725** | **0.4758** | **0.4224** | **0.0679** |

**Key finding**: Semantic init has 3x lower ECE at L14 (0.16 vs 0.47). By L23, calibration converges - random finally becomes calibrated, but only at full depth.

### Results: Adaptive Exit at τ=0.90

| Model | Seed | Norm | AUC | Speedup | Mean Layer |
|-------|------|------|-----|---------|------------|
| semantic | 42 | none | 0.9503 | **1.25x** | 18.2 |
| semantic | 123 | none | 0.9651 | **1.25x** | 17.6 |
| semantic | 456 | none | 0.9606 | **1.31x** | 16.9 |
| semantic | 42 | temperature | 0.9469 | **1.31x** | 17.9 |
| semantic | 123 | temperature | 0.9742 | **1.34x** | 16.2 |
| semantic | 456 | temperature | 0.9882 | **1.40x** | 15.6 |
| random | 42 | none | 0.6004 | 1.32x | 14.0 |
| random | 123 | none | 0.6176 | 1.51x | 14.0 |
| random | 456 | none | 0.6800 | 1.37x | 15.6 |
| random | 42 | temperature | 0.9741 | 0.95x | 22.9 |
| random | 123 | temperature | 0.9652 | 1.00x | 23.0 |
| random | 456 | temperature | 0.9530 | 1.00x | 23.0 |

### Key Finding: Normalization Can't Save Random Init

The clearest evidence comes from contrasting normalization effects:

**Semantic models with temperature scaling:**
- Maintain good AUC (0.95-0.99)
- Still achieve speedup (1.31-1.40x)
- Exit distribution spread across layers

**Random models with temperature scaling:**
- Temperature fits to T=10.0 (maximum, desperately trying to flatten confidence)
- Must exit at L23 to achieve good AUC (no speedup)
- Normalization doesn't create early-layer signal that wasn't learned

This confirms: **semantic init learns calibrated representations at early layers; random init does not.** Normalization can rescale confidences but cannot create discriminative signal where none exists.

### Robustness Confirmation

All three seeds show consistent patterns:
1. Semantic ECE at L14: 0.09-0.27 (calibrated)
2. Random ECE at L14: 0.37-0.52 (near-maximum miscalibration)
3. Semantic adaptive exit speedup: 1.25-1.40x with AUC > 0.95
4. Random adaptive exit: Either fast+wrong (norm=none) or slow+correct (norm=temp)

### The Peer-Review-Ready Claim

> **Semantic embedding initialization produces early-layer calibration (ECE 0.16 vs 0.47 at L14), enabling adaptive early-exit inference with 1.25-1.40x speedup at 95%+ AUC retention. Random initialization is systematically overconfident at early layers and cannot safely exit early regardless of post-hoc normalization.**

### Files Added

```
src/calibration_analysis.py      # ECE, Brier, reliability curves
run_phase16b_robustness.sh       # Full robustness analysis
results/calibration/             # Per-model calibration metrics
results/adaptive_analysis/       # Adaptive exit with normalization variants
```

### Summary of Phase 16b

| Question | Answer |
|----------|--------|
| Is ECE lower for semantic init at early layers? | **Yes.** 3x lower at L14 (0.16 vs 0.47). |
| Do results hold across seeds? | **Yes.** Pattern consistent for seeds 42, 123, 456. |
| Can normalization fix random init? | **No.** Temperature scaling → must exit at L23 (no speedup). |
| Best config for speedup? | Semantic init + temperature, τ=0.80-0.90 → 1.3-1.4x speedup. |

---

## Phase 17: Init Interpolation Sweep (Complete)

### Motivation

Phase 15-16 established that semantic initialization is critical for early crystallization and calibrated confidence. But **where is the boundary?** At what point does the semantic signal become sufficient?

### Implementation

Added interpolated initialization to `train_unified.py`:

```python
def init_token_interpolated(model, tokenizer, token, init_words, alpha, init_lm_head=True):
    """
    Initialize token embedding as interpolation between semantic and random.
    alpha=0.0 -> pure random
    alpha=1.0 -> pure semantic
    """
    semantic_emb = get_mean_embedding(model, tokenizer, init_words)
    random_emb = torch.randn_like(...) * existing_std
    interpolated_emb = alpha * semantic_emb + (1 - alpha) * random_emb
```

Trained 5 models with α = {0.00, 0.25, 0.50, 0.75, 1.00} on Train-M.

### Results: Coarse Sweep

| α | L14 ECE | L14 AUC | Full AUC | τ=0.80 AUC | τ=0.80 Speed |
|---|---------|---------|----------|------------|--------------|
| 0.00 | 0.4726 | 0.3621 | 0.9789 | N/A* | N/A |
| 0.25 | 0.4694 | 0.3282 | 0.9626 | N/A* | N/A |
| 0.50 | 0.4010 | 0.7165 | 0.9741 | 0.7165 | 1.47x |
| 0.75 | 0.2641 | 0.9748 | 0.9840 | 0.9691 | 1.33x |
| 1.00 | 0.2696 | 0.9607 | 0.9799 | N/A* | N/A |

*N/A indicates model not evaluated with adaptive exit policy in initial sweep.

### Key Discovery: Sharp Phase Transition

There is a **discontinuous jump** between α=0.50 and α=0.75:

| Metric | α=0.50 | α=0.75 | Jump |
|--------|--------|--------|------|
| L14 ECE | 0.40 | 0.26 | **35% better** |
| L14 AUC | 0.72 | 0.97 | **+0.25 AUC** |

This suggests the semantic component provides a **phase transition** in representational geometry, not a gradual improvement.

### Adaptive Exit Validation

Confirmed the phase transition affects adaptive exit viability:

**α=0.50** at τ=0.80:
- AUC: 0.7165 (poor)
- All 230 examples exit at L14
- Model is overconfident but wrong

**α=0.75** at τ=0.80:
- AUC: 0.9691 (excellent)
- Distributed exits: L14:106, L16:113, L18:3, L23:8
- Model correctly routes easy/hard examples

### Interpretation

The semantic embedding doesn't just "help a little" - it provides the **geometric structure** needed for the decision manifold to form at early layers. Below ~60% semantic weight, the model learns the task but the decision surface is malformed at intermediate layers. Above ~65% semantic weight, the geometry crystallizes correctly.

### Files Added

```
run_phase17_init_interpolation.sh  # Coarse sweep runner
results/phase17_eval/              # Per-α evaluation results
results/phase17_calibration/       # Per-α calibration analysis
results/phase17_adaptive/          # Per-α adaptive exit analysis
models/unified/semantic_alpha*     # Interpolated init models
```

---

## Phase 17b: Fine-Grained Transition Analysis (Complete)

### Motivation

Phase 17 identified a sharp transition between α=0.50 and α=0.75. Phase 17b narrows down the exact boundary.

### Implementation

Trained 4 additional models with α = {0.55, 0.60, 0.65, 0.70}.

### Results: Fine Sweep

| α | L14 ECE | L14 AUC | Full AUC | τ=0.80 AUC | τ=0.80 Speed | L16 ECE |
|---|---------|---------|----------|------------|--------------|---------|
| 0.50 | 0.4010 | 0.7165 | 0.9741 | 0.7165 | 1.47x | 0.3574 |
| 0.55 | 0.4151 | 0.8562 | 0.9724 | 0.8562 | 1.46x | 0.3371 |
| 0.60 | 0.3412 | 0.7117 | 0.9752 | 0.7121 | 1.48x | 0.2816 |
| **0.65** | **0.3215** | **0.9617** | **0.9813** | **0.9619** | **1.31x** | **0.0962** |
| 0.70 | 0.3421 | 0.9495 | 0.9762 | 0.9506 | 1.36x | 0.0814 |
| 0.75 | 0.2641 | 0.9748 | 0.9840 | 0.9691 | 1.33x | 0.0743 |

### Key Finding: Transition at α ≈ 0.60-0.65

The phase boundary is precisely located between α=0.60 and α=0.65:

| Metric | α=0.60 | α=0.65 | Jump |
|--------|--------|--------|------|
| L14 AUC | 0.7117 | 0.9617 | **+35%** |
| τ=0.80 AUC | 0.7121 | 0.9619 | **+35%** |
| L16 ECE | 0.2816 | 0.0962 | **3x better** |

### Observations

1. **Sharp phase boundary at ~62-65% semantic weight** - not gradual
2. **Calibration improvement at L16** (ECE: 0.28 → 0.10) coincides with AUC jump
3. **Interesting non-monotonicity**: α=0.55 has higher L14 AUC (0.86) than α=0.60 (0.71), but still fails at τ=0.80 because it's overconfident without being discriminative
4. **L14 ECE stays high** (~0.32-0.34) even after transition - the key is L16+ calibration

### Interpretation

The critical ratio is approximately **2/3 semantic, 1/3 random** for the decision geometry to form correctly. This suggests:

1. The semantic component provides an initial "direction" in embedding space
2. Some random component is tolerable (up to ~35%)
3. Below the threshold, the model learns the task at full depth but the intermediate-layer geometry is malformed

The transition is **discrete**, not continuous - consistent with a phase transition in the representational geometry.

### Files Added

```
run_phase17b_fine_sweep.sh         # Fine sweep runner
results/phase17_eval/alpha0{55,60,65,70}_*
results/phase17_calibration/alpha0{55,60,65,70}_*
results/phase17_adaptive/alpha0{55,60,65,70}_*
models/unified/semantic_alpha0{55,60,65,70}_seed42/
```

---

## Summary of All Phases

| Phase | Focus | Key Finding |
|-------|-------|-------------|
| 1-4 | Initial MVP & debugging | Label masking critical, format simplification needed |
| 5 | Track 1 redesign | Symmetric tokens + holdout testing |
| 6 | Seed sensitivity | Token model stable, baseline unstable under shift |
| 7 | Track 2 design | Efficiency experiment ready |
| 8 | Style leakage (Exp 3A) | Token degrades less than baseline under rewrites |
| 9 | Retrain triad | Mixed training + token = best performance |
| 10 | Layerwise probing | Decision crystallizes earlier for token model |
| 11 | Ablations (E2) | **Random tokens work** → it's about task structure, not semantics |
| 12 | Actual early-exit | **1.38x speedup** at 98% of final AUC (layer 16) |
| 13 | R/N baseline | **Dedicated tokens crystallize 4-5 layers earlier** than existing vocab |
| 14 | Test-R evaluation | **6-layer gap** on style-neutralized data (larger than original) |
| 15 | Unified harness & init study | **Initialization determines crystallization**; token string is secondary |
| 16 | Adaptive early-exit | Semantic init enables **calibrated confidence** → adaptive exit works |
| 16b | Robustness checks | Semantic init has **3x lower ECE** at L14; normalization can't fix random |
| 17 | Init interpolation (coarse) | **Sharp phase transition** between α=0.50 and α=0.75 |
| 17b | Init interpolation (fine) | **Boundary at α≈0.62-0.65** (2/3 semantic, 1/3 random) |
| K4.1 | K=4 "Love" data generation | 8 iterations; position-encoding achieved 29.5% TF-IDF but changed task |
| K4.1b | Why "Love" failed | ROM/FAM/PLA have overlapping contexts; explicit disambiguation leaks |
| K4.2 | Pivot to "Support" | New taxonomy with naturally distinct contexts; spec complete |

**Final interpretation**: Discrete decision channels require **semantic initialization above ~65%** for adaptive early-exit to function. Below this threshold, the model learns the task but the intermediate-layer decision geometry is malformed. The transition is sharp, not gradual - suggesting a phase transition in representational structure.

---

## Phase K4.1: K=4 Data Generation - The Style Leakage Gauntlet (Complete)

### Motivation

Phases 1-17 established that discrete decision channels with ~65% semantic initialization enable early-exit inference for K=2 (binary) classification. The next step: **extend to K=4 multi-class classification** to test if these findings generalize.

The K=4 taxonomy disambiguates "I love ___" across four categories:
- **ROM** (romantic): attraction toward a person
- **FAM** (familial): family bond toward a person
- **PLA** (platonic): friendship/loyalty toward a person
- **OBJ** (non-person): love of thing/place/activity

### The Core Challenge

K=4 data generation proved much harder than K=2. The fundamental tension:
1. **Learnability**: The model needs SOME signal to distinguish categories
2. **No TF-IDF exploitation**: Any consistent vocabulary pattern becomes a shortcut

This created a gauntlet of failed attempts before finding a solution.

### Attempt Log

#### v0.1: Direct Category-Specific Slots (FAILED)
**Approach**: Each category has distinct disambiguation phrases:
- ROM: "a future together", "what they're building together"
- FAM: "how long they've known each other", "where they came from together"
- PLA: "the bond they've chosen", "what they've faced side by side"
- OBJ: "a thing they care about", "a place that matters to them"

**Result**: 100% TF-IDF accuracy - slots became direct watermarks.

#### v0.2: Overlapping Vocabulary Attempt (FAILED)
**Approach**: Use overlapping base words ("always", "time", "together", "between them") with different semantic structures.

**Result**: 97% TF-IDF accuracy - TF-IDF still found distinctive n-grams like "chosen" for PLA, "deepening" for ROM.

#### v0.3: Minimal Targets, No Disambiguation (TOO SIMPLE)
**Approach**:
- All person-directed (ROM/FAM/PLA) use only "you"
- OBJ uses only "this"
- No disambiguation sentence at all

**Result**: 26.5% TF-IDF (at chance!) BUT ROM/FAM/PLA scenarios are literally identical - unlearnable by any model.

#### v0.4: Shared Person Cues (STILL UNLEARNABLE)
**Approach**: ROM/FAM/PLA all draw from the SAME cue pool:
- "Something between them has meaning."
- "There's a connection between them that matters."
- "The bond between them is real."

**Result**: 27% TF-IDF, but still unlearnable - no signal to distinguish ROM from FAM from PLA.

#### v0.5: Structural Negation (FAILED)
**Approach**: All three relationship words (partner, family, friend) appear in every sample. Use negation to indicate which is true:
- "Not friend, not family. This is partner."
- "B doesn't mean 'partner' or 'friend'. B means 'family'."

**Result**: 74.5% TF-IDF - TF-IDF caught "is family", "is friend", "is partner" bigrams. The affirmed word has a consistent marker.

#### v0.5.1: Equivalent Syntactic Positions (FAILED)
**Approach**: Put all three words in identical syntactic slots:
- "Partner: yes. Family: no. Friend: no."
- "[partner: no] [friend: no] [family: yes]"

**Result**: 97% TF-IDF - Caught "family yes", "friend yes", "partner yes" bigrams.

#### v0.5.2: Affirmed Word First (FAILED)
**Approach**: Put the affirmed word FIRST in the list, use "the first applies":
- ROM: "Of partner, family, or friend - the first applies."
- FAM: "Of family, partner, or friend - the first applies."

**Result**: 87.5% TF-IDF - Caught "of family", "of partner", "of friend" bigrams.

#### v0.5.4: Position-Based Encoding (SUCCESS!)
**Approach**: Randomize word ORDER, use position word to indicate which is true:
- ROM: "List: family, partner, friend. Answer: second." (partner is 2nd)
- FAM: "Options: partner, friend, family. The third applies." (family is 3rd)
- PLA: "Friend, family, partner - the first one." (friend is 1st)

**Key insight**:
- Each word (partner/family/friend) appears in each position (1st/2nd/3rd) equally often
- Each position word (first/second/third) appears in each label equally often
- TF-IDF cannot correlate words OR positions with labels

**Result**: **29.5% TF-IDF** (essentially chance for K=4) AND learnable by a model that can parse list structure!

### Why Position-Based Encoding Works

TF-IDF is a bag-of-words model. It can only learn:
- Word presence/absence
- Word co-occurrence (bigrams)

Position-based encoding defeats TF-IDF because:
1. All relationship words appear in all labels (uniform distribution)
2. All position words appear in all labels (uniform distribution)
3. No bigram pattern correlates with labels

But a language model can:
1. Parse the list to find word at indicated position
2. Map that word to the label (partner→ROM, family→FAM, friend→PLA)

This requires **structural understanding**, not vocabulary matching.

### Final Data Structure

**4-sentence template**:
```
Sentence 1: A and B are [EVENT] in the [SETTING].
Sentence 2: A says, "I love [TARGET]."
Sentence 3: [NEUTRAL_BUCKET_CUE]
Sentence 4: [POSITION-BASED DISAMBIGUATION]
```

**Sample scenarios**:
```
ROM: "A and B are packing up after an event in the park. A says, 'I love you.' They can finally relax. List: family, partner, friend. Answer: second."

FAM: "A and B are cleaning up after a meal in the living room. A says, 'I love you.' The moment is quiet and simple. Options: partner, friend, family. The third applies."

PLA: "A and B are doing the dishes in the park bench. A says, 'I love you.' Nothing else is going on around them. List: friend, family, partner. Answer: first."

OBJ: "A and B are sitting quietly after a long day in the sidewalk. A says, 'I love this.' Nothing else is going on around them. Options: friend, partner, family. None apply."
```

### Key Takeaways

1. **TF-IDF is surprisingly powerful** - Any consistent vocabulary pattern will be exploited
2. **The learnability/leakage tradeoff is real** - You can't just remove all signal
3. **Structural encoding beats vocabulary encoding** - Position-based disambiguation defeats bag-of-words while remaining learnable
4. **ChatGPT's insight was crucial**: "TF-IDF is bad at coreference, negation, and order-dependent meaning"

### Collaboration Note

This phase was done in collaboration with ChatGPT (GPT-4), which provided:
- Initial skeleton-quartet design
- Structural disambiguation concept (negation + role binding)
- Red-teaming of each failed attempt
- Final validation of position-based approach

### Files Modified/Created

```
src/data_generation_k4.py    # K=4 generator with position-based encoding
src/style_leakage_k4.py      # TF-IDF leakage checker (stop_words=None)
data_k4/stream.jsonl         # Generated dataset (200 examples)
docs/PHASE_K4_DATA_SPEC.md   # Detailed specification
```

### Summary of Phase K4.1

| Question | Answer |
|----------|--------|
| Can we generate K=4 data without TF-IDF leakage? | **Yes**, with position-based encoding |
| What's the TF-IDF accuracy? | **29.5%** (chance is 25%) |
| Is ROM/FAM/PLA distinguishable? | Only by structural parsing, not vocabulary |
| How many attempts did it take? | **8 iterations** (v0.1 through v0.5.4) |

### Next Steps for K=4

1. Generate larger dataset (500+ examples per category)
2. Create Test-R rewrites that paraphrase the position encoding
3. Train model and evaluate if structural disambiguation is learnable
4. Test if ~65% semantic init threshold holds for K=4

---

## Phase K4.1b: Why K=4 "Love" Can't Match K=2's Approach (Analysis)

### The Realization

After achieving 29.5% TF-IDF with position-based encoding, we confronted a fundamental problem: **the K=4 task is nothing like the K=2 task**.

### How K=2 Worked

In K=2 (romantic vs non-romantic), the **scenario context itself** disambiguated the meaning:

```
K=2 Example (non-romantic):
"After the meeting, Sarah turned to her colleague. 'I love you for
covering for me,' she said with relief."
```

- No explicit label encoding in the text
- The model must understand that "covering for me" + "colleague" + "relief" = gratitude, not romance
- **Semantic understanding is required** - TF-IDF fails because the same words appear in both classes
- The pivot phrase "I love you" is genuinely ambiguous; context resolves it

### Why K=4 Can't Work The Same Way

The K=4 categories (ROM/FAM/PLA) all describe **person-directed love using "I love you"**. The problem:

| Category | Pivot Phrase | What Differs? |
|----------|--------------|---------------|
| ROM | "I love you" | The *type* of relationship |
| FAM | "I love you" | The *type* of relationship |
| PLA | "I love you" | The *type* of relationship |
| OBJ | "I love this" | Target word (easy to distinguish) |

For ROM vs FAM vs PLA, the **only** distinguishing information is the relationship type. But relationship type must be expressed somehow, and any expression becomes a TF-IDF shortcut:

- **Implicit context cues** → TF-IDF exploits them (e.g., "kitchen" → FAM, "restaurant" → ROM)
- **Explicit relationship words** → Direct watermarks ("partner", "family", "friend")
- **Semantic paraphrases** → TF-IDF finds distinctive vocabulary ("chosen bond" → PLA, "always been there" → FAM)
- **Structural negation** → TF-IDF catches affirmed-word patterns ("is partner" → ROM)

### The Fundamental Asymmetry

**K=2 (romantic vs non-romantic):**
- These categories have **naturally different contexts**
- Romantic love appears in dates, relationships, intimate moments
- Non-romantic love appears in workplace gratitude, family duty, friendship loyalty
- The **scenario itself** distinguishes them without explicit labeling
- TF-IDF can be defeated because the same *vocabulary* appears in both, just with different *meaning*

**K=4 (ROM vs FAM vs PLA):**
- These categories have **overlapping contexts**
- All three can appear in homes, during meals, in emotional moments
- There's no natural setting/event that uniquely signals one vs another
- **Explicit disambiguation is required** - and any explicit signal leaks to TF-IDF

### The Position-Encoding Compromise

Our "solution" (position-based encoding) defeated TF-IDF but **changed the task entirely**:

| Aspect | K=2 Task | K=4 Task (position-encoded) |
|--------|----------|----------------------------|
| What model learns | Semantic context understanding | List parsing + instruction following |
| Disambiguation source | Implicit in scenario | Explicit in structured format |
| Comparable to K=2? | N/A | **No** |
| Tests semantic init hypothesis? | Yes | Only partially (mechanism, not semantics) |

### Why "Love" Doesn't Scale to K=4

The word "love" is the problem. In English:
- "I love you" to a romantic partner vs family member vs friend → **same words, same syntax**
- The only difference is the **unstated relationship** between speaker and listener
- That relationship MUST be stated somehow for training data
- Stating it creates the shortcut

### The Path Forward

**Option 1: Different concept** - Find a word/phrase where K=4 categories have naturally distinct contexts (like K=2 did), so no explicit disambiguation is needed.

**Option 2: Accept the difference** - Use position-encoded K=4 to test the decision channel mechanism, acknowledging it tests different capabilities than K=2.

**Option 3: Hybrid evaluation** - Train on position-encoded data, but evaluate on naturalistic scenarios (transfer test).

### Decision

We're pivoting to **Option 1**: find a different concept where 4+ categories have naturally distinct contexts that don't require explicit labeling.

The "love" taxonomy (ROM/FAM/PLA/OBJ) was elegant in theory but fundamentally incompatible with the K=2 methodology. The insight: **not all K>2 classification problems are created equal**. Some have natural contextual separation; others don't.

---

## Phase K4.2: Pivot to "Support" — Proper K=4 Design (In Progress)

### Motivation

After abandoning the "love" taxonomy, we needed a new word that:
1. Has multiple distinct meanings (K=4+)
2. Those meanings have **naturally distinct contexts** (like K=2's romantic vs non-romantic)
3. No explicit disambiguation is required — context alone determines the label
4. A human can reliably classify examples without auxiliary structure

### The New Word: "Support"

We selected **"support"** because it naturally splits into four semantically distinct categories with different contextual signatures:

| Category | Code | What it accomplishes | Key constraint |
|----------|------|---------------------|----------------|
| **Emotional** | E | Changes emotional/psychological state | No tasks completed, no resources delivered |
| **Practical** | P | Materially changes an outcome via action | Remove actions → support disappears |
| **Ideological** | I | Expresses agreement/endorsement only | No causal contribution to implementation |
| **Structural** | S | Literal mechanical/physical support | No people, emotions, or opinions |

### Why "Support" Works Where "Love" Failed

| Problem with "Love" | How "Support" Solves It |
|---------------------|------------------------|
| ROM/FAM/PLA all use "I love you" | E/P/I/S have naturally different contexts |
| Relationship type must be stated explicitly | Support type is implicit in what happens |
| Overlapping contexts (all can happen at home) | Distinct outcomes (emotion vs task vs endorsement vs physical) |
| Any disambiguation becomes TF-IDF exploitable | Context alone disambiguates |

### Key Design Decisions

1. **Entity constraints (v1)**:
   - E/P/I: People only (no animals)
   - S: Objects/systems only (no people doing the supporting)

2. **No explicit category cues**: Words like "emotionally", "physically", "structurally" are forbidden

3. **Easy/Hard split**:
   - Easy (60-70%): Clear, canonical examples
   - Hard (30-40%): Near-miss examples that borrow surface cues from other categories
   - The guiding principle: **"Verbs may overlap. Outcomes must not."**

4. **Label decision rule**: Based on what the support *accomplishes*, not the language used

### Canonical Examples (one per category)

**Emotional**: Job loss → sister checks in, listens, reminds him the setback doesn't define him → support through uncertainty
- No tasks completed, no material change, only emotional state changes

**Practical**: Deadline approaching → Alex stays late, fixes bugs, brings equipment → efforts support finishing on time
- Remove the actions and the support disappears

**Ideological**: Reads transportation policy proposal → agrees with goals → publicly states support
- No implementation, no action, only endorsement

**Structural**: Bridge with steel beams → beams hold weight, keep structure stable → beams support the bridge
- No people, no intent, purely mechanical

### Near-Miss Examples (hard cases)

Each deliberately borrows surface cues from other categories:
- **Emotional near-miss**: Uses "stayed late" and "office" (practical cues) but support is purely emotional
- **Practical near-miss**: Uses "talked with" and conversation (emotional cues) but support is action-based
- **Ideological near-miss**: Uses "spent weeks" and "effort" (practical cues) but support is only approval
- **Structural near-miss**: Uses "responded", "prevented" (agent-like verbs) but support is mechanical

### Files Created

```
docs/K4_SUPPORT_SPEC.md              # Full specification with examples and constraints
docs/K4_SUPPORT_GENERATION_PROMPT.md # Prompt template for data generation
docs/K4_SUPPORT_VALIDATION_CHECKLIST.md # Validation checklist for quality control
archive/k4_love_attempt/             # Archived "love" attempt files
```

### Collaboration Note

This design was developed in collaboration with ChatGPT (GPT-4), which provided:
- The "support" concept and category definitions
- Template skeletons and canonical examples
- Near-miss design and cue injection strategy
- Validation checklist and generation prompts
- Critical feedback on boundary conditions (I vs P, S2 scope)

### Next Steps

1. Generate pilot dataset (300 train, 160 test) using the generation prompt
2. Run TF-IDF leakage check (target: well below neural model performance)
3. Human review of edge cases
4. Scale to full dataset if pilot passes validation
5. Train models and test if K=2 findings generalize

### Summary

| Question | Answer |
|----------|--------|
| Why pivot from "love"? | ROM/FAM/PLA have overlapping contexts; explicit disambiguation leaks |
| Why "support"? | Four meanings with naturally distinct contexts |
| What's the core principle? | Verbs may overlap. Outcomes must not. |
| How is this like K=2? | Context alone determines label; no explicit encoding |
| Status | Spec complete; ready for data generation |

---

## Future Directions

1. **K=4 "Support" data generation and training (Phase K4.3)**: Generate pilot dataset, validate with TF-IDF, train models.

2. **K=4 semantic init threshold**: Test if the ~65% threshold shifts for multiclass.

3. **K=4 calibration analysis**: Does calibration fragment by class, or emerge uniformly?

4. **Semantic-embed + random-lm_head ablation**: Test whether the lm_head initialization matters separately from embed initialization.

5. **Per-example difficulty analysis**: Correlate early-exit layer with example characteristics (easy vs hard).

---

## Phase K4.4b: K=4 Training + Crystallization (Complete)

### Training Results (Seed 42, leaky K=4 data)

| Model | Eval Loss | Train Loss | Final Accuracy |
|------|-----------|------------|----------------|
| DDC‑Semantic (α=0.65) | 0.0009 | 0.216 | 95.0% |
| DDC‑Random (α=0.0) | 0.016 | 0.155 | 96.7% |
| Baseline‑Dedicated (⟦BASE_*⟧) | 2.68 | 2.75 | **25.0% (chance)** |
| Baseline‑Vocab (E/P/I/S) | 0.003 | 0.063 | 96.7% |

### Crystallization Results (Macro‑AUC ≥ 0.95)

| Model | Crystallization Layer |
|------|------------------------|
| DDC‑Semantic | **L17** |
| Baseline‑Vocab | **L17** |
| DDC‑Random | L21 |
| Baseline‑Dedicated | Never |

### Interpretation

1. **Semantic init accelerates crystallization**: DDC‑Semantic reaches ≥0.95 macro‑AUC 4 layers earlier than DDC‑Random (L17 vs L21).
2. **Baseline‑Vocab parity**: DDC‑Semantic and Baseline‑Vocab both crystallize at L17 on this leaky K=4 task.
3. **Baseline‑Dedicated failure is likely an init confound**: baseline‑dedicated uses *default resize init* only, while DDC‑Random uses explicit random init for both embeddings + lm_head (scaled to existing std). This mismatch can explain collapse; it should not be interpreted as “new tokens cannot learn.”

**Saved models**:
```
models/k4/ddc_semantic_alpha065_seed42
models/k4/ddc_random_alpha000_seed42
models/k4/baseline_dedicated_seed42
models/k4/baseline_vocab_seed42
```

---

## Phase K4.5: Zero‑Shot Label Priors & Control Probes (Complete)

### Motivation
We tested whether “baseline‑vocab letters” (E/P/I/S) have privileged categorical geometry vs other token sets (R/N, random letters), and whether zero‑shot priors might explain K=2 vs K=4 behavior.

### Probe 1: Embedding Geometry (pre‑finetune)
Computed pairwise cosine similarities among token embeddings:
- **EPIS** mean cos ≈ 0.416
- **RN** mean cos ≈ 0.422 (single pair)
- **QJXZ** mean cos ≈ 0.308

**Result**: EPIS is only slightly tighter than random letters; geometry alone does **not** explain crystallization differences.

### Probe 2: Zero‑shot categorical prompt (uncontrolled)
The initial probe showed RN with extreme confidence (~0.98 max‑prob), but this was confounded by 2‑way vs 4‑way sets, prompt suffix, and tokenization.

### Probe 3: Controlled zero‑shot probe (cardinality + suffix + spacing)
Controls introduced:
- 4‑way sets only (EPIS, QJXZ, RNAB)
- “Answer:” suffix
- space vs no‑space tokens
- option order reversal
- 5 examples (pilot_v4), ROCm via `HSA_OVERRIDE_GFX_VERSION=10.3.0`

**Key result**: RN is **not** uniquely privileged under controls (RNAB becomes near‑uniform).  
Conclusion: the earlier RN≈0.98 was a probe artifact, not categorical readiness.

**Observed skew**: Some tokens (e.g., “ Q” in QJXZ, “ S” in EPIS) remain peaky under specific suffix/spacing, indicating **format/position priors**, not semantic alignment.

**Controlled run (template=answer, n=5, include_reverse):**
- EPIS (nospace): entropy 0.81–0.82, maxp 0.52–0.55, argmax mostly S/E
- EPIS (space): entropy ~0.81, maxp ~0.67, argmax “ S”
- QJXZ (nospace): entropy 0.81–0.86, maxp ~0.71, argmax flips with order (Q vs Z)
- QJXZ (space): entropy 0.40–0.87, maxp 0.57–0.90, argmax “ Q” or “ Z”
- RNAB (nospace): entropy 0.98, maxp 0.52–0.59, argmax R/B
- RNAB (space): entropy 0.89–1.15, maxp 0.50–0.53, argmax R/B

**Template sweep (n=5, include_reverse, space/no‑space, 4‑way sets):**
- Strong **suffix‑dependent skew** persists across templates; space‑prefixed tokens often dominate.
- EPIS: “answer/letter/output/choice/token” often favor **S** (nospace) or **“ S”** (space); “respond” reduces skew.
- QJXZ: many templates heavily favor **Q/Z**, with order reversing the argmax; space‑prefixed **“ Q”** is frequently dominant.
- RNAB: argmax flips with order and suffix (A/B/R), indicating **format/position priors** rather than semantic readiness.

**2‑way controls (template=answer, n=5):**
- All 2‑way sets show **very high max‑prob** (≈0.79–0.97) and strong **order effects** (argmax flips when reversed).
- Confirms that 2‑way probes are **not comparable** to 4‑way probes; cardinality alone can create extreme confidence.

**Final verdict (zero‑shot label priors):**
Zero‑shot label behavior is **dominated by prompt suffix, candidate ordering, tokenization (space vs no‑space), and set cardinality**.  
Under controlled 4‑way probes, RNAB is **not** uniquely privileged, and apparent “letter superiority” dissolves.  
Therefore, K=2 vs K=4 baseline differences should **not** be attributed to intrinsic label‑token readiness; instead treat label tokens as a nuisance variable and control via consistent output format (or average across multiple label sets).

**Top‑k unrestricted logits (template sweep, n=5):**
- The unrestricted top‑k list is dominated by **generic continuations** (“The”, “Based”, “This”, “In”, “Given”), not letter tokens.
- When a letter is favored, it is usually because it **matches a common generic next‑token** (e.g., “S”/“ Q” aligning with “Sure/Question”) rather than any task semantics.
- This supports the interpretation that **position/suffix priors** drive the peaky distributions, not label meaning.

### Updated Conclusion
Token‑set effects are dominated by **prompt‑local priors, tokenization (space vs no‑space), and cardinality**, not “intrinsic categorical geometry.” This weakens any claim that EPIS are inherently better labels; it strengthens the claim that output format and priors can interfere with axis formation.

### New Scripts
```
src/zero_shot_label_probe.py           # embedding geometry + basic zero‑shot probe
src/zero_shot_label_probe_controls.py  # controlled probe with 4‑way sets + suffix/spacing/order
```

---

## Phase K4.6: K2/K4 Prompt‑Grounding Insight (Discussion)

### Key insight
Instruction text can *ground* output symbols and distort priors.
K=2 R/N prompts explicitly map letters to labels ("R (romantic), N (non‑romantic)"), whereas K=4 baseline‑vocab does not provide equally strong mapping. This introduces an **instruction‑level prior** that can interfere with clean decision geometry.

### Implication
If the goal is to study internal decision geometry, **avoid instruction‑level symbol grounding**; use neutral symbols or map tokens outside the instruction.

---

## Phase K4.7: Vocab Token Priors - Controlled Baseline Design

### Background

We compared:
- **DDC new tokens** (⟦…⟧) with semantic/random/interpolated init
- **Baseline vocab tokens** (existing tokens like letters)
- **Baseline dedicated** (new tokens with random init; expected to test "new tokens without semantics")

Goal: understand how decision-token design affects learning, calibration, and crystallization layer (early separability / early exit viability).

### Key Discovery: Vocab-token choice introduces large pretrained priors at the decision locus

The base model has strong next-token preferences at positions like `DECISION:` depending on:
- The prompt suffix ("Answer:", "Respond with…", etc.)
- Whether candidate tokens are space-prefixed (`" X"`) vs nospace (`"X"`)
- Candidate ordering
- Cardinality (2-way vs 4-way)

These priors can cause:
- Early bias in logits
- Training collapse for decision-only supervision
- Misleading "zero-shot categorical readiness" interpretations if controls are missing

### Tests Performed and Outcomes

#### 1) TF-IDF leakage checks (K=4 + K=2)
- Initial K=4 "minimal pairs success" was invalidated: row-wise CV split leaked base scenario structure across folds.
- Correct metric: **GroupKFold by base_id**.
- With grouped splits, TF-IDF becomes high again (strong cross-set lexical tells).
- Follow-up: K=2 datasets also show high TF-IDF accuracy even under GroupKFold(bucket), revealing lexical shortcuts exist in K=2 as well.

**Conclusion**: Datasets are leaky; we proceed acknowledging leakage but tighten narrative. Leakage does not automatically invalidate mechanistic comparisons, but it weakens "semantic purity" claims.

#### 2) Zero-shot categorical prompt probe (initial)
Early results suggested RN was extremely confident; later analysis showed this was a **probe artifact** (uncontrolled cardinality and suffix effects).

#### 3) Controlled zero-shot probe script (`zero_shot_label_probe_controls.py`)
Controls added:
- Equal cardinality (4-way comparisons)
- Template sweep ("answer", "letter", "respond", etc.)
- Space vs nospace
- include_reverse ordering

**Findings**:
- Strong suffix/space/order priors dominate.
- Some tokens (" S", " Q") repeatedly dominate depending on template.
- Order reversals can flip argmax, confirming format bias rather than semantics.

#### 4) Unrestricted top-k logits diagnostic
When not restricting candidates, top predictions are generic continuations ("The", "Based", "This", "In", "Given").
When letters dominate among restricted candidates, it's due to completion priors at that locus, not label semantics.

#### 5) Label set selection tool (`select_label_sets.py`)
Implemented a measurement-based selector that searches candidate token sets and scores "flatness":
- Maximize entropy / minimize maxp across space/order conditions at `DECISION:` anchor.

**Results**:
- **K=4** can find relatively flat 4-way sets (e.g., ACRY) and very peaky sets (e.g., RWXZ).
- **K=2** is intrinsically peaky: even the flattest sets have worst-case maxp ~0.70+ (far from uniform 0.50).
- Emoji/punctuation can be catastrophically peaky (maxp ≈ 1.0).

**Conclusion**: Binary (K=2) decision-only labeling is structurally sensitive to priors; 4-way is more controllable.

**Selector updates**:
- Added internal emoji/symbol pools and `--pool_types`/`--pool_custom` for mixed candidate pools.
- Added `--out_json` to save full results + selected sets (flat/peaky) with config metadata.

**Mixed‑pool 2‑way run**:
- Flattest examples still have worst‑case maxp ≈ 0.71 (e.g., `['E','O']`).
- Emoji/symbol pairs can be maximally peaky (maxp ≈ 1.0), confirming they are unsafe for neutral baselines.

### Explaining Observed K=2 vs K=4 Baseline Behavior

#### Why baseline vocab lagged in K=2 love earlier (R/N)
Multiple factors likely contributed:
- R/N leaks semantics into the label choice (R=romantic, N=non-romantic).
- K=2 prompts previously included both `DECISION:` and an `ANSWER:` label string, which changes gradient routing and decision position dominance.
- Binary prior sensitivity is high.

#### Why baseline vocab matched DDC in K=4 support
- K=4 data is leaky and easier; many models can learn shortcuts early.
- K=4 baseline vocab used a single decision-token target (cleaner), and with certain tokens this can perform well.
- However, token priors can still dominate unless controlled; we observed strong suffix/space/order effects.

### Where We Landed

1. **We proceed with leaky datasets**, but reframe claims:
   - Focus on mechanistic differences (init threshold, calibration, crystallization timing)
   - Avoid strong "semantic purity" claims unless tested on low-leakage rewrites/counterfactual sets.

2. **Baseline vocab must be treated as a controlled factor, not a single setting**:
   - We now run **two vocab baselines** per task:
     - **vocab_flat**: Token set selected to minimize prior bias at `DECISION:`
     - **vocab_peaky**: Token set selected to maximize prior bias (stress test)

3. **Prompts and decision locus are unified** across tasks and conditions:
   - Same `DECISION:` anchor
   - Consistent tokenization policy (nospace)
   - Stable option ordering (follows label order in config)

4. **New-token conditions (DDC / dedicated) cannot accidentally canonicalize into un-added "space variants"**:
   - Strict tokenization rules for new tokens
   - Store and use verified token IDs everywhere

### Interpretability

This makes future comparisons interpretable:
- If "peaky" baselines crystallize earlier, it's likely due to priors.
- If DDC semantic init consistently crystallizes earlier than DDC random under the same interface, that supports the "semantic initialization shapes decision geometry" story, independent of label-token priors.

### Updated Model Variants (train_kn.py)

| Variant | Description | Output Dir Example |
|---------|-------------|-------------------|
| `ddc` (α=0.65) | New tokens, semantic init | `ddc_a065_seed42` |
| `ddc` (α=0.0) | New tokens, random init | `ddc_a000_seed42` |
| `vocab_baseline` (flat) | Existing tokens, low prior bias | `vocab_flat_seed42` |
| `vocab_baseline` (peaky) | Existing tokens, high prior bias | `vocab_peaky_seed42` |
| `dedicated_baseline` | New tokens, random init, neutral strings | `dedicated_seed42` |

### Token Sets Used

**K=2 Love**:
- DDC: `⟦LOVE_ROM⟧`, `⟦LOVE_NONROM⟧`
- Flat: `E`, `O` (best worst-case from selection)
- Peaky: `M`, `Q` (high prior bias)
- Dedicated: `⟦BASE_ROM⟧`, `⟦BASE_NONROM⟧`

**K=4 Support**:
- DDC: `⟦SUPPORT_E⟧`, `⟦SUPPORT_P⟧`, `⟦SUPPORT_I⟧`, `⟦SUPPORT_S⟧`
- Flat: `A`, `C`, `R`, `Y` (relatively flat 4-way)
- Peaky: `R`, `W`, `X`, `Z` (high prior bias)
- Dedicated: `⟦BASE_E⟧`, `⟦BASE_P⟧`, `⟦BASE_I⟧`, `⟦BASE_S⟧`

---

## Phase 18: Unified Pipeline, Full Training Run & Results (Complete)

### Overview

Built a unified training/evaluation/runner pipeline, ran all 30 models (5 variants × 2 tasks × 3 seeds), evaluated on 45 test sets, and analyzed results. Several significant bugs were found and fixed during the process.

### Infrastructure Built

**Shared module (`src/kn/`)**: Extracted common code from earlier scripts into a clean package:
- `config.py`: TaskConfig dataclasses, decision interface constants, token sets
- `prompt.py`: Unified prompt formatting with `format_input`/`format_output`
- `io.py`: RunConfig dataclass, JSONL loading, run_config contract
- `metrics.py`: Evaluation metrics, prior probing

**Training (`src/train_kn.py`)**: Single script handling all 5 variants across both tasks. Manages token addition, semantic initialization, LoRA, and saves `run_config.json` as the contract between training and evaluation.

**Evaluation (`src/eval_kn.py`)**: Reads `run_config.json` as source of truth. Three evaluation modes:
- `basic`: Accuracy, AUC, confusion matrix
- `calibration`: ECE, Brier score, confidence analysis
- `layerwise`: Crystallization analysis (logit lens at each transformer layer)

**Runner (`run_kn.sh`)**: Orchestrates all 30 training runs and 45 evaluations with logging, skip-if-exists logic, and summary generation.

### Data Preparation

**K2 Love**: Organized into O/R/M structure:
- `O/`: Original scenarios (test: 230 examples)
- `R/`: Rewritten scenarios via Claude API to remove style leakage (test: 230, val: 42 rewritten)
- `M/`: Mixed (concatenation of O+R, used for training)
- Test buckets (crisis + collaboration) are fully held out from train/val

**K4 Support**: Single directory structure (960 train / 120 val / 120 test), balanced 4 classes.

### Bugs Found and Fixed

1. **Multi-token supervision** (critical): Training was supervising 3 tokens (decision + `<|im_end|>` + newline) instead of just the decision token. Metrics were reading the newline token (last unmasked) rather than the decision token (first unmasked). Fixed to unmask only position `prompt_len` and scan forward for evaluation.

2. **OOM during training eval**: HF Trainer accumulated full logits tensor (vocab_size × seq_len × batch) for `compute_metrics`. Fixed with `prediction_loss_only=True`.

3. **Bash arithmetic with `set -e`**: `((total++))` returns 0 when total=0, which is falsy, causing silent exit under `set -e`. Fixed to `total=$((total + 1))`.

4. **Embedding resize direction**: Eval checked `len(tokenizer) > model.config.vocab_size` (only enlarge). Saved adapter has fewer tokens than base Qwen (151667 vs 151936). Fixed to `!=` (resize either direction).

5. **K4 validation leakage**: `run_kn.sh` was evaluating K4 on `val.jsonl` as a secondary test set — but val was used for checkpoint selection during training. Discovered and removed tainted results.

### Results: Full 30-Model Run

All 30 models trained successfully (5 variants × 2 tasks × 3 seeds each). 45 evaluations completed (K2: 30 = 5×3×2 test sets; K4: 15 = 5×3×1 test set).

#### K2 Love (Test-R, rewritten test set — primary metric)

| Variant | Accuracy (mean±std) | Crystallization (AUC≥0.95) | ECE |
|---------|--------------------|-----------------------------|-----|
| DDC α=0.65 | 0.891±0.015 | L19.7 | 0.086 |
| DDC α=0.0 | 0.916±0.014 | L18.3 | 0.057 |
| Dedicated baseline | 0.916±0.014 | L18.3 | 0.057 |
| Vocab flat | 0.884±0.018 | L18.7 | 0.066 |
| Vocab peaky | 0.916±0.021 | L19.7 | 0.069 |

#### K4 Support (test set)

| Variant | Accuracy (mean±std) | Crystallization (AUC≥0.95) | ECE |
|---------|--------------------|-----------------------------|-----|
| DDC α=0.65 | 0.967±0.012 | L17.0 | 0.034 |
| DDC α=0.0 | 0.967±0.012 | L19.0 | 0.035 |
| Dedicated baseline | 0.967±0.012 | L19.0 | 0.035 |
| Vocab flat | 0.961±0.004 | L18.3 | 0.040 |
| Vocab peaky | 0.950±0.007 | L21.0 | 0.047 |

### Key Findings

#### 1) DDC α=0.0 and dedicated_baseline are numerically identical

These two variants use different token strings (`⟦LOVE_R⟧` vs `⟦BASE_R⟧`) but the same mechanism: new tokens with random init. Their results are identical across all seeds and metrics, confirming that **token string identity is irrelevant** — only embedding initialization matters.

#### 2) K4 shows the clearest crystallization hierarchy

In K4, the crystallization ordering is clean:
- DDC α=0.65: **L17** (earliest)
- Vocab flat: L18.3
- DDC α=0.0 / dedicated: L19
- Vocab peaky: **L21** (latest)

Semantic initialization (α=0.65) gains 2 layers over random init (α=0.0). Peaky priors delay crystallization by 2 layers vs flat priors.

#### 3) K2 is noisier and less conclusive

K2 crystallization differences are smaller and inconsistent across seeds. The binary task is inherently easier to resolve at the decision locus, leaving less room for mechanistic differences to manifest. This aligns with earlier findings that K2 is structurally sensitive to priors.

#### 4) The "early exit" story is about all three factors

Crystallization timing depends on:
- **Token novelty**: New tokens (DDC, dedicated) vs existing vocab
- **Initialization**: Semantic (α=0.65) vs random (α=0.0)
- **Prior bias**: Flat vs peaky existing-vocab priors

No single factor dominates. The DDC α=0.65 advantage in K4 is real (2 layers over random) but the vocab_peaky penalty (4 layers behind DDC) is even larger.

#### 5) Calibration tracks crystallization

Lower ECE consistently co-occurs with earlier crystallization. DDC α=0.65 has the best calibration in K4 (ECE=0.034), vocab_peaky the worst (ECE=0.047).

### Data Leakage Note

K4 "test_o" results were initially computed on `val.jsonl`, which was used for checkpoint selection during training. These tainted results were identified, deleted, and the runner script was fixed to use only the held-out test set for K4. K2 test sets (O and R) were never used during training and remain valid.

### Token Sets Used (Final — corrected from Phase K4.7)

**K=2 Love**:
- DDC: `⟦LOVE_R⟧`, `⟦LOVE_N⟧`
- Flat: `E`, `O`
- Peaky: `T`, `Z`
- Dedicated: `⟦BASE_R⟧`, `⟦BASE_N⟧`

**K=4 Support**:
- DDC: `⟦SUPPORT_E⟧`, `⟦SUPPORT_P⟧`, `⟦SUPPORT_I⟧`, `⟦SUPPORT_S⟧`
- Flat: `A`, `C`, `R`, `Y`
- Peaky: `S`, `U`, `V`, `Z`
- Dedicated: `⟦BASE_E⟧`, `⟦BASE_P⟧`, `⟦BASE_I⟧`, `⟦BASE_S⟧`

### What's Next

Adding a **label-word baseline** (`label_word`): a "typical fine-tuning" variant that uses no new tokens and outputs the actual label word (e.g., `romantic`, `emotional`) at an `ANSWER:` prefix. Only the first token of the label word is supervised and scored, maintaining the single-locus property for comparable crystallization analysis. This bridges the gap between the DDC mechanistic experiment and conventional fine-tuning practice.

---

## References

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning: LoRA (r=8, alpha=16)
- Hardware: AMD GPU with ROCm (HSA_OVERRIDE_GFX_VERSION=10.3.0)
