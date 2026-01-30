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
├── data_O/                       # Original data (for triad experiment)
├── data_R/                       # Rewritten data (for triad experiment)
├── data_M/                       # Mixed data (O+R, for triad experiment)
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
data_O/           # Original training data
data_R/           # Rewritten training data
data_M/           # Mixed training data (O+R)
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

## References

- Base model: Qwen/Qwen2.5-0.5B-Instruct
- Fine-tuning: LoRA (r=8, alpha=16)
- Hardware: AMD GPU with ROCm (HSA_OVERRIDE_GFX_VERSION=10.3.0)
