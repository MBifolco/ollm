# K=4 "Support" Training Pipeline Specification

## Overview

Train discrete decision channel (DDC) models on the K=4 "support" polysemy task and analyze crystallization dynamics under realistic (leaky) language conditions.

**Primary research question**: How does decision geometry behave when lexical shortcuts exist, and can DDCs provide earlier, better-calibrated commitment?

---

## 1. Dataset

### Source
- **File**: `data/k4_support/train.jsonl`
- **Total examples**: 1200
- **Structure**: 300 minimal-pair base sets × 4 variants (E/P/I/S)

### Label Distribution
| Category | Code | Count | Description |
|----------|------|-------|-------------|
| Emotional | E | 300 | Support changes psychological/emotional state |
| Practical | P | 300 | Support consists of actions/resources that materially change outcome |
| Ideological | I | 300 | Support is agreement/approval only, no causal contribution |
| Structural | S | 300 | Support is literal/mechanical/systemic (non-human target) |

### Difficulty Distribution
- **Hard (near-miss)**: 852 examples (71%)
- **Easy (canonical)**: 348 examples (29%)

### Known Leakage (Accepted)
- **TF-IDF accuracy (GroupKFold by `base_id`)**: 93.0% ± 0.6%
- **Evaluation method**: GroupKFold only (StratifiedKFold is invalid for minimal-pair data)
- **Signature words**: E→"felt/calmer", I→"knew/position", P→"completed", S→"remained/stable"
- **Interpretation**: Leakage is a property of natural language; comparative findings remain valid

---

## 2. Data Splits

### Strategy: Group-aware by `base_id`

**Critical**: Never split variants from the same base across train/val/test. All 4 variants of a base must go together.

### Proposed Split (by base_id)
| Split | Base IDs | Examples | Purpose |
|-------|----------|----------|---------|
| Train | (shuffled) | 960 (80%) | Model training |
| Val | (shuffled) | 120 (10%) | Hyperparameter tuning, early stopping |
| Test | (shuffled) | 120 (10%) | Final evaluation |

### Implementation
```python
import random

# Shuffle base_ids to avoid generation-order bias
# (base IDs were generated sequentially by domain, which could introduce structure)
all_base_ids = list(range(300))
random.seed(42)
random.shuffle(all_base_ids)

# Split shuffled base_ids
train_bases = set(all_base_ids[:240])   # 80%
val_bases = set(all_base_ids[240:270])  # 10%
test_bases = set(all_base_ids[270:])    # 10%

train_data = [ex for ex in data if ex["base_id"] in train_bases]
val_data = [ex for ex in data if ex["base_id"] in val_bases]
test_data = [ex for ex in data if ex["base_id"] in test_bases]
```

**Note**: Shuffling prevents "generation order bias" — base IDs were created sequentially by domain/difficulty, so contiguous splits could introduce unintended structure.

---

## 3. Model Architecture

### Base Model
- **Model**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Parameters**: ~500M
- **Tokenizer**: Qwen2 tokenizer (extended with decision tokens)

### Decision Tokens
Four new tokens added to vocabulary:

| Token | Meaning | Initialization |
|-------|---------|----------------|
| `⟦SUPPORT_E⟧` | Emotional support | Semantic (from "emotional") |
| `⟦SUPPORT_P⟧` | Practical support | Semantic (from "practical") |
| `⟦SUPPORT_I⟧` | Ideological support | Semantic (from "ideological") |
| `⟦SUPPORT_S⟧` | Structural support | Semantic (from "structural") |

### Initialization Strategy
Based on K=2 findings (α ≈ 0.65 threshold):

```python
# Semantic initialization with α interpolation
alpha = 0.65  # or sweep [0.5, 0.65, 0.75, 1.0]
semantic_embed = get_embedding("emotional")  # or practical/ideological/structural
random_embed = torch.randn_like(semantic_embed)
token_embed = alpha * semantic_embed + (1 - alpha) * random_embed
```

### Embedding Geometry Logging (Important for Interpretability)
Log the following at initialization time:

```python
# For each decision token, log:
# 1. The source token(s) used for semantic seed
# 2. Embedding norms
# 3. Pairwise cosine similarities between semantic seeds

semantic_seeds = {
    "E": get_embedding("emotional"),
    "P": get_embedding("practical"),
    "I": get_embedding("ideological"),
    "S": get_embedding("structural"),
}

# Log cosine similarity matrix (E vs P, E vs I, etc.)
# This matters because "emotional" vs "ideological" may be closer than expected
# and could affect class-conditional crystallization patterns
```

**Why this matters**: For K=4, the geometry of semantic seeds may influence class-conditional behavior. If two seeds are very similar, their corresponding classes may crystallize together.

### Fine-tuning Method
- **Method**: LoRA (Low-Rank Adaptation)
- **Rank (r)**: 8
- **Alpha**: 16
- **Target modules**: q_proj, v_proj (attention layers)
- **Dropout**: 0.05

---

## 4. Training Configuration

### Input Format
```
<|im_start|>user
Classify the type of support in this scenario:

{scenario}

Respond with your classification token.<|im_end|>
<|im_start|>assistant
⟦SUPPORT_{label}⟧<|im_end|>
```

**Critical for fair comparison**: Baseline-vocab model must use the **exact same prompt format**, just with different output tokens ("Emotional" instead of "⟦SUPPORT_E⟧"). Do not let baseline drift into verbose answers or extra text — this would confound the comparison.

### Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning rate | 2e-4 | With linear warmup |
| Warmup steps | 100 | ~10% of training |
| Batch size | 8 | Effective batch size |
| Gradient accumulation | 4 | If memory constrained |
| Epochs | 10 | With early stopping |
| Max sequence length | 512 | Sufficient for scenarios |
| Optimizer | AdamW | β1=0.9, β2=0.999 |
| Weight decay | 0.01 | Standard |
| Early stopping patience | 3 | Based on val loss |

### Hardware
- **GPU**: AMD GPU with ROCm
- **Environment variable**: `HSA_OVERRIDE_GFX_VERSION=10.3.0`

---

## 5. Model Variants to Train

### Primary Comparison
| Model | Decision Tokens | Init | Purpose |
|-------|-----------------|------|---------|
| **DDC-Semantic** | ⟦SUPPORT_E/P/I/S⟧ | α=0.65 | Primary model |
| **DDC-Random** | ⟦SUPPORT_E/P/I/S⟧ | α=0.0 | Init ablation |
| **Baseline-Dedicated** | ⟦BASE_E/P/I/S⟧ | N/A | Dedicated tokens, no semantic init |
| **Baseline-Vocab** | Single existing tokens | N/A | Existing vocab only |

### Baseline Tokenization Control (Important)
For fair comparison, baseline labels must be **single tokens** to avoid multi-token variance confounding crystallization depth:

| Model | Output Tokens | Notes |
|-------|---------------|-------|
| DDC-Semantic | `⟦SUPPORT_E⟧` | New token, semantic init |
| DDC-Random | `⟦SUPPORT_E⟧` | New token, random init |
| Baseline-Dedicated | `⟦BASE_E⟧` | New token, default init (isolates "dedicated tokens" effect) |
| Baseline-Vocab | `E` or `emotional` | Must verify single-token in Qwen tokenizer |

**Rationale**: This mirrors K=2 Phase 13 (R/N baseline) which isolated the "dedicated tokens" effect from semantic initialization.

### Optional: Init Sweep
If time permits, prioritize α ∈ {0.0, 0.65, 1.0} to verify K=2 threshold generalizes (skip intermediate values initially).

---

## 6. Evaluation Metrics

### Primary Metrics

#### 6.1 Classification Performance
- **Accuracy**: Overall and per-class
- **Macro F1**: Balanced across classes
- **Confusion matrix**: E/P/I/S patterns

#### 6.2 Layerwise Crystallization
For each layer L ∈ {0, 4, 8, 12, 16, 20, 24}:
- **Linear probe**: Multiclass logistic regression (softmax) on hidden states at layer L
- **Probe position**: Extract hidden states at the **decision token position only** (last token before generation)
- **Metric**: **Macro-AUC** (one-vs-rest, averaged across E/P/I/S)
- **Crystallization depth**: First layer where macro-AUC > 0.85 **for 2+ consecutive layers**

**K=4 adjustments**:
- AUC > 0.9 is too strict for 4-class; use **0.85** as threshold
- Report both macro-AUC and per-class AUC to detect class-conditional patterns
- Stability criterion: require sustained performance across consecutive layers

**Stability criterion**: Single-layer AUC spikes can happen in leaky regimes. Require threshold to be sustained before declaring crystallization.

#### 6.3 Calibration
- **ECE (Expected Calibration Error)**: By layer
- **Reliability diagrams**: Confidence vs accuracy
- **Class-conditional ECE**: Does calibration differ by E/P/I/S?

**Critical comparison**: Early-layer ECE (L8-L12) between DDC-semantic, DDC-random, and baseline. This is where leakage skeptics will look first. If calibration separates at early layers, the core thesis is strongly supported.

#### 6.4 Early Exit Analysis
For each candidate exit layer:
- **Exit accuracy**: If we stop at layer L, what's accuracy?
- **Accuracy retention**: % of final accuracy retained
- **Depth fraction**: L / total_layers (proxy for speedup)
- **Measured latency** (if feasible): Actual inference time at layer L vs full model
- **Confidence-conditioned accuracy**: Accuracy retention *at a given confidence threshold*

**Speedup note**: K=2 showed that theoretical speedup (depth fraction) differs from actual speedup due to overhead. If timing is feasible, report both. Otherwise, label depth fraction as a "proxy" and note the caveat.

**Key insight for leaky data**: Early exits may be confident but wrong. Calibration is what makes early exit safe. If DDC-semantic exits earlier **only when confidence is calibrated** while random-init exits early but overconfidently, this directly supports the core thesis.

### Key Comparisons (K=4 vs K=2)

| Metric | K=2 Finding | K=4 Hypothesis |
|--------|-------------|----------------|
| Crystallization depth | DDC ~4-5 layers earlier | Similar or slightly later |
| Init threshold | α ≈ 0.65 | Same or similar |
| ECE at early layers | 3x lower for semantic init | Expect similar pattern |
| Early exit viable layer | L14-16 (of 24) | TBD |

---

## 7. Analysis Plan

### Phase 1: Train Models
1. Train DDC-Semantic (α=0.65)
2. Train DDC-Random (α=0.0)
3. Train Baseline-Vocab
4. Track training curves, save checkpoints

### Phase 2: Crystallization Analysis
1. Extract hidden states at each layer for test set
2. Train linear probes per layer
3. Plot layerwise AUC curves
4. Compare DDC-Semantic vs DDC-Random vs Baseline

### Phase 3: Calibration Analysis
1. Compute ECE at each layer
2. Generate reliability diagrams
3. Compare calibration by init condition
4. Check class-conditional patterns (E vs P vs I vs S)

### Phase 4: Early Exit Evaluation
1. Identify candidate exit layers (where AUC > threshold)
2. Compute accuracy at each exit layer
3. Plot accuracy vs speedup tradeoff
4. Compare to K=2 results

---

## 8. Expected Outputs

### Files
```
models/k4_ddc_semantic/       # DDC with semantic init
models/k4_ddc_random/         # DDC with random init
models/k4_baseline_vocab/     # Baseline with existing tokens

results/k4_crystallization.json    # Layerwise AUC data
results/k4_calibration.json        # ECE by layer
results/k4_early_exit.json         # Exit accuracy analysis

figures/k4_layerwise_auc.png       # Crystallization curves
figures/k4_calibration_by_layer.png
figures/k4_early_exit_tradeoff.png
figures/k4_confusion_matrix.png
```

### Key Plots
1. **Layerwise AUC**: DDC-Semantic vs DDC-Random vs Baseline (like K=2 Figure 1)
2. **ECE by layer**: Calibration dynamics
3. **Early exit tradeoff**: Accuracy retention vs speedup
4. **Class-conditional crystallization**: Do E/P/I/S crystallize at different depths?

---

## 9. Success Criteria

**Important reframe**: Accuracy is a **sanity check**, not the primary success metric. The interesting axes are crystallization depth, calibration quality, and early-exit safety.

### Minimum Viable Result
- DDC-Semantic achieves >70% test accuracy (sanity check)
- Crystallization is observable (AUC increases with depth, stabilizes)
- Some early-exit layer achieves >90% of final accuracy

### Strong Result (Validates K=2 Generalization)
- DDC-Semantic crystallizes earlier than Baseline-Vocab (by 3+ layers)
- Semantic init (α=0.65) shows better early-layer calibration (ECE at L8-L12)
- Early exit is viable with calibrated confidence
- K=4 patterns qualitatively match K=2

### Bonus Result
- Class-conditional crystallization reveals interesting E/P/I/S patterns
- Init threshold α ≈ 0.65 holds for K=4
- Embedding geometry (cosine similarity of seeds) correlates with crystallization patterns

**Note**: Even if accuracy is 75-80%, if crystallization is earlier, calibration is better, and early exit is safer — that's still a strong win. The story is about *dynamics*, not raw performance.

---

## 10. Timeline Estimate

| Phase | Tasks | Duration |
|-------|-------|----------|
| Setup | Split data, prepare scripts | 1-2 hours |
| Training | Train 3 models | 2-4 hours (GPU dependent) |
| Analysis | Crystallization + calibration | 2-3 hours |
| Plotting | Generate figures | 1 hour |
| **Total** | | **6-10 hours** |

---

## 11. Risk Factors

### Known Risks
1. **High TF-IDF leakage (93%)**: Accepted; comparative analysis remains valid
2. **K=4 may be harder than K=2**: May need more epochs or different hyperparameters
3. **Class imbalance effects**: All classes balanced, but difficulty isn't

### Mitigations
1. Focus on comparative results (DDC vs Baseline), not absolute performance
2. Monitor per-class metrics for any systematic issues
3. Save frequent checkpoints for debugging

---

## 12. Code Entry Points

**Note**: Scripts below are **to be created**. They should follow patterns established in K=2 codebase.

### Data Preparation
```bash
# To be created: src/prepare_k4_splits.py
python src/prepare_k4_splits.py --input data/k4_support/train.jsonl --output data/k4_support/
```

### Training
```bash
# To be created: src/train_k4.py (or adapt existing training scripts)
python src/train_k4.py --model ddc_semantic --alpha 0.65 --output models/k4_ddc_semantic/
python src/train_k4.py --model ddc_random --alpha 0.0 --output models/k4_ddc_random/
python src/train_k4.py --model baseline_dedicated --output models/k4_baseline_dedicated/
python src/train_k4.py --model baseline_vocab --output models/k4_baseline_vocab/
```

### Analysis
```bash
# To be created: src/analyze_k4_crystallization.py, src/analyze_k4_calibration.py
python src/analyze_k4_crystallization.py --model models/k4_ddc_semantic/ --output results/
python src/analyze_k4_calibration.py --model models/k4_ddc_semantic/ --output results/
```

### Existing Scripts to Reference
- `src/train.py` — K=2 training (adapt for K=4)
- `src/evaluate.py` — K=2 evaluation
- `src/exp16_calibration.py` — Calibration analysis patterns
- `src/exp12_early_exit.py` — Early exit analysis patterns

---

## Appendix: Leakage Context

This spec acknowledges that the K=4 dataset contains lexical shortcuts. Per the project reframing:

> "We study how decision formation behaves under realistic language conditions that include lexical shortcuts, rather than attempting to eliminate them."

### Official Leakage Metric
- **Method**: TF-IDF + Logistic Regression with **GroupKFold by `base_id`**
- **Accuracy**: 93.0% ± 0.6% (chance = 25%)
- **Note**: StratifiedKFold numbers are invalid for minimal-pair data and should not be used

### Why Comparative Analysis Remains Valid
1. Both models see identical leaky data
2. Shortcut availability alone doesn't explain calibration differences
3. Baseline models in K=2 had shortcuts but didn't crystallize early

This is documented in `docs/EXPERIMENT_HISTORY.md` Phase K4.3c.
