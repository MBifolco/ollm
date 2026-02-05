# Discrete Decision Channels (DDCs)

**Forcing categorical decisions into explicit, probeable tokens**

This project investigates whether **explicit decision tokens**, placed at a fixed decision locus and initialized in controlled ways, can change *when* and *how* large language models form decisions—independently of token string semantics—and whether this enables **reliable early-exit inference**.

The work focuses on **mechanistic behavior** (representation formation, calibration, decision stability), not leaderboard performance.

---

## Core Concept

Instead of training a model to emit text labels such as `romantic` or `non-romantic`, we introduce **dedicated decision tokens** into the model’s vocabulary and train the model to emit **exactly one token** at a designated decision point.

```
Scenario: <context text>
Task: Classify the meaning of "love" in this scenario.
DECISION: ⟦LOVE_ROM⟧
```

This creates a **discrete decision channel**: a single-token categorical bottleneck that forces the model to compress its internal reasoning into an explicit, probeable symbol.

---

## Research Questions

This project asks:

1. Can explicit decision channels cause decisions to **crystallize earlier** in the network?
2. Does **embedding initialization** determine decision geometry, independent of token string?
3. When do early representations become **accurate and calibrated**?
4. Under what conditions is **adaptive early-exit** safe?

---

## Key Findings (High-Level)

### 1. Crystallization timing depends on three factors
Decision crystallization layer (when logit-lens AUC first exceeds a threshold) is shaped by:
- **Token novelty**: Dedicated new tokens vs existing vocabulary
- **Embedding initialization**: Semantic (α=0.65) vs random (α=0.0)
- **Prior bias**: Flat vs peaky existing-vocab priors

In K=4, DDC (α=0.65) crystallizes at **L17**, random-init tokens at L19, and peaky vocab at L21.

---

### 2. Semantic initialization provides a modest advantage
Initializing decision-token embeddings via semantic interpolation (α=0.65) gains ~2 layers over random init (α=0.0) in K=4. Earlier work identified a phase transition around α ≈ 0.6–0.7, and the full multi-seed run confirms DDC α=0.65 consistently crystallizes earliest. The effect is clearer in K=4 than K=2.

---

### 3. Token identity is irrelevant; embedding geometry is not
DDC α=0.0 and dedicated_baseline use different token strings (`⟦LOVE_R⟧` vs `⟦BASE_R⟧`) but identical initialization. Their results are **numerically identical** across all seeds, metrics, and layers — confirming that effects arise from embedding geometry, not symbolic meaning.

---

### 4. Calibration tracks crystallization
Lower ECE consistently co-occurs with earlier crystallization. DDC α=0.65 achieves the best calibration (ECE=0.034 in K=4), while vocab_peaky has the worst (ECE=0.047). This link is what makes early-exit viable.

---

### 5. Prior bias at the decision locus delays crystallization
Existing-vocab tokens with strong pretrained priors ("peaky" tokens) crystallize 4 layers later than DDC tokens and 2–3 layers later than flat-prior vocab tokens. This confirms that uncontrolled token priors are a confound in decision-only classification.

---

## Tasks Studied

| Task | Classes | Description |
|-----|--------|-------------|
| **K=2 Love** | 2 | Disambiguate uses of “love” (romantic vs non-romantic) |
| **K=4 Support** | 4 | Classify support as emotional, practical, ideological, or structural |

---

## Experimental Discipline

Key safeguards implemented:

- Unified training & evaluation harness
- Fixed decision locus and prompt contract
- Verified single-token outputs
- Stored `run_config.json` as a trust contract
- Controlled vocab baselines (flat vs peaky)
- Multi-seed replication
- Layerwise probing and per-layer calibration

All major bugs discovered earlier are explicitly fixed and documented in the experiment history.

---

## Project Structure

```
ollm/
├── src/
│   ├── kn/
│   ├── train_kn.py
│   └── eval_kn.py
├── data/
│   ├── k2_love/{O,R,M}/
│   └── k4_support/
├── run_kn.sh
└── docs/
    └── EXPERIMENT_HISTORY.md
```

---

## Running Experiments

```bash
python src/train_kn.py --task k2_love --variant ddc --alpha 0.65 --seed 42
python src/eval_kn.py --model_path models/k2_love/ddc_a065_seed42 --mode all
```

---

## Model Variants

| Variant | Purpose |
|------|--------|
| `ddc (α=0.65)` | Semantic-initialized decision channels |
| `ddc (α=0.0)` | Random-initialized decision channels |
| `vocab_flat` | Existing tokens with minimized priors |
| `vocab_peaky` | Existing tokens with strong priors |
| `dedicated_baseline` | New tokens, random init |
| `label_word_first_token` | No new tokens, label words, first-token scoring |

### Label-Word Baseline

The `label_word_first_token` variant serves as a bridge between typical label-word fine-tuning and the single-token decision bottleneck used throughout this project. The model is prompted to output a natural-language label word (e.g., `romantic`, `nonromantic`) following an `ANSWER:` prefix, but training supervision and evaluation are applied only to the **first generated token** of the label word. This preserves a single, well-defined decision locus, making the baseline directly comparable to DDC variants in terms of decision timing, layerwise linear separability, and calibration. Because it uses pretrained vocabulary embeddings with no new tokens, any differences from DDC isolate the effect of explicit decision-token geometry.

---

## What This Is Not

- Not a claim about semantic purity of datasets  
- Not a leaderboard benchmark  
- Not prompt engineering  

This is a **mechanistic study of decision formation and calibration**.

---

## Citation

```bibtex
@misc{ddc2025,
  title={Discrete Decision Channels: Mechanistic Control of Decision Formation in LLMs},
  author={Author},
  year={2025}
}
```
