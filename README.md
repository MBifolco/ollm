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

### 1. Decision channels alter *when* decisions form
Dedicated decision tokens consistently become **linearly separable several layers earlier** than vocab-token baselines, even when final accuracy is similar.

---

### 2. Semantic initialization controls crystallization timing
Initializing new decision-token embeddings as an interpolation of semantically related words induces a **sharp phase transition** (α ≈ 0.6–0.7):

- Below threshold: late, unstable, poorly calibrated decisions  
- Above threshold: early, stable, calibrated decisions  

This transition is **independent of token string name**.

---

### 3. Token identity is irrelevant; embedding geometry is not
Tokens like `⟦LOVE_ROM⟧`, `⟦XYZ123⟧`, or emoji behave identically *when initialized identically*.  
Observed effects arise from **embedding initialization and training geometry**, not symbolic meaning.

---

### 4. Calibration emerges with crystallization
Earlier crystallization correlates strongly with:
- lower Expected Calibration Error (ECE)
- stable confidence trajectories across layers

This link is what makes early-exit viable.

---

### 5. Early-exit inference becomes practical
With confidence-gated adaptive exit, DDC models can stop computation earlier while preserving separability and calibration, yielding **measurable speedups** without accuracy collapse.

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
| `ddc (α≈0.65)` | Semantic-initialized decision channels |
| `ddc (α=0.0)` | Random-initialized decision channels |
| `vocab_flat` | Existing tokens with minimized priors |
| `vocab_peaky` | Existing tokens with strong priors |
| `dedicated_baseline` | New tokens, random init |

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
