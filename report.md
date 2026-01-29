
# **Semantic Decision Tokens as Compact and Robust Interfaces in Language Models**

## Abstract

Large language models typically represent decisions through natural language outputs, which are verbose, ambiguous, and sensitive to surface-level shortcuts. We investigate whether **explicit semantic decision tokens** can serve as compact, faithful, and robust internal representations that both improve generalization and enable efficient inference interfaces. In a controlled classification task distinguishing romantic versus non-romantic uses of the phrase “I love you,” we show that (1) a supervised decision-token objective yields substantially better and more stable decision boundaries than a label-only baseline, (2) the learned token can be exposed as a one-token inference interface without sacrificing accuracy, and (3) this advantage persists under aggressive surface-form perturbations that break shallow lexical shortcuts. Together, these results suggest that semantic tokens can function as meaningful internal bottlenecks and practical low-bandwidth interfaces for LLM decision-making.

---

## 1. Introduction

Language models “think” and “speak” in the same vocabulary. As a result, internal decisions are often expressed through verbose natural language, conflating reasoning, explanation, and output. This raises two problems:

1. **Faithfulness and robustness**: Natural language outputs are sensitive to stylistic and lexical shortcuts that may not reflect true semantic understanding.
2. **Efficiency**: Communicating a decision often requires multiple tokens, even when the underlying task is binary.

We ask whether introducing **explicit semantic decision tokens**—discrete symbols supervised to represent a decision—can improve both robustness and efficiency. Rather than relying on hidden chain-of-thought or latent variables, we test a minimal intervention: adding a small number of learned tokens and supervising their use as decision markers.

---

## 2. Task and Setup

### Task

Binary classification of whether the phrase **“I love you”** is used in a **romantic** or **non-romantic** sense within a short scenario.

The task is intentionally ambiguous and requires contextual interpretation rather than keyword detection.

### Models

* **Baseline**: Label-only supervision (“romantic” / “non-romantic”).
* **Token model**: Supervised to emit a **semantic decision token**
  ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧, followed by the label.

Both models share:

* identical architecture (Qwen-family instruction-tuned model),
* identical data splits,
* identical training procedures (except for the token objective).

---

## 3. Track 1: Meaning — Does the Token Learn a Better Decision Boundary?

We first test whether the decision token changes the **decision geometry**, independent of output length or decoding tricks.

### Experimental Design

* Held out entire **scenario buckets** (“crisis” and “collaboration”) to induce distribution shift.
* Trained with **three random seeds**.
* Evaluated using **forced-choice logit probing** and **AUC**, rather than raw accuracy alone.

### Results (Seed Sensitivity)

| Metric     | Baseline (Mean) | Token (Mean)              |
| ---------- | --------------- | ------------------------- |
| Accuracy   | 63.04% ± 7.81   | **86.38% ± 5.77**         |
| Token AUC  | —               | **0.9822 ± 0.0016**       |
| Δ Accuracy | —               | **+23.33% (min +15.22%)** |

**Key findings**

* The token model consistently outperforms the baseline across all seeds.
* AUC is extremely stable, indicating a clean and well-separated decision boundary.
* The baseline collapses under distribution shift, while the token model remains robust.

**Conclusion (Track 1)**
The semantic decision token is not merely a formatting trick; it reshapes the decision boundary and improves generalization.

---

## 4. Track 2: Interface Efficiency — Can the Token Be Used Directly?

We next ask whether the learned token can function as a **practical inference interface**, not just an internal signal.

### Experimental Design

* Same held-out test set (230 examples).
* **Constrained decoding**:

  * Token model: must emit ⟦LOVE_ROM⟧ or ⟦LOVE_NONROM⟧.
  * Baseline: evaluated via forced-choice sequence logprob.
* Measured both **decode steps** and **communicated tokens**.

### Results (Efficiency)

| Model    | Accuracy   | Decode Steps | Communicated Tokens | Decision Method      |
| -------- | ---------- | ------------ | ------------------- | -------------------- |
| Baseline | 53.48%     | 0            | 3.0 avg             | seq logprob          |
| Token    | **87.83%** | **1**        | **1**               | argmax {ROM, NONROM} |

*(“romantic” = 2 tokens, “non-romantic” = 4 tokens)*

**Key findings**

* The token interface preserves **exactly the same accuracy** as Track 1.
* The decision is communicated in **one token**, versus 2–4 tokens for the baseline.
* The interface is real: constrained generation emits the decision token directly.

**Conclusion (Track 2)**
A learned semantic token can serve as a **low-bandwidth, one-token decision interface** without sacrificing correctness.

---

## 5. Experiment 3A: Robustness — Is the Token Just Learning Style?

Synthetic datasets often contain surface-level shortcuts. To test whether the token captures **semantics rather than style**, we explicitly attacked the dataset.

### Shortcut Diagnosis

A TF-IDF classifier trained on surface text achieved:

* **97.39% accuracy**, AUC ≈ 1.0

This confirms extreme lexical leakage.

### Rewrite Attack

* All 230 test scenarios were **rewritten** to preserve label and bucket while aggressively altering surface form.
* No retraining was performed.

### Results (Separability Under Rewrite Shift)

| Metric          | Original | Rewritten  | Drop    |
| --------------- | -------- | ---------- | ------- |
| TF-IDF Accuracy | 97.39%   | 80.87%     | −16.52% |
| Token AUC       | 0.9786   | **0.9525** | −0.026  |
| Baseline AUC    | 0.8514   | 0.8079     | −0.044  |

**Key findings**

* Rewrites successfully break shallow lexical shortcuts.
* The token model’s AUC degrades **~1.7× less** than the baseline.
* Separability remains high despite aggressive surface perturbation.

**Conclusion (Experiment 3A)**
The decision token promotes **more invariant decision boundaries**, surviving style attacks better than a label-only model.

---

## 6. Discussion

Across three experiments, we observe a consistent pattern:

1. **Meaning**: Decision tokens improve generalization and stability.
2. **Interface**: The same token can be decoded as a one-token output.
3. **Robustness**: Token-based separators are less sensitive to surface shortcuts.

Importantly, this approach:

* does not rely on chain-of-thought,
* does not require architectural changes,
* and uses only standard supervised fine-tuning.

The results suggest that **explicit semantic tokens** can function as both internal bottlenecks and external APIs for model decisions.

---

## 7. Limitations and Future Work

* The task is binary and narrow; broader semantic domains remain to be tested.
* The dataset is synthetic, though explicitly stress-tested for shortcut reliance.
* Future work could explore:

  * phrase transfer (“I adore you”),
  * compositional tokens,
  * sample efficiency,
  * multi-label decision interfaces.

---

## 8. Conclusion

We show that supervising language models with explicit semantic decision tokens yields more robust decision boundaries and enables compact, faithful inference interfaces. By separating **decision representation** from **natural language expression**, semantic tokens offer a promising primitive for efficient and reliable LLM systems.

---

## What I *don’t* need from you

You’ve already given everything required for this version.

## Optional next steps (only if you want)

* A one-paragraph **Related Work** section
* A short **blog-friendly version**
* A figure sketch (decision boundary / interface diagram)
* Or help turning this into a workshop submission

Just tell me where you want to take it.
