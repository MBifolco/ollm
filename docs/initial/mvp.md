# MVP: Internal Semantic Token to Reduce Ambiguity Cost in LLM Inference

## Purpose of This MVP

This MVP tests a very narrow but fundamental hypothesis:

> **Some inefficiency and instability in LLM inference comes from linguistic ambiguity, not lack of model capacity. Explicitly naming collapsed semantic distinctions internally may reduce that cost.**

This is *not* a claim about:
- general intelligence
- emergent language
- agents
- safety
- scaling laws

It is a claim about **representation efficiency during inference**.

---

## Core Motivation

### Human language vs machine reasoning

Human languages are optimized for:
- social coordination
- compression for speech
- reliance on context, embodiment, and shared experience

They are *not* optimized for:
- deterministic inference
- stable abstraction
- repeated disambiguation at scale

Humans resolve ambiguity cheaply using tone, shared history, and physical context.

LLMs cannot.

---

### Ambiguity creates repeated inference cost

When a word collapses multiple meanings, an LLM must:
- infer which meaning applies
- hedge when uncertain
- reconstruct distinctions repeatedly
- maintain unstable representations across paraphrases

This cost appears as:
- longer reasoning traces
- verbose explanations
- higher entropy predictions
- explanation–label mismatch
- sensitivity to paraphrase or truncation

This cost is paid **every inference**, making it especially relevant at scale.

---

### Hypothesis Being Tested

> If an LLM is allowed to use a small number of **internal-only semantic tokens** to explicitly name collapsed distinctions, it may:
> - reduce inference cost
> - improve consistency and faithfulness
> - without harming accuracy

This MVP tests the *smallest possible version* of that idea.

---

## MVP Scope (Intentionally Minimal)

### Domain
**Love**, split into exactly two categories:

- Romantic
- Non-romantic (familial / care / duty / commitment)

Binary split only.  
No fine-grained taxonomy.

---

### Why “Love”
- Extremely common in English
- Extremely ambiguous
- Humans frequently clarify it manually (“not like that”, “as a friend”)
- Easy to label
- Easy to evaluate explanation faithfulness

---

## Internal Semantic Token

Add exactly **one internal-only token**:

