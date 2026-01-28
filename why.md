## 1. Background & Motivation (How We Arrived Here)

### Core Observation
Human language is not optimized for machine reasoning.

Many natural languages (English in particular) collapse multiple distinct concepts into a single surface word and rely on:
- context
- embodiment
- shared social knowledge

to disambiguate meaning.

Humans handle this cheaply.  
LLMs do not.

For an LLM, ambiguity creates **repeated inference cost**:
- longer internal reasoning
- hedging language
- unstable explanations
- higher entropy predictions
- inconsistency across paraphrases

This cost is paid **every inference**, not once at training.

---

### Example: “Love”
English collapses multiple meanings into one word:
- romantic desire
- familial care
- duty/commitment
- altruistic concern

Other languages lexicalize these distinctions more explicitly.

LLMs trained on English must *infer* which meaning applies every time “love” appears, often reconstructing the same distinction repeatedly from context.

This led to the hypothesis:

> Some inference inefficiency in LLMs comes not from lack of intelligence, but from **linguistic under-specification**.

---

### Why embeddings alone don’t solve this
While embeddings can encode nuance implicitly:
- the distinctions are soft, not addressable
- they must be re-inferred every time
- they are fragile under paraphrase or truncation

Explicitly naming a distinction (even internally) may:
- reduce entropy
- shorten reasoning
- improve consistency
- reduce token usage

Humans invent jargon, notation, and technical vocabulary for exactly this reason.

---

### Key Hypothesis
> Allowing an LLM to use a small number of **internal-only semantic tokens** (not shown to users) to name collapsed distinctions can reduce inference cost and/or improve output quality without hurting accuracy.



