# K=4 Support Semantics — Validation Checklist

This checklist exists to answer one question:

> *"Are we actually testing semantic understanding, or did shortcuts sneak back in?"*

It is intentionally redundant. Passing *most* checks is not enough.
The dataset should pass **all structural checks** and **most distributional checks**.

---

## A. Example-Level Semantic Checks (Human or LLM Review)

Apply these checks to **individual examples**.

### A1. Single-Label Sanity Check

For each example, ask:

* ❏ Could a reasonable human confidently assign **exactly one** category?
* ❏ Would two people likely agree without debate?

If "it depends" → the example is invalid.

---

### A2. Outcome-Based Label Test (Critical)

Ignore the wording. Ask only:

> What does the support *actually accomplish*?

* ❏ Emotional state changes → Emotional
* ❏ Task or outcome changes → Practical
* ❏ Belief or stance is expressed → Ideological
* ❏ Physical/system stability is enabled → Structural

If the answer relies on *verbs* rather than *effects*, reject the example.

---

### A3. Word Removal Test

Remove the word **"support"** mentally.

* ❏ Is the meaning of the support still clear?
* ❏ Would you still infer the same category?

If not → the example is leaning on the keyword, not context.

---

### A4. No Hidden Instructions Test

Scan the text for:

* lists
* enumerations
* ordering ("first", "second")
* procedural steps
* explicit choices

If present → reject.
These test instruction-following, not semantics.

---

## B. Category-Specific Guardrails

### B1. Emotional Support (E)

* ❏ No task is completed
* ❏ No resources are delivered
* ❏ Support changes feelings, coping, or perspective only
* ❏ Not reducible to "they helped with work"

Common failure:

> Emotional reassurance embedded inside practical action → usually Practical

---

### B2. Practical / Instrumental Support (P)

* ❏ Concrete actions or resources are described
* ❏ Removing the actions removes the support
* ❏ Emotional language may exist, but is not sufficient alone

Common failure:

> "They stayed late and encouraged the team" with no outcome → Emotional

---

### B3. Ideological / Evaluative Support (I)

* ❏ Support is agreement or approval only
* ❏ No causal contribution to execution or success
* ❏ Time/effort may exist, but not implementation

Fail if the person:

* organizes
* builds
* delivers
* enforces
* materially advances the outcome

---

### B4. Structural / Physical Support (S)

* ❏ No people are supported
* ❏ No opinions or judgments
* ❏ Support is literal, mechanical, or systemic
* ❏ Could exist in an empty world with no humans

Fail if:

* metaphorical ("the policy supports growth")
* social ("the system supports the community")

---

## C. Distributional / Leakage Checks (Dataset-Level)

These checks catch **shortcut learning**.

### C1. Verb Overlap Check

For each category, list the top 20 verbs.

* ❏ Significant overlap exists across categories
* ❏ No category has a unique "signature verb set"

Red flags:

* Emotional dominated by *listened, talked*
* Practical dominated by *helped, assisted*
* Ideological dominated by *endorsed, approved*

---

### C2. Noun Leakage Check

Scan for label-coded nouns:

* Ideological: *policy, proposal, law* (ok in moderation)
* Structural: *beam, bridge, framework* (ok in moderation)

* ❏ No category relies on a small set of nouns to signal the label

---

### C3. Length & Structure Balance

* ❏ Average length roughly similar across categories
* ❏ No category consistently shorter or more formulaic
* ❏ No templated phrasing repeats excessively

---

### C4. Easy vs Hard Ratio Check

Confirm intended difficulty mix:

* ❏ Training: ~60–70% Easy, ~30–40% Hard
* ❏ Test-Hard contains clear near-misses
* ❏ Hard examples reuse surface cues from other categories

If Hard examples look "obviously labeled" → they're not hard enough.

---

## D. Rewrite (Test-R) Validation Checks

Applied **after** rewrites.

### D1. Meaning Preservation

* ❏ Label does not change
* ❏ Outcome type remains the same

---

### D2. Surface Neutralization

* ❏ Repeated verbs are replaced
* ❏ Sentence structure varies
* ❏ Register and tone change

---

### D3. Domain Swapping

* ❏ Some examples move domains (office → school → home)
* ❏ Category remains inferable

---

## E. Model-Facing Sanity Checks (Pre-Training)

Before running full experiments:

### E1. TF-IDF Probe

Train a trivial TF-IDF + logistic classifier.

* ❏ Accuracy should be *well below* neural model performance
* ❏ Performance should degrade sharply on Test-R

If TF-IDF performs well → leakage exists.

Target ranges:
* Train: ideally 40-60% (moderate, not trivial)
* Test-R: near chance (~25% for K=4)

---

### E2. Random-Init Smoke Test

Train a random-init decision-token model.

* ❏ Should learn eventually
* ❏ Should show late crystallization
* ❏ Should be poorly calibrated early

If random-init performs *too well early* → labels are leaking.

---

## F. Final Go / No-Go Decision

You are ready to train **only if**:

* All example-level checks pass
* No category has unique lexical tells
* Hard examples genuinely require semantic reasoning
* TF-IDF fails meaningfully
* Rewrites preserve meaning without preserving cues

If anything fails:

> Fix the data. Do **not** compensate with modeling tricks.

---

## G. Quick Reference: Label Decision Rules

When in doubt, apply these:

| Question | If Yes → |
|----------|----------|
| Would the support still count if no tasks were completed? | Emotional |
| Can you point to a concrete deliverable/action that caused progress? | Practical |
| Is the support entirely a judgment/stance with no causal contribution? | Ideological |
| Could this support exist with no people, emotions, or opinions? | Structural |

---

## Closing Principle

> If the dataset is clean, the model's behavior is interpretable.
> If the dataset is dirty, every result is suspect.

This checklist is your guardrail against repeating the K=4 "love" failure — and it encodes everything learned the hard way.
