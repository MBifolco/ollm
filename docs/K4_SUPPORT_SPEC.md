# K=4 Support Semantics: Data & Experiment Specification

## Project Context

This project began as an exploratory investigation into whether **introducing discrete decision channels** in language models could lead to:
- earlier internal decision crystallization,
- better calibration,
- and meaningful early-exit behavior.

The initial experiments focused on a **binary task (K=2)**:
distinguishing *romantic* vs *non-romantic* uses of the word "love" based purely on context.

Those experiments demonstrated:
- semantic initialization (rather than token string) was the critical lever,
- early layers became linearly readable only past a semantic-init threshold (~65%),
- confidence became calibrated enough to support **adaptive early exit**,
- and these effects survived rewrite-based style neutralization.

The goal of K>2 is **not novelty for its own sake**, but to answer a deeper question:

> Does this mechanism still work when the model must choose between multiple, semantically related meanings of the same word?

---

## Why We Abandoned the First K=4 Attempt ("Love")

The first K=4 extension failed for a **fundamental reason**, not an implementation bug.

### What went wrong

The original K=4 taxonomy (ROM/FAM/PLA/OBJ for "love") had a core problem:
- ROM, FAM, and PLA all used "I love you" with **overlapping contexts**
- Any disambiguation had to be explicit (e.g., "Partner: yes. Family: no.")
- Explicit disambiguation became TF-IDF exploitable or changed the task to list-parsing

After 8 iterations, we achieved 29.5% TF-IDF accuracy using position-based encoding, but this tested **instruction-following**, not semantic understanding.

### Lesson learned

For K>2 to be meaningful:
- the label **must not be explicitly encoded anywhere**,
- the task must rely on **contextual interpretation of a single word**,
- the categories must have **naturally distinct contexts** (unlike ROM/FAM/PLA which overlap),
- and a human reader should be able to classify the example without any auxiliary structure.

This realization led to a full reset of the K=4 design.

---

## New K=4 Task: The Word "Support"

We selected **"support"** as the replacement for "love" because:

- it is a single, common word,
- it has multiple, distinct meanings,
- those meanings are context-dependent,
- the categories have **naturally distinct contexts** (unlike "love"),
- and they are close enough to create real ambiguity.

Crucially, "support" scales naturally to **more than two meanings** without feeling artificial.

---

## The Four Semantic Categories (K=4)

These categories are **semantic**, not syntactic.
The model must infer *what kind of support is being given* from context alone.

### 1. Emotional Support (E)

Support that primarily affects a person's **emotional or psychological state**.

Includes:
- reassurance
- listening
- presence
- encouragement

Key constraints:
- no material outcome is changed
- no task is completed
- the effect is internal to the person
- **people only** (no animals in v1)

Coverage (to avoid over-indexing on illness/death):
- **E-Hardship**: illness, grief, caregiving, fear
- **E-Setback**: job loss, rejection, failure, uncertainty, disappointment

Both sub-modes should be represented; these are generation quotas, not labels.

---

### 2. Practical / Instrumental Support (P)

Support that consists of **actions that materially change an outcome**.

Includes:
- completing tasks
- providing resources
- fixing problems
- organizing work

Key constraints:
- remove the actions, and the support disappears
- emotional reassurance alone is insufficient
- **people only** (no animals in v1)

---

### 3. Ideological / Evaluative Support (I)

Support expressed as **agreement, endorsement, or approval** of an idea, plan, or position.

Includes:
- supporting a proposal, policy, decision
- voicing agreement
- public or private endorsement

Key constraints:
- no action taken to implement
- no emotional reassurance to a person
- **people only**
- presence of other people is *not required* (can discover idea through reading/media/reflection)
- most examples should NOT involve close personal relationships (to distinguish from Emotional)

#### Clarification: Ideological vs Practical Support

Ideological support may involve time, attention, or discussion, but it must **not causally contribute** to the success or execution of the idea being supported.

Examples that drift *out* of Ideological support include:
- organizing implementation
- contributing work that is used in execution
- persuading others as part of an implementation effort
- providing resources that increase the likelihood of success

If a person's actions materially advance an outcome, the example belongs in **Practical support**, even if it also includes agreement or endorsement.

#### v2 Extension Note

For v1, Ideological support is strictly endorsement only, no action. A future v2 could allow minor *expressive* actions (e.g., signing a petition) that don't constitute implementation. This is deferred to avoid boundary blur.

---

### 4. Structural / Physical Support (S)

Support that is **mechanical, physical, or systemic**, not social.

Two sub-types:
- **S1: Load-bearing** - physical objects directly bearing weight or stress
- **S2: Enabling systems** - systems/platforms/frameworks that enable operation or function

Key constraints:
- no human intent
- no emotional or evaluative content
- "support" is literal, not metaphorical
- **objects/systems only** (no people doing the supporting)

#### Scope of Structural / Physical Support

Structural support should refer to **literal, engineered, or physical systems** that enable stability or operation.

Allowed domains include:
- load-bearing structures (bridges, beams, frames)
- mechanical or electrical systems
- infrastructure such as power, cooling, or scheduling systems
- software or technical systems that enable operation (e.g. runtimes, protocols)

Structural support should **not** be used for:
- policies, laws, or institutions
- social or organizational backing
- abstract or metaphorical uses of "support"

If the example could reasonably be interpreted as a social, ideological, or evaluative judgment, it does not belong in this category.

---

## Label Decision Rules (Human Sanity Check)

When generating or reviewing examples, use the following rules to decide the correct label. These are not visible to the model, but help keep the dataset consistent.

* **Emotional (E):**
  The support primarily affects a person's emotional or psychological state.
  If no tasks are completed and no resources are provided, the support would still count.

* **Practical / Instrumental (P):**
  The support consists of actions or resources that materially change an outcome.
  You should be able to point to something concrete that was done or delivered.

* **Ideological / Evaluative (I):**
  The support exists entirely as agreement, approval, or endorsement.
  It does not causally contribute to implementing, executing, or enforcing anything.

* **Structural / Physical (S):**
  The support could exist with no people, emotions, or opinions involved.
  It is literal, mechanical, or systemic, not metaphorical.

If an example seems to fit more than one category, prefer the label that reflects **what the support actually accomplishes**, not the language used to describe it.

---

## Template Skeletons

These define the structural patterns for generating examples.

### Emotional Support Templates

**E1: Coping with difficulty**
```
[Person A] is going through a difficult situation involving [stress/illness/loss/setback].
[Person B] stays close, listens, reassures, or encourages them.
The text includes a sentence where [Person B] supports [Person A].
```

**E2: Ongoing encouragement**
```
[Person A] is unsure about their abilities or future.
[Person B] offers reassurance and encouragement over time.
A sentence explicitly states that [Person B] supports [Person A].
```

### Practical Support Templates

**P1: Task completion**
```
A task or project needs to be completed.
[Person A] performs actions or provides resources that help achieve it.
The text states that [Person A] supports [Person B or the effort].
```

**P2: Logistics or resources**
```
A situation requires coordination, resources, or effort.
[Person A] contributes directly through work or resources.
The text uses the word support to describe this contribution.
```

### Ideological Support Templates

**I1: Endorsing a decision or idea**
```
An idea, proposal, or decision exists.
[Person A] becomes aware of it through reading, hearing, or reflection.
The text states that [Person A] supports the idea or decision.
```

**I2: Alignment without interaction**
```
A position or viewpoint is described.
[Person A] internally or publicly expresses agreement.
The word support is used to describe this stance.
```

### Structural Support Templates

**S1: Load-bearing**
```
A physical structure or system is described.
One element bears weight or maintains stability for another.
The text states that the structure supports something.
```

**S2: Enabling systems**
```
A system, platform, or framework is described.
Its role is to enable or sustain the operation of something else.
The text states that the system supports that operation or function.
```

---

## Canonical Examples (Easy)

These anchor the core meaning of each category with high clarity.

### Emotional Support - Canonical

> After losing his job, Mark struggled with anxiety and self-doubt. He spent long evenings worrying about what would come next. His sister checked in on him every day, listened when he needed to talk, and reminded him that this setback didn't define him. She supported him through the uncertainty.

**Why this works:**
- No illness or death (covers E-Setback)
- Clear emotional hardship
- No actions that materially change the situation
- Support = presence, reassurance, encouragement

---

### Practical Support - Canonical

> The team was falling behind schedule as the deadline approached. Alex stayed late each night to fix bugs, helped organize the remaining tasks, and brought in extra equipment when it was needed. His efforts supported the group in finishing the project on time.

**Why this works:**
- Support is entirely action-based
- Remove the actions and the support disappears
- No emotional reassurance
- No endorsement of ideas

---

### Ideological Support - Canonical

> After reading the proposal outlining changes to the city's transportation policy, Maria considered the arguments carefully. Although she wasn't involved in implementing the plan, she agreed with its goals and reasoning. She publicly stated that she supported the proposal.

**Why this works:**
- No personal relationship
- No action taken to help implementation
- Support exists only as agreement
- Source is reading, not interaction

---

### Structural Support - Canonical

> The old bridge relied on a network of steel beams beneath the roadway. These beams were designed to hold the weight of passing vehicles and keep the structure stable. Together, they supported the bridge as traffic moved across it each day.

**Why this works:**
- No people involved
- No intent, emotion, or judgment
- Support is purely physical and load-bearing
- Very hard to confuse with other categories

---

## Near-Miss Examples (Hard)

These deliberately borrow surface cues from other categories to prevent shortcut learning.

### Emotional Support - Near-Miss (uses "doing / staying late")

> When the startup began to fail, Jenna felt overwhelmed and blamed herself for the outcome. Evan stayed late with her in the office, not to fix the problems, but to listen as she talked through her fears and doubts. His presence supported her as she came to terms with what had happened.

**Why this is a near-miss:**
- Uses *stayed late* (often practical)
- Occurs in a work context (often practical or ideological)
- Explicitly rules out task completion
- Support changes **emotional state only**

---

### Practical Support - Near-Miss (uses "conversation / encouragement")

> As the conference deadline approached, the presentation was still unfinished. Maya talked with the team about what remained, then reorganized the tasks, completed the final slides herself, and arranged the materials for printing. Her actions supported the group in delivering the presentation on time.

**Why this is a near-miss:**
- Uses *talked with* (often emotional)
- Has interpersonal interaction
- But support clearly consists of **actions and outcomes**
- Remove the actions, and the support no longer exists

---

### Ideological Support - Near-Miss (uses effort and involvement language)

> After following the debate over the proposed environmental regulations, Daniel spent weeks reading reports and discussing the issue with colleagues. Although he took no part in drafting or enforcing the rules, he concluded that the policy was necessary and stated that he supported it.

**Why this is a near-miss:**
- Uses *spent weeks*, *discussing* (effort cues)
- Mentions colleagues (social context)
- But explicitly excludes action or assistance
- Support exists only as **approval**

---

### Structural Support - Near-Miss (uses human-like framing)

> During the renovation, engineers examined how the building responded to stress. The internal framework adjusted as weight shifted, preventing cracks from forming and keeping the structure intact. In this way, the system supported the building over time.

**Why this is a near-miss:**
- Uses *examined*, *responded*, *preventing* (agent-like language)
- Sounds active, almost intentional
- But no people are doing the supporting
- Support is **mechanical/systemic**, not social

---

## Easy / Hard Mix Strategy

This mirrors what worked in K=2, but adapted for multiclass.

### Training Set
- **60–70% Easy** (canonical patterns)
- **30–40% Hard** (near-miss patterns)

Rationale:
- enough clarity to learn the concepts
- enough pressure to prevent shortcuts

### Evaluation Sets
- **Test-Easy**: ~80% canonical, ~20% near-miss
- **Test-Hard**: ~20% canonical, ~80% near-miss
- **Test-R**: rewritten / style-neutralized versions (especially for Test-Hard)

This allows us to separately measure:
1. concept learning (Test-Easy)
2. robustness to ambiguity (Test-Hard)
3. robustness to surface style (Test-R)

---

## Hard Example Generation: Cue Injection Rules

To systematically create hard examples, inject cues from other categories while keeping the outcome distinct.

| Category | Inject These Cues | But Ensure |
|----------|-------------------|------------|
| Emotional | "deadline", "late night", "office", "project" | No task completion; emotional state changes |
| Practical | "reassured", "listened", "encouraged", "talked" | Actions produce material outcomes |
| Ideological | "weeks of work", "helped others understand", "effort" | No implementation; only approval |
| Structural | "responded", "protected", "prevented failure" | No human intent; mechanical/systemic |

The guiding principle:

> **Verbs may overlap. Outcomes must not.**

---

## Lexical Diversity Guidance

To reduce shortcut learning and lexical leakage:

* No category should rely on a small, repeated set of verbs or phrases.
* Overlap in surface language across categories is encouraged.
* Distinctions should emerge from **context and outcome**, not keywords.

Specific guidance:

* Emotional support should not overuse "listened," "talked," or "checked in."
  Use alternatives: "sat with," "stayed nearby," "made space," "kept them company," "reminded them," "stood by them."
* Practical support should not overuse "helped," "assisted," or "supported" as standalone signals.
* Ideological support should not rely exclusively on words like "endorse" or "approve."
* Structural support should avoid overusing technical nouns that trivially identify the category.

**Verbs may overlap across categories. Outcomes must not.**

---

## Guardrails (Non-Negotiable)

### No explicit category cues
- No: "emotionally", "physically", "structurally", "ideologically"
- The meaning must emerge from the situation

### No lists, options, or choices
- No: "choose which type", numbered lists, position encoding, "the answer is…"
- The label must be **latent**

### No task framing that turns it into parsing
- We are not testing instruction-following
- Only semantic inference

### Entity constraints (v1)
| Category | Allowed Entities |
|----------|------------------|
| Emotional | People only |
| Practical | People only |
| Ideological | People only |
| Structural | Objects/systems only |

### Forbidden phrases (use sparingly)
These words can bias toward specific categories if overused:
- "infrastructure" (biases S2)
- "endorse" (biases I)
- "comforted" (biases E)
- "assisted" (biases P)

Not banned globally, but track usage and ensure distribution.

---

## Rewrite / Style-Neutralized Test Set (Test-R)

Test-R examples should preserve meaning and label while neutralizing surface cues.

Rewrite guidelines:
* vary sentence structure and register
* replace repeated verbs with alternatives
* change domains (e.g., office → school → home) without changing the type of support
* avoid introducing label-coded nouns systematically

The goal is to ensure performance reflects semantic understanding rather than stylistic familiarity.

---

## Why We Keep the Word "Support"

We considered removing the word "support" entirely and asking:
> "What kind of support is being given?"

We decided **not to do this** (for now) because:

- the original project is about **polysemy** (one word, many meanings),
- removing the word changes the task to general situation classification,
- keeping the word preserves comparability with K=2 ("love").

That said, this remains a valid future ablation.

---

## What This Phase Is Testing (Explicitly)

This phase is **not** trying to prove novelty or usefulness.

It is testing:

1. Whether discrete decision channels scale beyond binary tasks
2. Whether semantic initialization still induces early crystallization
3. Whether calibration still emerges at early layers for K>2
4. Whether adaptive early-exit routes easy cases early and hard cases late

---

## Evaluation Focus

In addition to accuracy, this phase emphasizes **calibration**.

Planned metrics:
* one-vs-rest AUC per class, macro-averaged
* macro-averaged calibration error (ECE)
* confidence vs correctness at early layers
* per-class ECE to detect category-specific miscalibration

This mirrors the K=2 findings, where calibrated confidence enabled adaptive early exit.

---

## What Success Looks Like

Success does **not** require perfect performance.

Any of the following are meaningful outcomes:

- Early crystallization occurs for easy cases but not hard ones
- Semantic-init models show better calibration than random-init
- Adaptive exit provides speedups without collapsing accuracy
- A semantic-init threshold exists, even if shifted from K=2

---

## What Failure Still Teaches Us

If this fails, we still learn:

- that multiclass ambiguity breaks early calibration,
- that K>2 requires hierarchical decisions rather than flat ones,
- or that semantic init benefits diminish as class count grows.

All of these inform future architecture and interface design.

---

## Pilot Dataset Targets

### Initial Pilot (for validation before scaling)
- 50 Easy + 25 Hard per category = **300 training examples**
- 20 Easy + 20 Hard per category for Test = **160 test examples**

### Full Dataset (after pilot validation)
- 200 Easy + 100 Hard per category = **1200 training examples**
- 50 Easy + 50 Hard per category for Test = **400 test examples**
- Test-R rewrites for hard test examples

---

## Summary

This K=4 task is intentionally:
- semantically grounded,
- resistant to shortcut learning,
- comparable in spirit to K=2,
- and rich enough to stress-test calibration and early exit.

It is designed to answer a **mechanism question**, not to optimize a benchmark.

That is by design.
