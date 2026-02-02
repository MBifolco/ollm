# K=4 Support Semantics — Data Generation Prompt Template

## System / Instruction Prompt

> You are generating training data for a language understanding task.
>
> The goal is to create **natural, human-readable scenarios** where the word **"support"** is used, and a reader must infer **what kind of support is being given based on context alone**.
>
> Do **not** explain the category in the text.
> Do **not** encode the label explicitly.
> Do **not** rely on keywords alone.
>
> Each example should be realistic, unforced, and understandable to a non-technical reader.

---

## Task Description (for the model)

You will generate short scenarios involving the word **"support"**.
Each scenario belongs to **exactly one** of the following semantic categories:

1. **Emotional Support** – affects emotional or psychological state
2. **Practical / Instrumental Support** – involves actions or resources that materially change an outcome
3. **Ideological / Evaluative Support** – expresses agreement or approval without implementation
4. **Structural / Physical Support** – literal, mechanical, or systemic support (not metaphorical)

Only **one** category may apply per example.

---

## Category-Specific Constraints

### Emotional Support (E)

* Support changes how someone feels, copes, or understands a situation
* No tasks are completed
* No material outcome changes
* Examples may involve hardship, setbacks, or uncertainty
* Avoid overusing: *listened, talked, checked in*
* Use alternatives: *sat with, stayed nearby, made space, kept them company, reminded them, stood by them*

### Practical / Instrumental Support (P)

* Support consists of concrete actions, effort, or resources
* Removing the actions would remove the support
* Emotional reassurance alone is insufficient
* Avoid overusing: *helped, assisted*

### Ideological / Evaluative Support (I)

* Support is expressed as agreement, approval, or endorsement
* No causal contribution to implementation or execution
* May involve reading, discussion, or reflection
* Do **not** include organizing, building, enforcing, or delivering outcomes

### Structural / Physical Support (S)

* Support is literal, mechanical, or systemic
* No people, emotions, or opinions involved
* Refers to engineered systems, structures, or infrastructure
* Not metaphorical or institutional
* Allowed domains: bridges, beams, power grids, software runtimes, protocols, mechanical systems
* Not allowed: policies, laws, social programs

---

## Difficulty Control (IMPORTANT)

You will generate **either an EASY or HARD example**, as specified.

### EASY examples

* One interpretation is obvious to a human
* Minimal overlap with other categories
* Clear outcome type (emotion vs task vs belief vs structure)

### HARD (near-miss) examples

* Intentionally reuse surface cues from *other* categories
  * e.g. effort language in Ideological
  * conversation in Practical
  * work context in Emotional
* Still only one category is correct
* The correct label is determined by **what the support accomplishes**, not the words used

**Verbs may overlap. Outcomes must not.**

---

## Output Format

Return **only** the scenario text.

* 3–6 sentences
* Natural language
* Include the word **"support"** naturally
* Do not include labels, explanations, or metadata

---

## Example Generation Commands

### Generate Easy Examples

```
Generate 10 EASY scenarios of type: Emotional Support

Follow all constraints exactly. Each scenario should be 3-6 sentences,
include the word "support" naturally, and have only one valid interpretation.
```

### Generate Hard Examples

```
Generate 10 HARD (near-miss) scenarios of type: Practical Support

These should deliberately use surface cues from other categories
(e.g., conversation, encouragement) while the actual support is
action-based with material outcomes. Follow all constraints exactly.
```

---

## Internal Self-Check (Claude should do silently)

Before finalizing each example, ensure:

* Only one category applies
* A human could infer the type of support without keywords
* Removing the word "support" would not change the interpretation
* The example does not resemble instruction-following or list parsing
* No explicit category cues (emotionally, physically, etc.)

---

## Batch Generation Template

For efficient generation, use this format:

```
Generate a batch of K=4 "support" scenarios.

CATEGORY: [Emotional | Practical | Ideological | Structural]
DIFFICULTY: [Easy | Hard]
COUNT: [number]

Requirements:
- 3-6 sentences each
- Include "support" naturally
- No explicit category labels
- No lists or structured answers
- Hard examples must borrow cues from other categories

Output each scenario on its own line, numbered.
```

---

## Quality Control Notes

After generation, check:

1. **Label agreement**: Would two humans agree on the category?
2. **Verb diversity**: Are the same verbs repeated too often?
3. **Outcome clarity**: Is the *effect* of the support clear?
4. **No leakage**: Could a simple keyword match solve this?

If any check fails, regenerate or edit the example.
