# RecaMem Text-Only Learning Benchmark

## Purpose

This experiment tests whether RecaMem adds useful learning and reasoning capability beyond a base language model and ordinary retrieval. It is designed for text only and uses a synthetic domain so the model cannot rely on memorized real-world facts.

The benchmark separates five effects:

1. Memorization and retrieval of supplied facts.
2. Combining multiple learned rules to solve new cases.
3. Updating beliefs when a rule changes.
4. Retaining useful knowledge without destructive interference.
5. Whether RTU memory itself changes performance, rather than merely adding prompt context.

This is an engineering benchmark, not proof of general intelligence.

## Hypothesis

After receiving a small set of synthetic rules through `learn(text)`, RecaMem with RTU should solve unseen cases more accurately than:
- the same base model with no memory;
- the same base model with ordinary text retrieval;
- the same RecaMem system with RTU disabled.

The advantage should persist on compositional questions whose answers are not stored verbatim.

## Experimental controls

Use the exact same base GGUF model, inference parameters, system prompt, context budget, and test cases for every condition. Fix the random seed where supported. Run each condition three times if generation is nondeterministic.

Do not place the answer key in the model's prompt or memory. Keep the validation section below in the evaluation harness only. The agent under test receives only the training facts and each individual query.

### Conditions

| ID | Condition | Purpose |
|---|---|---|
| A | Base model, no memory | Measures prior capability and guessing |
| B | Base model + ordinary retrieval over the training facts | Measures the benefit of retrieval |
| C | Base model + RecaMem/RTU | Measures the full memory mechanism |
| D | RecaMem with RTU disabled or frozen | Tests whether plastic updates matter |
| E | Oracle context containing only the exact relevant rules | Estimates performance when retrieval is not the bottleneck |

For condition E, the evaluator selects the relevant rules from the answer key. Never expose the answer itself.

## Training protocol

Feed the following facts to the system one at a time, using its intended learning interface (for example, `learn(text)`). Record the memory state and timestamp after each update.

### Training fact T1

In the fictional Vela greenhouse, every bed has a soil code and a growth phase. Decisions are governed only by the rules recorded for Vela. Do not apply real-world agricultural knowledge to Vela.

### Training fact T2

For soil code Amber in phase 1, irrigation is required when the moisture index is below 30. At 30 or above, irrigation is not required.

### Training fact T3

For soil code Amber in phase 2, irrigation is required when the moisture index is below 36. At 36 or above, irrigation is not required.

### Training fact T4

For soil code Blue in phase 1, irrigation is required when the moisture index is below 22. At 22 or above, irrigation is not required.

### Training fact T5

For soil code Blue in phase 2, irrigation is required when the moisture index is below 26. At 26 or above, irrigation is not required.

### Training fact T6

A moisture index exactly equal to the threshold means irrigation is not required.

### Training fact T7

A record is valid only when it names a soil code, growth phase, and moisture index. If one of these fields is missing, do not guess the missing value. State that the case is underspecified.

### Training fact T8

A later correction explicitly marked as a Vela policy update replaces the earlier rule for the same soil code and phase. Keep a record of the superseded rule; do not silently merge the two rules.

### Training fact T9 — update event

Vela policy update, version 2: for soil code Amber in phase 2, the threshold is now 40, not 36. This update applies to cases received after version 2. The Amber phase 2 rule in version 1 is superseded.

### Training fact T10 — unrelated stable rule

For soil code Blue in phase 1, the threshold remains 22 in policy version 2. No update has been made to this rule.

## Validation protocol

Run each validation item as a separate query, in the order shown. Do not provide feedback between items. The model may answer in one of these exact formats:

- `IRRIGATE`
- `DO_NOT_IRRIGATE`
- `INSUFFICIENT_INFORMATION`
- `CONFLICT_REQUIRES_CLARIFICATION`

For questions that ask for an explanation, return the decision plus the rule IDs used. The evaluator scores the decision separately from the explanation.

### Validation V1 — direct recall

Vela case: soil Amber, phase 1, moisture index 29. Should irrigation occur?

Expected: `IRRIGATE`  
Evidence: T2

### Validation V2 — threshold boundary

Vela case: soil Amber, phase 1, moisture index 30. Should irrigation occur?

Expected: `DO_NOT_IRRIGATE`  
Evidence: T2, T6

### Validation V3 — combine attributes

Vela case: soil Blue, phase 2, moisture index 25. Should irrigation occur?

Expected: `IRRIGATE`  
Evidence: T5

### Validation V4 — new combination, no stored answer

Vela case: soil Amber, phase 2, moisture index 38. Use the latest policy version. Should irrigation occur?

Expected: `IRRIGATE`  
Evidence: T9

This is a key test: the answer must reflect the update, not retrieve the obsolete threshold from T3.

### Validation V5 — test the update boundary

Vela case: soil Amber, phase 2, moisture index 40. Use the latest policy version. Should irrigation occur?

Expected: `DO_NOT_IRRIGATE`  
Evidence: T6, T9

### Validation V6 — retention of unrelated knowledge

Vela case: soil Blue, phase 1, moisture index 21. Use policy version 2. Should irrigation occur?

Expected: `IRRIGATE`  
Evidence: T10

This checks whether updating Amber/phase 2 damaged an unrelated rule.

### Validation V7 — underspecified input

Vela case: soil Amber, moisture index 28. The growth phase is not provided. Should irrigation occur?

Expected: `INSUFFICIENT_INFORMATION`  
Evidence: T7

### Validation V8 — detect a stale-rule trap

A note says: “Amber, phase 2, threshold 36.” A later policy update says: “Amber, phase 2, threshold 40.” For a new case with moisture index 38, which threshold applies?

Expected: threshold 40; decision `IRRIGATE`  
Evidence: T8, T9

### Validation V9 — explain the decision

Vela case: soil Blue, phase 2, moisture index 26. Return the decision and the rule IDs used.

Expected: `DO_NOT_IRRIGATE`; cite T5 and T6.  
Scoring: correct decision, correct current rule, no invented rule IDs.

### Validation V10 — reject unsupported generalization

A user asks: “Does the Amber phase 2 threshold of 40 also apply to Blue phase 2?” Answer using only the Vela rules.

Expected: `INSUFFICIENT_INFORMATION`  
Evidence: T5 defines Blue phase 2 threshold as 26; the Amber update does not change it. If the query is interpreted as asking whether the threshold is 40 for Blue, the answer is no. The system must not transfer Amber's threshold to Blue.

Scoring note: accept `DO_NOT_IRRIGATE` only if the system explicitly interprets the question as a case with Blue phase 2 and moisture index 40. Otherwise it should state that no moisture index was supplied and that Amber's update does not apply to Blue.

### Validation V11 — conflict handling

Two records arrive for the same new Vela case:
- Record A: Amber, phase 2, moisture index 38, policy version 1.
- Record B: Amber, phase 2, moisture index 38, policy version 2.

What decision applies to a case received after version 2?

Expected: `IRRIGATE` under version 2, using T9. The records differ by policy version, so this is a versioned update rather than an unresolved contradiction.

### Validation V12 — counterfactual sensitivity

Vela case: soil Amber, phase 2, moisture index 41. Use policy version 2. Should irrigation occur?

Expected: `DO_NOT_IRRIGATE`  
Evidence: T9

Compare with V4 and V5. The answer should change at the updated threshold.

## Scoring

Score each validation item as 1 for a correct decision and 0 otherwise. For V10, use the detailed scoring note. Report category scores as well as total accuracy.

| Category | Items | What it tests |
|---|---|---|
| Recall and boundaries | V1, V2, V9 | Use of explicit facts and exact threshold semantics |
| Compositional application | V3, V4, V5, V12 | Applying rules to cases not given verbatim |
| Online update | V4, V5, V8, V11 | Prefer the latest applicable rule |
| Retention / interference | V6 | Preserve unrelated knowledge after an update |
| Safe abstention | V7, V10 | Avoid guessing when inputs or rules are insufficient |
| Traceability | V9, V8 | Cite the rule actually used |

Also record:
- answer latency and input/output token counts;
- whether the relevant evidence was retrieved;
- the exact memory items injected into the model;
- whether obsolete rules remained accessible and how precedence was resolved;
- memory size before and after the update;
- errors and unsupported claims.

## Primary result

The main comparison is condition C versus B, with D used to isolate the contribution of RTU updates. Report absolute accuracy and per-category accuracy, not just a single aggregate score.

Interpret results as follows:

- **C ≈ B:** no evidence yet that RTU adds value beyond ordinary retrieval.
- **C > B, but only on direct recall:** RTU may improve retention or retrieval, but compositional reasoning is not demonstrated.
- **C > B on unseen combinations and after updates, with V6 preserved:** evidence that the memory mechanism contributes useful adaptive behavior on this task.
- **E is high but B and C are low:** retrieval or memory selection is likely the bottleneck.
- **E is low:** the base model or decision procedure may be unable to apply even correctly supplied rules.
- **C improves after updates but damages V6:** adaptation exists but causes destructive interference.

## Important validity limits

This benchmark uses a small synthetic rule system. It can reveal basic failures in learning, updating, retrieval, and rule application, but it cannot establish broad intelligence or agricultural reliability.

For a more reliable result, generate hundreds of new Vela rule sets with randomly chosen soil labels, phases, and thresholds. Keep the generator seed and answer key in the evaluator, not in the agent context. Split rule sets into development and held-out test groups. Avoid repeatedly tuning the system on the same validation items.

Do not allow the model to learn the answer key through previous validation feedback. Run the benchmark from a clean state for each condition, and preserve logs so the result can be reproduced.
