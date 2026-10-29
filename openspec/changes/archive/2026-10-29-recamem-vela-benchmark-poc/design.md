# Design

## Context
The PoC uses the local MiniCPM 2B GGUF for decision generation and the ONNX anchor for retrieval. Conditions run from a clean engine state. The answer key stays in the evaluator only.

## Decisions

### Condition construction
- A: `GgufBackend` only, no engine memory; prompt = query + format instruction.
- B: store training facts in a plain list; retrieve by lexical overlap (ordinary retrieval) and inject top-k into prompt.
- C: `DroidEngine(gguf_backend=g)`; teach T1-T10; chat uses recall (extractive) for decision? The harness will call `generate_with_memory` with strategy="prompt" for decisions, and also record recall hits. For final decision, use a strict prompt with retrieved facts.
- D: same as C but `base_lr=0` and `surprise_threshold=1.1` so plastic updates are skipped (best-effort; hidden injection unavailable with this backend).
- E: oracle selects the relevant rules from the answer key (not the answer) and injects them.

### Update handling
Add `_apply_policy_update` in `_teach_impl`: after inserting propositions, if a proposition contains "policy update" or "superseded", extract `(soil, phase)` tokens and supersede prior active facts whose text contains the same soil+phase and a threshold, excluding the new fact. Implemented as a helper called per proposition.

### Decision extraction
Regex search for the four tokens in model output (case-insensitive). First match wins. If none, unparseable.

### Reporting
JSON at `reports/vela-poc-<timestamp>.json` with provenance, per-item, categories, totals.

## Risks
- LLM may produce extra text; mitigated by max_tokens=64 and strict prompt.
- Hidden injection unavailable; D approximates C.
- Supersede helper is heuristic; tests cover Amber/Blue cases.
