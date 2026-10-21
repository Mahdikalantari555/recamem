# dual-mode-generation Specification

## Purpose

Dual-mode generation: fast path (RTU-only, <50ms) for factual recall and routing; slow path (RTU+LLM, 200–2000ms) for synthesis, reasoning, and open-ended generation. Automatic routing based on query type and confidence.

## Requirements

### Requirement: Fast path — RTU factual recall
The system SHALL answer factual queries using only `recall()` + template synthesis without LLM generation. `recall()` SHALL rank candidates with a documented scorer in which every contributing signal has a named weight, a documented source, and a stated direction. The scorer SHALL replace the previously hardcoded per-query boosts. The dense-similarity gate SHALL be applied to the dense signal before any other signal contributes, so a non-lexical or associative signal can never admit a fact that the gate rejects.

#### Scenario: Fast path for fact lookup
- **WHEN** query classified as `specific_fact` and retrieval confidence > 0.7
- **THEN** `chat()` SHALL return synthesized facts in < 50ms
- **THEN** no LLM generation SHALL occur

#### Scenario: Documented scorer order
- **WHEN** two candidates pass the dense gate with equal dense similarity and equal lexical overlap
- **THEN** their relative order SHALL follow the documented signal order
- **THEN** changing an undocumented constant SHALL NOT change the order

#### Scenario: Gate stays authoritative
- **WHEN** a candidate's dense similarity is below the gate
- **THEN** no lexical, associative, recency, or confidence signal SHALL place it above a candidate that passed the gate

#### Scenario: Signals are auditable
- **WHEN** a reviewer reads the scorer definition
- **THEN** each signal SHALL state its weight, its source, and whether it is additive or multiplicative
- **THEN** the public result fields `similarity`, `original_similarity`, and `contextual_alignment` SHALL remain populated

### Requirement: Slow path — RTU+LLM synthesis
The system SHALL use memory injection + generation for synthesis, reasoning, and creative tasks. The generative component SHALL be any backend implementing `generate(prompt, max_tokens, temperature, top_p)`: a GGUF model, an external LLM provider, or a frozen byte-RTU backend. The slow path SHALL pass retrieved facts through the pinned injection strategy and SHALL report which component produced the answer.

#### Scenario: Slow path for synthesis
- **WHEN** query classified as `synthesis`, `reasoning`, or `creative`
- **THEN** `generate_with_memory()` SHALL be called with appropriate injection strategy
- **THEN** response SHALL include generated text grounded in retrieved facts

#### Scenario: Byte-RTU backend is a peer of the GGUF backend
- **WHEN** a profile attaches a `ByteRtuBackend` instead of a GGUF backend
- **THEN** the slow path SHALL call `generate(prompt, max_tokens=…, temperature=…, top_p=…)` with the same call shape
- **THEN** the answer path SHALL be recorded as `fused`

#### Scenario: Both backends supplied
- **WHEN** a `DroidEngine` is constructed with both `gguf_backend` and `byte_rtu_backend`
- **THEN** construction SHALL raise `ValueError` rather than silently choosing one

### Requirement: Automatic path routing
The system SHALL route queries using System-1 decision head (from `system1-mcp-surface`): `decide_action(query)` returns `fast_recall` or `slow_synthesis` with confidence.

#### Scenario: Routing decision
- **WHEN** `decide_action("what is NDVI?")` returns `fast_recall` with confidence 0.92
- **THEN** fast path SHALL be used
- **WHEN** `decide_action("explain how remote sensing helps agriculture")` returns `slow_synthesis` with confidence 0.85
- **THEN** slow path SHALL be used

### Requirement: Explicit mode override
The user SHALL be able to force fast or slow mode via CLI flag or chat command (`/fast`, `/slow`).

#### Scenario: User forces slow mode
- **WHEN** user sends `/slow explain remote sensing`
- **THEN** slow path SHALL be used regardless of classification

### Requirement: Fused RTU+Byte-RTU grounding path

The system SHALL provide a fused answer path in which retrieved facts are injected
into a query-last prompt consumed by a frozen byte-level RTU. The component's
weights SHALL remain frozen, no optimizer step SHALL occur, and output SHALL be
bounded by `min(max_tokens, max_output_bytes)`. With no recalled facts the path
SHALL return the no-information fallback and SHALL NOT free-generate. The producing
path SHALL be observable as `recall`, `fused`, `llm`, or `researched`.

#### Scenario: Grounded fused answer

- **WHEN** a profile with a trained checkpoint has taught facts and receives a slow-mode query that recalls at least one fact
- **THEN** the fused answer SHALL be produced by the frozen byte-RTU component primed with those facts
- **THEN** the answer path SHALL be reported as `fused`

#### Scenario: No facts, no generation

- **WHEN** `recall()` returns no hits for a fused profile
- **THEN** the path SHALL return the no-information fallback string
- **THEN** no generated text SHALL be returned and no sampling loop SHALL run

#### Scenario: Weights stay frozen

- **WHEN** several fused answers are produced
- **THEN** every model parameter and every `state`/`decaytrace`/`embedtrace` buffer SHALL be unchanged
- **THEN** no optimizer step SHALL be taken during inference

#### Scenario: Output bounds

- **WHEN** the backend is asked for `max_tokens` greater than `max_output_bytes`
- **THEN** the output SHALL be capped at `max_output_bytes`
- **WHEN** `max_tokens` is smaller
- **THEN** the output SHALL be capped at `max_tokens`
- **WHEN** `max_tokens` is zero or negative
- **THEN** the output SHALL be the empty string

#### Scenario: Prompt has no trailing newline

- **WHEN** the fused path primes the component
- **THEN** the encoded prompt SHALL NOT contain a trailing `"
"` added by the backend
- **THEN** the query SHALL remain the last content in the prompt

#### Scenario: Answer path is observable

- **WHEN** an answer is returned by the engine
- **THEN** `_last_answer_path` SHALL be one of `recall`, `fused`, `llm`, or `researched`
- **THEN** the console badge SHALL display that value beside the answer

#### Scenario: Researched answer is grounded in acquired facts

- **WHEN** an answer was produced from facts acquired during the request
- **THEN** the answer path SHALL be reported as `researched`
- **THEN** the answer SHALL NOT have been produced by free generation without acquired facts
