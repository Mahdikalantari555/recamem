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
The system SHALL use memory injection + LLM generation for synthesis, reasoning, creative tasks.

#### Scenario: Slow path for synthesis
- **WHEN** query classified as `synthesis`, `reasoning`, or `creative`
- **THEN** `generate_with_memory()` SHALL be called with appropriate injection strategy
- **THEN** response SHALL include LLM-generated text grounded in retrieved facts

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