# Spec Delta

## MODIFIED Requirements

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
