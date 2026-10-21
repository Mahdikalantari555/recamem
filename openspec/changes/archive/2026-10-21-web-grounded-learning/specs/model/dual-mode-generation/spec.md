# Spec Delta

## MODIFIED Requirements

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
- **THEN** the encoded prompt SHALL NOT contain a trailing `"\n"` added by the backend
- **THEN** the query SHALL remain the last content in the prompt

#### Scenario: Answer path is observable

- **WHEN** an answer is returned by the engine
- **THEN** `_last_answer_path` SHALL be one of `recall`, `fused`, `llm`, or `researched`
- **THEN** the console badge SHALL display that value beside the answer

#### Scenario: Researched answer is grounded in acquired facts

- **WHEN** an answer was produced from facts acquired during the request
- **THEN** the answer path SHALL be reported as `researched`
- **THEN** the answer SHALL NOT have been produced by free generation without acquired facts
