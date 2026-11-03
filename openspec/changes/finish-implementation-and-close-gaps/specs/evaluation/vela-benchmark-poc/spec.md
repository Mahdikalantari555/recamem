# vela-benchmark-poc Specification (Delta)

## ADDED Requirements

### Requirement: Plastic-update contribution is isolatable

The harness SHALL probe the backend for hidden injection capability
(`docs/backend-contracts.md`) and SHALL run conditions C and D on the same
model, seed, context budget and V1-V12 item set. When no backend supports
hidden injection, the comparison SHALL be reported as `blocked` naming the
missing capability, and prompt-only C-versus-D results SHALL NOT be used to
claim a plastic-memory effect.

#### Scenario: Capability probe fails

- **WHEN** the selected backend reports hidden injection unsupported
- **THEN** the report SHALL record `blocked` with the backend id and the missing capability
- **THEN** no "RTU improves decisions" claim SHALL be recorded

#### Scenario: Capability probe succeeds

- **WHEN** hidden injection is available
- **THEN** C and D SHALL be run from clean state on matched budgets
- **THEN** the report SHALL include a paired comparison with a confidence interval over repeated runs where generation is nondeterministic

### Requirement: Negative results are published

The report SHALL state the outcome even when C does not outperform B or D. It
SHALL NOT tune prompts, item sets or extraction heuristics after seeing the
answer key to improve a condition.

#### Scenario: Memory does not beat retrieval

- **WHEN** condition C scores at or below condition B
- **THEN** `docs/vela-poc-results.md` SHALL publish the scores and state that no advantage was measured
- **THEN** the earlier PoC report SHALL remain available unchanged

### Requirement: Stronger model and held-out rule sets

A follow-up run SHALL use a model with measurable instruction-following on the
decision-token format and at least one held-out generated Vela rule set, with
the same conditions and provenance fields as the original run.

#### Scenario: Extraction heuristics removed

- **WHEN** the chosen model emits the required decision tokens directly
- **THEN** the report SHALL state that extraction heuristics were or were not needed
- **THEN** unparseable items SHALL still be scored zero and flagged
