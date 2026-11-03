# edge-performance Specification (Delta)

## ADDED Requirements

### Requirement: Published target misses have a disposition

Every target published in `docs/release-matrix.md` SHALL end in one of three
states: `met` with a passing measurement, `partial` with the measured miss and
its provenance, or `re-baselined` with an approved requirement revision that
records the old target, the new target, the measurement and the justification.
A target SHALL NOT be deleted, silently lowered, or omitted to obtain a pass.

#### Scenario: Target met after optimization

- **WHEN** a re-measured p95 latency is below the documented target on the declared hardware
- **THEN** the report SHALL record `met` with commit, command, seed, configuration, hardware and the measurement

#### Scenario: Target still missed

- **WHEN** a re-measured value still exceeds its target
- **THEN** the row SHALL remain `partial` and SHALL retain the original target and the measured value
- **THEN** the acceptance tier SHALL NOT be recorded as passing on that row

#### Scenario: Approved re-baseline

- **WHEN** a reviewed requirement revision changes a target
- **THEN** the evidence record SHALL contain both targets, the measurement that missed the old one, the reason and the approval reference
- **THEN** the revision SHALL NOT remove any benchmark case

### Requirement: Re-measurement precedes optimization claims

Before and after measurements SHALL be recorded for every optimization,
produced by the same command, seed and hardware class, with the delta reported.

#### Scenario: Optimization regression

- **WHEN** an optimization makes a measured metric worse
- **THEN** the report SHALL publish the regression rather than reverting silently
