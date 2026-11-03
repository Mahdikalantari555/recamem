# release-evidence Specification (Delta)

## MODIFIED Requirements

### Requirement: Separate verification tiers

Release evidence SHALL distinguish offline contracts, real-model integration,
trained grounding, concurrency and migration checks, and hardware benchmarks.
Each record MUST include commit, command, configuration, seed where applicable,
dependency versions, outcome and skip reason. Missing artifacts SHALL be
reported as blocked, not passed. Completion SHALL require a passing evidence
record in **all five** tiers; a passing offline or benchmark tier SHALL NOT
substitute for a blocked real-model or grounding tier. Target misses SHALL
remain published as `partial` until implementation improves or a reviewed
requirement revision is approved, and the revision SHALL retain the original
target and the measurement that missed it.

#### Scenario: Mock anchor succeeds

- **WHEN** offline tests pass using hashed mock embeddings
- **THEN** only the offline tier SHALL pass and real-anchor quality SHALL remain unverified

#### Scenario: Checkpoint unavailable

- **WHEN** trained grounding cannot run because no checkpoint was supplied
- **THEN** the release record SHALL identify the missing artifact and SHALL NOT claim grounding success

#### Scenario: Gate goes green honestly

- **WHEN** `scripts/check_release_evidence.py --require-complete` exits 0
- **THEN** every one of the five tiers SHALL have a passing evidence record
- **THEN** no unresolved row SHALL remain in the ledger
- **THEN** the guard's row schema, tier list and `blocked` semantics SHALL be unchanged from the pre-change baseline
