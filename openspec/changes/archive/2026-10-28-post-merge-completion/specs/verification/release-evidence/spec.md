# Spec Delta

## Purpose

Provide auditable evidence that shipped behavior satisfies its requirements, distinguishing implemented features from plans, mocks, skipped checks and unmeasured research targets.

## ADDED Requirements

### Requirement: Complete requirement and task traceability
The project SHALL maintain a versioned ledger mapping every current requirement and historical task to implementation, verification evidence and a disposition of verified, missing, partial, blocked or superseded. Superseded entries MUST reference an approved replacement requirement; unresolved entries MUST identify an owning task.

#### Scenario: Historical archive with unchecked tasks
- **WHEN** an archived task has no executable evidence
- **THEN** its disposition SHALL remain unresolved regardless of archive location
- **THEN** a follow-up task SHALL implement or explicitly revise the missing behavior

#### Scenario: Existing behavior is verified
- **WHEN** code and tests satisfy a historical task
- **THEN** the ledger SHALL record source paths, test identifiers, commit and command result
- **THEN** the project SHALL NOT reimplement that behavior solely because its checkbox is stale

### Requirement: Dependency closure before completion
Completion of this change SHALL require implementation and evidence-backed closure of `delta-plastic-memory`, `autonomous-sleep-consolidation` and `edge-slm-optimization`, plus every unresolved historical behavior in its ledger. An intentionally revised target MUST remain visible with the approved replacement contract.

#### Scenario: Dependency remains a proposal
- **WHEN** any dependent change has incomplete implementation or acceptance evidence
- **THEN** this change SHALL remain incomplete and SHALL NOT be archived as fully implemented

### Requirement: Separate verification tiers
Release evidence SHALL distinguish offline contracts, real-model integration, trained grounding, concurrency and migration checks, and hardware benchmarks. Each record MUST include commit, command, configuration, seed where applicable, dependency versions, outcome and skip reason. Missing artifacts SHALL be reported as blocked, not passed.

#### Scenario: Mock anchor succeeds
- **WHEN** offline tests pass using hashed mock embeddings
- **THEN** only the offline tier SHALL pass and real-anchor quality SHALL remain unverified

#### Scenario: Checkpoint unavailable
- **WHEN** trained grounding cannot run because no checkpoint was supplied
- **THEN** the release record SHALL identify the missing artifact and SHALL NOT claim grounding success

### Requirement: Evidence-backed performance claims
Performance and quality claims SHALL reference reproducible measurements on identified hardware and model artifacts. Target misses MUST be published without changing thresholds or deleting cases to manufacture success. Required targets that miss SHALL block acceptance until implementation improves or a reviewed requirement revision is approved.

#### Scenario: Optimization misses its target
- **WHEN** an HNSW, sparse-memory, sleep-recovery or test-time-compute benchmark misses its specified target
- **THEN** the report SHALL retain the measured result, configuration and target
- **THEN** documentation SHALL describe it as an unmet target rather than shipped performance

### Requirement: Regression gates for merged behavior
The completion release SHALL preserve profile isolation, atomic manifest ordering, frozen byte-RTU inference, bounded output, no-facts fallback, dense-gated associative scoring, contract coverage and legacy checkpoint compatibility.

#### Scenario: New adapter is introduced
- **WHEN** a new memory implementation is enabled
- **THEN** the existing adapter and fused-engine regression suites SHALL still pass
- **THEN** existing profiles SHALL retain their explicitly stored adapter unless migration was requested
