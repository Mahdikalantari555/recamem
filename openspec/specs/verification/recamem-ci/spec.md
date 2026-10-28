# verification/recamem-ci Specification

## Purpose
Continuous verification for recamem: an offline gate that runs without model weights, network access or a GPU, regression tests that pin every fixed defect, and a rule that tests needing real models are reported as blocked rather than passed.

## Requirements

### Requirement: Offline gate runs without weights or network
The system SHALL provide an offline gate that runs in a clean environment built from the declared dependencies, without model weights, network access, or a GPU. Its result SHALL be pass or fail.

#### Scenario: Clean environment run
- **WHEN** the offline gate runs in a fresh virtual environment with the declared dependencies and no model files
- **THEN** it SHALL pass or fail with no network request
- **THEN** every test that needs a real model SHALL be reported as blocked, not passed

### Requirement: Real-model tests are blocked, not passed
Tests that require a real model SHALL be marked and SHALL report `blocked` with the missing path when the model is absent. The gate SHALL NOT count a blocked test as passed.

#### Scenario: Missing model is blocked
- **WHEN** a real-model test runs without its model file
- **THEN** the test SHALL report blocked with the missing path
- **THEN** the gate summary SHALL list the blocked count separately from the passed count

### Requirement: Every fixed defect has a pinning test
Every defect recorded as fixed in the defect ledger SHALL have a regression test that fails on the defective behaviour and passes on the fix. The ledger SHALL name the test for each fixed row.

#### Scenario: Abbreviation regression is pinned
- **WHEN** the abbreviation sentence case is removed from the segmenter
- **THEN** the regression test for "Dr. Mina Rostami" SHALL fail

#### Scenario: Silent mock regression is pinned
- **WHEN** the anchor falls back to mock embeddings without being selected
- **THEN** the fail-fast test SHALL fail

#### Scenario: Ledger row without a test is rejected
- **WHEN** a ledger row is marked fixed and names no test
- **THEN** the offline gate SHALL fail

### Requirement: Documented surfaces match the code
The offline gate SHALL check that the CLI subcommands, the MCP tool names including aliases, and the test count stated in documentation match what the code exposes and what the collector finds. Documentation SHALL NOT contain a hand-written test count.

#### Scenario: MCP tool list drift
- **WHEN** an MCP tool is added or removed without updating the documented list
- **THEN** the gate SHALL fail

#### Scenario: Stale test count
- **WHEN** the README states a test count that differs from the collected count
- **THEN** the gate SHALL fail

### Requirement: Declared packaging
The project SHALL declare its runtime and optional dependencies in one packaging file. Every module imported by the default code path SHALL appear in the runtime declaration.

#### Scenario: Import without declared dependency
- **WHEN** the default code path imports a third-party module not declared as a runtime dependency
- **THEN** the gate SHALL fail
