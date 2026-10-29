# evaluation/vela-benchmark-poc Specification

## Purpose

Proof-of-concept execution of the Vela text-only learning benchmark against RecaMem, using a local GGUF model for decision generation and reporting per-category accuracy.

## ADDED Requirements

### Requirement: Reproducible PoC run
The harness SHALL run conditions A, B, C, and E from a clean state, feeding T1-T10 via the intended learning interface, and evaluate V1-V12 in order without feedback between items.

#### Scenario: Clean-state runs
- **WHEN** the harness starts a condition
- **THEN** it SHALL create a fresh engine/knowledge store and SHALL NOT reuse state from a previous condition

### Requirement: Decision extraction
The harness SHALL extract exactly one of `IRRIGATE`, `DO_NOT_IRRIGATE`, `INSUFFICIENT_INFORMATION`, `CONFLICT_REQUIRES_CLARIFICATION` from each model output, or mark the item as unparseable.

#### Scenario: Unparseable output
- **WHEN** no decision token is found
- **THEN** the item SHALL be scored 0 and flagged in the report

### Requirement: Category scoring
The report SHALL include per-category accuracy for Recall and boundaries, Compositional application, Online update, Retention / interference, Safe abstention, and Traceability, plus total accuracy.

#### Scenario: Report shape
- **WHEN** a run completes
- **THEN** the report SHALL contain condition id, per-item decision and expected, category scores, total accuracy, latency, and retrieved evidence

### Requirement: Provenance
The report SHALL state the model path, anchor mode, commit hash, command, and whether RTU plastic updates were enabled.

#### Scenario: Missing model
- **WHEN** the GGUF model path is absent
- **THEN** the run SHALL be reported as blocked with the missing path and SHALL NOT substitute a mock

### Requirement: No answer-key leakage
The harness SHALL NOT place the answer key or expected decisions in any prompt or memory; the answer key SHALL exist only in the evaluator.

#### Scenario: Prompt audit
- **WHEN** the harness builds a prompt
- **THEN** the prompt SHALL contain only the training facts relevant to the condition and the single query
