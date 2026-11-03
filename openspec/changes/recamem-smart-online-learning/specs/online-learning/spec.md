# Online Continuous Learning Specification (Delta)

## Purpose
Defines real-time associative plastic learning integrated with local frozen GGUF generative inference.

## ADDED Requirements

### Requirement: Real-Time Fast-Weight Ingestion
The engine SHALL ingest streaming text propositions into the plastic matrix memory in $O(1)$ constant time per step without backpropagating into base model parameters.

#### Scenario: Ingest novel proposition
- **WHEN** `engine.teach(text)` is invoked
- **THEN** the proposition embedding SHALL be evaluated for surprise against the current state
- **THEN** state matrix $S$ SHALL be updated via Gated Delta Rule if surprise exceeds the threshold

### Requirement: Zero Base Weight Drift
The generative slow tier (GGUF model) SHALL remain frozen throughout continuous online interaction.

#### Scenario: Slow path synthesis
- **WHEN** generative chat synthesis `/slow` is executed
- **THEN** retrieved associative facts SHALL be injected into the context window
- **THEN** the GGUF model weights SHALL not be modified
