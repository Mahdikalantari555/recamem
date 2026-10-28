# plastic-adapter-interface Specification

## Purpose

Abstract `PlasticAdapter` interface defining the contract between Droid orchestration and plastic memory implementations (associative RTU, LoRA, Mamba, etc.), enabling hot-swappable memory mechanisms.

## Requirements

### Requirement: PlasticAdapter abstract base class
The system SHALL define an abstract base class `PlasticAdapter` with required methods: `update(x: Tensor) -> Tensor`, `retrieve(q: Tensor) -> Tensor`, `predict_next() -> Tensor`, `compute_surprise(x: Tensor) -> tuple[Tensor, float]`, `state_dict() -> dict`, `load_state_dict(dict)`.

#### Scenario: Concrete adapter implements all methods
- **WHEN** a class inherits from `PlasticAdapter`
- **THEN** it MUST implement all abstract methods
- **THEN** `isinstance(adapter, PlasticAdapter)` SHALL be True

### Requirement: Update with input embedding
The system SHALL accept an input embedding $x \in \mathbb{R}^D$ and update internal plastic state, returning the adapted output.

#### Scenario: Update returns adapted output
- **WHEN** `adapter.update(x)` is called
- **THEN** internal state SHALL be updated
- **THEN** output tensor of shape $(D,)$ SHALL be returned

### Requirement: Retrieve with query embedding
The system SHALL accept a query embedding $q \in \mathbb{R}^D$ and return a retrieved/conditioned representation.

#### Scenario: Retrieve conditions on query
- **WHEN** `adapter.retrieve(q)` is called
- **THEN** output SHALL reflect plastic memory conditioned on $q$
- **THEN** output shape SHALL be $(D,)$

### Requirement: Predict next expectation
The system SHALL generate a top-down prediction $\hat{e}_t$ from current state for surprise computation.

#### Scenario: Prediction available before input
- **WHEN** `adapter.predict_next()` is called
- **THEN** prediction tensor of shape $(D,)$ SHALL be returned

### Requirement: Surprise computation
The system SHALL compute prediction error and surprise scalar against an input.

#### Scenario: Surprise for novel input
- **WHEN** `adapter.compute_surprise(x)` called with orthogonal input
- **THEN** returns (error_vector, surprise≈1.0)

### Requirement: State serialization
The system SHALL serialize and deserialize complete plastic state via `state_dict()`/`load_state_dict()`.

#### Scenario: Round-trip state preservation
- **WHEN** `state = adapter.state_dict(); new_adapter.load_state_dict(state)`
- **THEN** `new_adapter` SHALL behave identically to `adapter`

### Requirement: Optional lifecycle operations have safe defaults
Every conforming adapter SHALL expose consolidation, health inspection, snapshot creation and snapshot restoration even when it supplies only the required methods. Defaults SHALL report unsupported optimization explicitly, preserve state and provide independent restorable snapshots. Adapter parameters and buffers SHALL participate in the standard module lifecycle.

#### Scenario: Minimal third-party adapter
- **WHEN** an adapter implements only required operations
- **THEN** optional consolidation SHALL return an explicit no-op report rather than raise an absent-method error
- **THEN** health inspection SHALL distinguish unsupported measurements from measured zero

#### Scenario: Snapshot does not alias live state
- **WHEN** an adapter snapshot is created and the live adapter is subsequently updated
- **THEN** the snapshot SHALL remain unchanged
- **THEN** restoring it SHALL reproduce the saved state and retrieval output

### Requirement: Scheduled updates negotiate adapter capabilities
Teaching SHALL apply learning-rate scheduling only through a declared supported update capability. Legacy adapters SHALL still receive their compatible update call exactly once. Results SHALL distinguish requested and applied learning rates; unsupported scheduling MUST NOT be reported as applied, and failed updates MUST NOT be reported as successful learning.

#### Scenario: Scheduled adapter update
- **WHEN** an adapter declares learning-rate support
- **THEN** the update SHALL receive the effective rate and the result SHALL report the rate actually applied

#### Scenario: Legacy update signature
- **WHEN** an adapter accepts only the original update arguments
- **THEN** teaching SHALL update its state without unexpected keyword errors or retries
- **THEN** scheduling SHALL be reported as unsupported rather than falsely applied

#### Scenario: Update fails internally
- **WHEN** an adapter raises while updating
- **THEN** the caller SHALL receive a structured failure or propagated named error
- **THEN** the update SHALL NOT be retried implicitly or logged as successful

### Requirement: Complete independent state serialization
Adapter serialization SHALL preserve numerical state, capacity, eviction counters, key registry, schedule state, adapter identity and version. Restored behavior MUST match uninterrupted execution, and exported snapshots MUST NOT share mutable tensors with the live adapter.

#### Scenario: Capacity-limited profile restarts
- **WHEN** a profile at capacity is saved, loaded and updated again
- **THEN** eviction choice, eviction count and retrieval output SHALL match an uninterrupted reference run

#### Scenario: Unsupported checkpoint version
- **WHEN** a state declares an unsupported adapter version or incompatible shape
- **THEN** loading SHALL fail explicitly before replacing valid live state
