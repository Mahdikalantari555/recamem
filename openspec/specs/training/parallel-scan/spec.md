# training/parallel-scan Specification

## Purpose
Enables high-throughput batched training by executing linear recurrence over full sequence chunks using associative scan instead of single-step loops.

## Requirements

### Requirement: Chunked Recurrent Forward
The system SHALL compute multi-step linear hidden state trajectories in parallel chunks across sequence length $L$. The recurrence $h_t = \text{decay} \odot h_{t-1} + e_t$ SHALL be evaluated with the closed-form associative composition of affine maps, so the Python control flow is over scan levels and chunk boundaries only, never over tokens. The batch dimension SHALL be processed in the same tensor operations as the sequence dimension. A carried state SHALL make chunk boundaries exact, and the sequential single-step loop SHALL be retained as the named test reference. The chunked path SHALL be measurably faster than the reference and SHALL be selected in the training loop only by an explicit flag.

#### Scenario: Equivalence with sequential step
- **WHEN** a sequence of length $L$ is processed via chunked parallel scan
- **THEN** the resulting final hidden states and output logits SHALL match the sequential single-step execution within relative tolerance 1e-4
- **THEN** the equivalence SHALL hold for a sequence longer than the chunk size, so at least one carried-state boundary is exercised

#### Scenario: Multi-sequence batching
- **WHEN** a batch of shape $(B, L)$ is provided to the training forward pass
- **THEN** the model SHALL execute recurrence across all batch elements concurrently without Python per-token loops
- **THEN** the batch dimension SHALL NOT be iterated in Python

#### Scenario: Chunk boundary continuity
- **WHEN** the sequence is split into chunks of length $C$ and a chunk boundary falls at index $k$
- **THEN** the state at index $k$ SHALL equal the state the sequential loop produces at index $k$
- **THEN** no token SHALL be recomputed or skipped at the boundary

#### Scenario: Measured speed-up
- **WHEN** the chunked and sequential paths are timed on the fixed benchmark corpus at sequence length 512
- **THEN** the reported wall-clock factor SHALL be at least 10
- **THEN** the measured factor and the hardware SHALL be recorded, and a run that cannot measure a stable baseline SHALL report the skip rather than a lower threshold

#### Scenario: Flag-gated training integration
- **WHEN** the training loop is invoked without the parallel-scan flag
- **THEN** the existing sequential path SHALL run unchanged
- **WHEN** the flag is enabled
- **THEN** the state trajectory SHALL come from the chunked implementation and the outputs SHALL satisfy the equivalence tolerance
