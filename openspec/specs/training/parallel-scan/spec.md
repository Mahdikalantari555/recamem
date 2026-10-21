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

### Requirement: Chunked trace-buffer training pass

The system SHALL advance the persistent trace buffers over a training chunk
without a Python loop over tokens. After a chunk, the `states`, `decaytrace`, and
`embedtrace` buffers SHALL hold the same values the per-token path produces for the
same chunk and the same starting state. The gradient-hook contribution that
injects the state gradient into the embedding weights and into the decay parameters
SHALL be accumulated across the chunk and applied once, and its total SHALL equal
the sum of the per-token contributions.

#### Scenario: Trace buffers match the per-token reference

- **WHEN** a chunk is processed by the chunked path and by the per-token path from the same starting state
- **THEN** the `states`, `decaytrace`, and `embedtrace` buffers SHALL match within the documented tolerance
- **THEN** the comparison SHALL cover a chunk longer than the chunk size, a batch, and the real decay initialization

#### Scenario: Embedding gradient contribution is accumulated

- **WHEN** the chunked path applies the gradient-hook contribution to the embedding weights
- **THEN** the applied contribution SHALL equal the sum of the per-token contributions within the documented tolerance

#### Scenario: Decay gradient is accumulated

- **WHEN** the chunked path sets the gradient of the decay parameters
- **THEN** the value SHALL equal the sum of the per-token values within the documented tolerance

#### Scenario: No per-token Python loop

- **WHEN** the chunked path processes a chunk
- **THEN** the number of Python-level iterations over the chunk's tokens SHALL NOT grow with the chunk length

### Requirement: Chunked training loss and gradient equivalence

The chunked training pass SHALL produce the same per-token loss values and the
same accumulated gradients as the per-token path for the same chunk and the same
starting state, within the documented tolerance. The per-token path SHALL be
retained as the named reference implementation and SHALL NOT be removed.

#### Scenario: Loss values match

- **WHEN** the same chunk is evaluated by both paths from the same starting state
- **THEN** the per-token loss values SHALL match within the documented tolerance

#### Scenario: Accumulated gradients match

- **WHEN** the same chunk is evaluated by both paths from the same starting state
- **THEN** the accumulated parameter gradients SHALL match within the documented tolerance

#### Scenario: Reference path is retained

- **WHEN** the test suite runs
- **THEN** the per-token path SHALL still exist and SHALL be the oracle for the chunked path

### Requirement: Chunked optimizer step granularity

The chunked path SHALL take one optimizer step per chunk, not one per byte. The
accumulated gradient SHALL be scaled by the inverse of the chunk length so that the
update magnitude stays on the per-byte scale. The learning-rate schedule SHALL be
a function of the number of bytes consumed, and both paths SHALL evaluate that
schedule from the number of bytes consumed.

#### Scenario: One optimizer step per chunk

- **WHEN** the chunked path processes a chunk of length $L$
- **THEN** the optimizer SHALL step exactly once
- **THEN** the step counter SHALL advance by $L$ bytes

#### Scenario: Gradient is scaled to the per-byte magnitude

- **WHEN** the chunked path accumulates a gradient over a chunk of length $L$
- **THEN** the gradient applied to the optimizer SHALL be the accumulated gradient divided by $L$

#### Scenario: Schedule is a function of bytes consumed

- **WHEN** both paths train over the same number of bytes
- **THEN** both SHALL evaluate the learning-rate schedule from the bytes consumed
- **THEN** the chunked path SHALL evaluate it once per chunk at the bytes consumed at the chunk boundary

### Requirement: Chunked runs report what is equivalent

A run performed with the chunked path SHALL report which quantities are equivalent
to the per-token path and which are not. It SHALL NOT report the chunked path as
producing identical parameters to the per-token path, because it takes one
optimizer step per chunk.

#### Scenario: Run report states the equivalence boundary

- **WHEN** a training run report is produced for a chunked run
- **THEN** the report SHALL state that the loss, the gradients, and the buffers are equivalent to the per-token path
- **THEN** the report SHALL state that the parameter trajectory is not equivalent

#### Scenario: Unmeasured equivalence is not claimed

- **WHEN** the chunked path has not been verified against the per-token path on the target environment
- **THEN** the run report SHALL NOT claim the equivalence
- **THEN** the report SHALL name the missing verification
