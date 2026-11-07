## ADDED Requirements

### Requirement: CPU Muir candidate
The system SHALL offer an explicitly selected Muir-inspired candidate with channel-wise decay, separate sensitivity-allocated low-rank erase and write gates, and configurable static or dynamic readout. The implementation SHALL document deviations from the paper and SHALL run on CPU with existing core dependencies, without GGUF or accelerator dependencies.

#### Scenario: Explicit core selection
- **WHEN** a caller selects the Muir candidate with a valid configuration
- **THEN** the candidate exposes the SequenceCore interface without changing the C4 default or loading incompatible C4 weights.

### Requirement: Correct streaming state
The candidate SHALL carry both recurrent matrix state and any causal convolution history between calls and report their CPU memory footprint.

#### Scenario: Step and sequence parity
- **WHEN** identical inputs and initial states are processed as a sequence or one token at a time
- **THEN** outputs and final states agree within documented precision tolerances, including across chunk boundaries.

### Requirement: Evidence before promotion
Muir SHALL remain experimental until reproducible CPU comparisons against C4 have been recorded at matched parameter and training budgets.

#### Scenario: Evaluation report
- **WHEN** a comparison is run
- **THEN** it reports model configurations, actual parameter counts, dataset provenance, seeds, hardware, token budgets, language and recall metrics, time, and memory without implying TPU results apply to CPU.
