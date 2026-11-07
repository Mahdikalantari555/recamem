## ADDED Requirements

### Requirement: Reproducible CPU performance baseline
The system SHALL provide a configurable CPU profiling harness for the selected C4 and experimental Muir cores, reporting configuration, seeds, hardware, parameters, timing and streaming parity without claiming language understanding.

#### Scenario: Synthetic CPU measurement
- **WHEN** the harness runs on synthetic data
- **THEN** the report SHALL identify that dataset as synthetic, record the actual model shapes and hardware, distinguish observed RSS from peak memory and report whole-sequence versus incremental numerical error.

### Requirement: Evidence-gated acceleration
Optional Rust or third-party acceleration SHALL preserve the default Python-only install and be evaluated for output, state and gradient correctness and representative end-to-end latency, memory and quality before promotion.

#### Scenario: Rust toolchain unavailable
- **WHEN** no Rust toolchain is installed
- **THEN** the system SHALL report native compilation as blocked and retain the existing reference path without presenting it as a successful Rust benchmark.

### Requirement: Security boundary independent of libraries
The system SHALL NOT treat an orchestration, parser or browser dependency as proof of network isolation or prompt-injection protection; all research actions SHALL remain subject to the approved research security policy.

#### Scenario: Browser isolation unavailable
- **WHEN** the required public-only network boundary cannot be enforced
- **THEN** optional rendered browsing SHALL remain unavailable rather than falling back to unprotected networking.
