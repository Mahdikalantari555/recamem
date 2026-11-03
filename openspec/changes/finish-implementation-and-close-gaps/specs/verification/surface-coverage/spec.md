# surface-coverage Specification (Delta)

## ADDED Requirements

### Requirement: Shipped surfaces are complete or explicitly blocked

Every navigation entry and documented surface SHALL either expose real data from
the engine or be removed. A surface that cannot run SHALL name the missing
toolchain or artifact and SHALL NOT display invented values. The desktop shell
SHALL be compiled and smoke-tested, or reported as `blocked` with the exact
missing toolchain and WebView.

#### Scenario: Placeholder page

- **WHEN** a page is listed in navigation but has no backend contract
- **THEN** it SHALL be implemented against a real endpoint or removed from navigation
- **THEN** a surface test SHALL fail if it renders fabricated metrics

#### Scenario: Tauri toolchain unavailable

- **WHEN** `recamem ui --dry-run` cannot find a compiled Tauri binary
- **THEN** it SHALL report `shell=browser`, `binary exists=no` and the missing toolchain
- **THEN** the surface SHALL NOT be described as shipped

### Requirement: Doctor reflects real runtime state

`recamem doctor` SHALL exit 0 only when the real anchor loaded and a named
generation backend is available. It SHALL exit non-zero with an explicit mock
anchor and SHALL name the expected model path in its report.

#### Scenario: No weights

- **WHEN** the ONNX anchor and GGUF artifact are absent
- **THEN** `doctor` SHALL exit 1, report `anchor=unavailable` or explicit `mock`, and name the expected path
- **THEN** no health or UI surface SHALL display a real-anchor badge
