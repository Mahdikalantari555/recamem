# onnx-anchor Specification (Delta)

## ADDED Requirements

### Requirement: Real-anchor quality is measured

At closure, the real ONNX MiniLM anchor SHALL be loaded from a named local
model directory and at least one retrieval-quality measurement SHALL be taken
with it, recorded with the model checksum, the ONNX Runtime version, the seed,
the command and the hardware. The measurement SHALL be labelled `real` and
SHALL NOT be mixed with mock-embedding numbers.

#### Scenario: Mock and real compared

- **WHEN** a benchmark is run with the real anchor
- **THEN** its report SHALL state `anchor=real` and the model checksum
- **THEN** it SHALL NOT reuse or restate a mock-anchor result as a real-anchor result

#### Scenario: Real anchor unavailable

- **WHEN** the ONNX weights cannot be obtained
- **THEN** `doctor`, the API health payload and all evidence records SHALL agree that the anchor is unavailable or explicitly mock
- **THEN** the real-model tier SHALL remain `blocked`

### Requirement: Runtime state agrees across surfaces

The anchor mode reported by `recamem doctor`, `GET /api/v1/health`, the CLI
runner and every benchmark report SHALL be identical for the same process
configuration.

#### Scenario: Conflicting badges

- **WHEN** one surface reports `real` while another reports `mock` for the same profile
- **THEN** the consistency test SHALL fail
- **THEN** the release record SHALL NOT be accepted until they agree
