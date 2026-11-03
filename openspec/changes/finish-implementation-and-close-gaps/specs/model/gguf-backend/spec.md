# gguf-backend Specification (Delta)

## ADDED Requirements

### Requirement: Manual real-model tier is executed at closure

Before completion is declared, the manual real-model procedure SHALL be run at
least once against a real GGUF artifact. Its evidence record SHALL contain the
artifact sha256, the `llama-cpp-python` version, the model path, the seed, the
command, the commit, the hardware and the observed output. A run that used
`MockLlama` or a mock anchor SHALL NOT count.

#### Scenario: Real generation verified

- **WHEN** a real GGUF and the binding are present
- **THEN** generation, streaming and KV prefill SHALL be exercised against that artifact
- **THEN** the capability matrix SHALL report the real capabilities, not the stub contract

#### Scenario: Artifact unavailable

- **WHEN** the GGUF artifact or binding cannot be obtained
- **THEN** the tier SHALL remain `blocked` with the missing path
- **THEN** completion SHALL NOT be declared

### Requirement: Provenance reference is portable

A published real-model result SHALL name a model identifier and checksum rather
than a machine-local absolute path as its primary identity. A machine-local
path MAY be included as a secondary field with the host it belonged to.

#### Scenario: Foreign path in report

- **WHEN** a report names a path that does not exist on the verification host
- **THEN** the report SHALL state the origin host and the artifact checksum
- **THEN** the checksum SHALL be the value used to identify the artifact in the ledger
