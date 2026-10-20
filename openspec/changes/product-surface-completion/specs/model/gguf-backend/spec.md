# Spec Delta

## MODIFIED Requirements

### Requirement: Per-profile model config
The system SHALL store GGUF model path/repo_id and parameters in Droid profile config. The memory console SHALL provide a download action that shows the resolved target path and the file size before the download starts, and SHALL report the returned path on success or the exception message on failure.

#### Scenario: Profile uses specific model
- **WHEN** Droid profile config has `gguf_model: "models/phi-3-mini-q4.gguf"`
- **THEN** `DroidEngine` SHALL initialize `GgufBackend` with that model

#### Scenario: Download shows destination before starting
- **WHEN** user supplies a repo id, filename, and local directory in the console
- **THEN** the console SHALL display the resolved target path
- **THEN** it SHALL display the file size, or `size unknown` when the Hub provides no metadata
- **THEN** the download SHALL start only after the user confirms

#### Scenario: Download failure is reported
- **WHEN** the download fails
- **THEN** the console SHALL show the exception message and the attempted target path
- **THEN** the console SHALL NOT report success or silently fall back to a different mechanism without saying so
