# Spec Delta

## ADDED Requirements

### Requirement: Load failure is a named error
The GGUF backend SHALL raise a named error when the model file cannot be loaded or the runtime is unavailable. It SHALL NOT substitute a mock model, and mock generation SHALL run only when explicitly selected by the caller.

#### Scenario: Missing GGUF file
- **WHEN** the configured GGUF path does not exist
- **THEN** loading SHALL raise an error naming the path
- **THEN** the profile SHALL NOT answer a generation request through a mock model

#### Scenario: Explicit mock is labelled
- **WHEN** the caller explicitly selects the mock backend
- **THEN** the backend SHALL report itself as mock in the capability matrix and in the answer path
