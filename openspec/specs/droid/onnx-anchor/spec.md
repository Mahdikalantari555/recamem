# droid/onnx-anchor Specification

## Purpose
Provides a sub-25MB, high-speed ONNX runtime semantic feature extractor that anchors the RTU memory layer to prevent syntactic and representation collapse.

## Requirements

### Requirement: ONNX Inference Pipeline
The system SHALL execute semantic feature extraction using an ONNX runtime session over quantized `all-MiniLM-L6-v2` loaded from a local model path. Construction SHALL fail with a named error if the model or tokenizer cannot be loaded, and SHALL NOT substitute mock embeddings. Mock embeddings SHALL be used only when explicitly selected by the caller (`--anchor mock` or the equivalent code option). The anchor SHALL expose an `is_mock` flag that is `True` only when mock embeddings were explicitly selected. Any inference exception on a real session SHALL raise a named error rather than return an embedding of another kind.

#### Scenario: Sub-15ms CPU latent extraction
- **WHEN** a sentence or text chunk is passed to the ONNX anchor with a real session loaded
- **THEN** the model SHALL output a 384-dimensional normalized semantic vector in under 15 milliseconds on CPU
- **THEN** this scenario SHALL be reported as blocked when the model file is absent

#### Scenario: Cache persistence
- **WHEN** the model is initialized
- **THEN** weights and tokenizers SHALL load from the local model directory without re-downloading if present

#### Scenario: Missing model fails fast
- **WHEN** the model file cannot be located
- **THEN** construction SHALL raise an error naming the expected path
- **THEN** no profile SHALL be loaded with a mock anchor as a side effect

#### Scenario: Mock fallback is observable
- **WHEN** the caller selects mock embeddings
- **THEN** the anchor SHALL set `is_mock` to `True`
- **THEN** the returned embedding SHALL be a finite, normalized, 384-dimensional vector
- **THEN** any UI surface or `doctor` output SHALL report the anchor as mock and SHALL NOT print a real-anchor badge
