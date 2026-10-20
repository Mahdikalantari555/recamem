# droid/onnx-anchor Specification

## Purpose
Provides a sub-25MB, high-speed ONNX runtime semantic feature extractor that anchors the RTU memory layer to prevent syntactic and representation collapse.

## Requirements

### Requirement: ONNX Inference Pipeline
The system SHALL execute semantic feature extraction using an ONNX runtime session over quantized `all-MiniLM-L6-v2` loaded from the Hugging Face cache. The anchor SHALL expose an `is_mock` flag that is `False` only while a real ONNX session is loaded. Any fallback to mock embeddings, whether caused by a missing model, a tokenizer failure, or an inference exception, SHALL set `is_mock` to `True` and emit a warning that names the cause. The mock state SHALL NOT be reported to the user as a real anchor.

#### Scenario: Sub-15ms CPU latent extraction
- **WHEN** a sentence or text chunk is passed to the ONNX anchor with a real session loaded
- **THEN** the model SHALL output a 384-dimensional normalized semantic vector in under 15 milliseconds on CPU

#### Scenario: Cache persistence
- **WHEN** the model is initialized
- **THEN** weights and tokenizers SHALL load from the Hugging Face cache without re-downloading if cached

#### Scenario: Mock fallback is observable
- **WHEN** the model file cannot be located, or an inference call raises
- **THEN** the anchor SHALL set `is_mock` to `True` and emit a warning naming the failure
- **THEN** the returned embedding SHALL still be a finite, normalized, 384-dimensional vector
- **THEN** any UI surface reporting anchor status SHALL read `is_mock` and SHALL NOT print a success badge while it is `True`
