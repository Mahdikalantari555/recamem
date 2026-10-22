# gguf-backend Specification

## Purpose

GGUF model backend using llama.cpp: load quantized LLMs (2B–8B params, 2–6 GB), run inference on CPU/GPU, manage model lifecycle per Droid profile.

## Requirements

### Requirement: GGUF model loading
The system SHALL load GGUF models from local path or Hugging Face Hub using `llama-cpp-python`, supporting quantization formats Q4_K_M, Q5_K_M, Q8_0.

#### Scenario: Load local GGUF
- **WHEN** `GgufBackend(model_path="models/llama-3.2-3b-q4_k_m.gguf")` is called
- **THEN** the model SHALL load and be ready for inference
- **THEN** context window SHALL be configurable (default 4096)

#### Scenario: Download from Hugging Face
- **WHEN** `GgufBackend(repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF", filename="*q4_k_m*")` is called
- **THEN** the model SHALL download and cache via `huggingface_hub`
- **THEN** subsequent loads SHALL use cached file

### Requirement: Token generation with streaming
The system SHALL generate tokens with optional streaming callback for real-time output.

#### Scenario: Streaming generation
- **WHEN** `generate(prompt, stream=True, callback=fn)` is called
- **THEN** `callback(token)` SHALL be invoked for each generated token
- **THEN** final output SHALL be returned on completion

### Requirement: Model parameter configuration
The system SHALL expose temperature, top_p, top_k, repeat_penalty, max_tokens as configurable parameters.

#### Scenario: Parameters affect output
- **WHEN** `generate(..., temperature=0.7, top_p=0.9)` is called
- **THEN** sampling SHALL use specified parameters

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

### Requirement: Real-model verification is a documented manual tier
Automated verification SHALL NOT require GGUF weights. The backend capability matrix SHALL report `llama_cpp` and any model path as unavailable when they are absent, and real-model loading, streaming and KV prefill SHALL be verified only by the documented manual procedure in `docs/testing.md` and `docs/backend-contracts.md`.

#### Scenario: No weights in the environment
- **WHEN** `llama_cpp` or a model file is unavailable
- **THEN** the capability matrix SHALL report KV priming and hidden injection unsupported
- **THEN** the offline suite SHALL pass against the documented stub contract instead of skipping silently
- **THEN** no generation, streaming or latency claim SHALL be recorded as verified

#### Scenario: Manual tier is executed
- **WHEN** an operator supplies a local GGUF artifact
- **THEN** the manual procedure SHALL record the binding version, the artifact checksum and the observed output
- **THEN** only that record SHALL count as real-model evidence
