# Spec Delta

## MODIFIED Requirements

### Requirement: Prompt stuffing injection
The system SHALL format retrieved facts as a structured context block prepended to the user prompt. The budget SHALL include the query, separators, special tokens and reserved output, measured with the selected backend tokenizer or byte encoder. An oversized query SHALL produce an explicit budget error rather than silently truncate the user's query.

#### Scenario: Facts injected as context
- **WHEN** `generate_with_memory(query, facts, strategy="prompt")` is called
- **THEN** prompt SHALL include `Context:\n- fact1\n- fact2\n\nQuery: {query}` when at least one fact fits
- **THEN** total prompt and reserved output SHALL not exceed model context window

#### Scenario: Multibyte input meets the backend budget
- **WHEN** facts and query contain multibyte text
- **THEN** truncation SHALL respect actual token or byte counts rather than a character-divided-by-four estimate
- **THEN** the complete query SHALL remain last

### Requirement: Hidden state injection
For a backend that advertises hidden-state injection, the system SHALL project the RTU trace into the declared hidden dimension and apply it at the designated layer during generation. Preparing a projected tensor without applying it MUST NOT count as injection. Explicit hidden mode on an unsupported backend SHALL fail with a capability error.

#### Scenario: Hidden injection at layer
- **WHEN** `generate_with_memory(query, facts, strategy="hidden", layer=12)` is called on a capable backend
- **THEN** the trace SHALL be projected to the backend hidden dimension and added to layer 12 residual stream
- **THEN** generation SHALL be conditioned on plastic memory without prompt tokens for that trace

#### Scenario: Injection scope ends after generation
- **WHEN** hidden-mode generation completes or raises
- **THEN** its hook and temporary state SHALL be removed
- **THEN** another profile or request SHALL NOT inherit the trace

#### Scenario: Unsupported hidden mode
- **WHEN** hidden mode is explicitly requested on a backend without a working hidden-state hook
- **THEN** the system SHALL report an unsupported-capability error without claiming injection occurred

### Requirement: KV cache priming
For a backend that advertises KV priming, the system SHALL encode the facts and populate the backend cache before generating the first query token. Returning token IDs without using them MUST NOT count as priming. Explicit KV mode on an unsupported backend SHALL fail with a capability error.

#### Scenario: KV cache primed with facts
- **WHEN** `generate_with_memory(query, facts, strategy="kv_cache")` is called on a capable backend
- **THEN** fact embeddings SHALL be encoded and KV cache pre-filled
- **THEN** first token generation SHALL attend to primed keys/values

#### Scenario: Independent request caches
- **WHEN** two profiles issue KV-primed requests, including a request that fails
- **THEN** their caches SHALL remain isolated and be cleaned up after use

### Requirement: Strategy selection
The system SHALL honor profile-pinned strategies and choose automatic strategies from backend-advertised capabilities. Auto mode SHALL prefer prompt for fewer than five facts, hidden for five through twenty and KV for more than twenty, falling back to prompt when the preferred operation is unsupported. Byte-RTU profiles SHALL always resolve to prompt. Reported strategy MUST describe what actually executed.

#### Scenario: Auto strategy selection
- **WHEN** 3 facts are retrieved
- **THEN** prompt strategy SHALL be used
- **WHEN** 15 facts are retrieved and the backend supports hidden injection
- **THEN** hidden strategy SHALL be used
- **WHEN** 15 facts are retrieved and the backend lacks hidden injection
- **THEN** prompt strategy SHALL be used with an explicit unsupported-capability fallback reason

#### Scenario: Fused profile pins prompt
- **WHEN** a profile attaches a byte-RTU backend and declares `injection_strategy: "prompt"`
- **THEN** strategy selection SHALL return prompt for 3, 15, and more than 20 facts
- **THEN** hidden and KV modes SHALL NOT be selected for that profile

#### Scenario: Strategy survives a profile round trip
- **WHEN** a profile is saved and loaded
- **THEN** `injection_strategy` SHALL be restored from `config.json`
- **THEN** the loader SHALL NOT drop the key when filtering GGUF-prefixed settings

#### Scenario: Prompt keeps the query last
- **WHEN** facts are injected with prompt strategy
- **THEN** the rendered prompt SHALL end with the query after the fact block
- **THEN** reordering that places the query before facts SHALL be rejected

## ADDED Requirements

### Requirement: Injection failures remain diagnosable
Injection results SHALL identify requested and executed strategy, capability status and fallback reason. Error handling MUST NOT re-execute a failed tensor operation or synthesize fake fact tokens. Invalid input SHALL produce a stable error without changing model state or global random state.

#### Scenario: Malformed trace
- **WHEN** a trace is non-tensor, nonfinite or dimensionally incompatible
- **THEN** the operation SHALL fail with a validation error without a secondary exception in its fallback formatter
- **THEN** the backend state SHALL remain unchanged
