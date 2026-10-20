# memory-injection Specification

## Purpose

Memory injection strategies: insert RTU-retrieved facts into LLM prompt context, inject into hidden states, or prime KV cache for grounded generation without prompt length explosion.

## Requirements

### Requirement: Prompt stuffing injection
The system SHALL format retrieved facts as a structured context block prepended to the user prompt.

#### Scenario: Facts injected as context
- **WHEN** `generate_with_memory(query, facts, strategy="prompt")` is called
- **THEN** prompt SHALL include `Context:\n- fact1\n- fact2\n\nQuery: {query}`
- **THEN** total context SHALL not exceed model context window

### Requirement: Hidden state injection
The system SHALL inject the RTU trace vector $h_t$ into the LLM's hidden states at a designated layer via addition or concatenation.

#### Scenario: Hidden injection at layer
- **WHEN** `generate_with_memory(query, facts, strategy="hidden", layer=12)` is called
- **THEN** $h_t$ SHALL be projected to LLM hidden dim and added to layer 12 residual stream
- **THEN** generation SHALL be conditioned on plastic memory without prompt tokens

### Requirement: KV cache priming
The system SHALL pre-fill the KV cache with fact embeddings so the model attends to them from the first generated token.

#### Scenario: KV cache primed with facts
- **WHEN** `generate_with_memory(query, facts, strategy="kv_cache")` is called
- **THEN** fact embeddings SHALL be encoded and KV cache pre-filled
- **THEN** first token generation SHALL attend to primed keys/values

### Requirement: Strategy selection
The system SHALL select injection strategy based on fact count, query type, and profile config: `prompt` for <5 facts, `hidden` for 5–20, `kv_cache` for >20 or when context window constrained. The profile config SHALL be authoritative when it pins a strategy. A profile that attaches a byte-RTU backend SHALL resolve to `prompt` regardless of fact count, because the `hidden` and `kv_cache` implementations are stubs and a byte-level component consumes bytes only.

#### Scenario: Auto strategy selection
- **WHEN** 3 facts retrieved
- **THEN** `prompt` strategy SHALL be used
- **WHEN** 15 facts retrieved
- **THEN** `hidden` strategy SHALL be used

#### Scenario: Fused profile pins prompt
- **WHEN** a profile attaches a byte-RTU backend and the config declares `injection_strategy: "prompt"`
- **THEN** `select_strategy()` SHALL return `prompt` for 3 facts, 15 facts, and more than 20 facts
- **THEN** `hidden` and `kv_cache` SHALL NOT be selected for that profile

#### Scenario: Strategy survives a profile round trip
- **WHEN** a profile is saved and loaded
- **THEN** `injection_strategy` SHALL be restored from `config.json`
- **THEN** the loader SHALL NOT drop the key when filtering GGUF-prefixed settings

#### Scenario: Prompt keeps the query last
- **WHEN** facts are injected with the `prompt` strategy
- **THEN** the rendered prompt SHALL end with the query, after the fact block
- **THEN** reordering that places the query before the facts SHALL be rejected
