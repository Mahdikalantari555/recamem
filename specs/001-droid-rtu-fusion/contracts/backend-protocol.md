# Contract: Generative Backend Protocol

Status: new contract, additive only. Version 1.0.

## Scope

The engine already speaks one generative-backend protocol to `GgufBackend`
(`src/model/gguf_backend.py:62`). Any backend assigned to the engine's backend
slot must satisfy the same protocol. This contract makes that implicit protocol
explicit so the fused backend is a drop-in peer with zero engine surgery.

## Protocol

```
generate(prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9) -> str
tokenize(text: str) -> list[int]
```

Rules:

1. `generate` returns a `str`. It never raises for prompt content reasons
   (Unicode, empty, over-long): an empty or whitespace prompt returns `""`
   (FR-015). Decoding failures produce valid text, never an exception.
2. `temperature` and `top_p` are accepted for protocol parity. The fused
   backend ignores them; the component owns its sampling temperature
   (`src/model/rtu.py:231`). A backend MUST NOT crash on non-default values.
3. `generate` must be idempotent with respect to persistent state: the backend
   resets its own component state at the start of every call, so two calls with
   the same prompt start from the same state. Output text may still differ —
   sampling is stochastic (`rtu.py:228-233`).
4. `tokenize` exists so `inject_kv_cache` (`memory_injection.py:100-101`)
   degrades to prompt stuffing instead of raising. For the fused backend it is
   trivial UTF-8 bytes and is never fed to a real LLM tokenizer.
5. Frozen rule: a conforming fused backend MUST NOT modify component parameters
   or advance `step_count` during `generate`. Verification contract (test 4 of
   the feature's test file): checksum every parameter and buffer before and
   after whole answers; they must be identical (FR-003, SC-004).

## Mutual exclusion at construction

`DroidEngine.__init__` accepts `gguf_backend` (this protocol) and, new, a fused
backend flag/wiring. Supplying both is a `ValueError` (FR-008). One slot
(`self.gguf_backend`) holds whichever backend the profile configures — the fused
backend is not a second slot.

## Profile manifest contract

Fused profiles add four keys (see `data-model.md` for the full table):
`rtu_checkpoint`, `injection_strategy`, `rtu_dim`, `rtu_layers`. Load-time
rule: `rtu_checkpoint` present and file absent → raise, never random-init
(FR-007).

## Compatibility

Existing `GgufBackend` behavior is unchanged. Existing profiles without the new
keys load unchanged and never take the fused path. The standalone raw mode
(`app.py`) is a separate surface and is untouched (FR-013).
