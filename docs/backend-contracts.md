# Backend capability and injection contracts

Memory injection is a backend operation, not a label. Each generation backend
advertises what it can do; the engine reports the strategy that actually ran,
and falls back only with an explicit reason.

## Support matrix

| Backend | Prompt | Hidden state | KV priming | Notes |
| --- | --- | --- | --- | --- |
| `ByteRtuBackend` | yes | no | no | Frozen byte decoder; prompt-only by design. |
| `GgufBackend` (no binding / `MockLlama`) | yes | no | no | Nothing is faked; the mock never advertises prefill. |
| `GgufBackend` with a llama.cpp binding | yes | no | yes, when the binding exposes `eval()` + `reset()` | llama.cpp has no residual-stream hook. |
| `HookableDecoderBackend` (your own `nn.Module`) | yes | yes | no | Scoped forward hooks on the declared layer. |

`capabilities_of(backend)` returns a `BackendCapabilities` record. A backend that
declares `supports_hidden_injection()`/`supports_kv_priming()` is trusted,
*including an explicit `False`* — duck typing never overrides a declaration.
A backend that declares nothing is prompt-only.

## Result contract

Every injection result reports:

| Field | Meaning |
| --- | --- |
| `requested` | What the caller asked for (`prompt`, `hidden`, `kv_cache`, or `auto`). |
| `executed` | What actually ran. |
| `applied` | `True` only if the hook fired / the prefill primed tokens. |
| `reason` | Why `executed != requested` (`unsupported_capability:*`, `prompt_only_backend:*`, `no_facts_fit`). |
| `cleanup` | `removed` / `released` after the scope exits. |
| `capabilities` | The backend's advertised capabilities at call time. |

Explicit unsupported modes raise `UnsupportedCapabilityError`; automatic
selection falls back to prompt rendering and records the reason.

## Prompt budgeting

`fit_prompt` counts the **complete** prompt — header, separators, fact lines and
the query — with the backend tokenizer (UTF-8 bytes for byte backends), against
`context_window - output_reserve`. Facts are trimmed whole from the tail; the
query is never truncated, and a query that cannot fit raises `QueryBudgetError`.
The rendered prompt always ends with `Query: <query>`.

## Scoped operations

```python
from src.model.backend_contracts import HiddenInjection, KvPrefill

with HiddenInjection(decoder_backend, trace, layer_idx=12):
    text = decoder_backend.generate(prompt)   # hook live only here

with KvPrefill(gguf_backend, ["fact one", "fact two"]):
    text = gguf_backend.generate(prompt)      # prefilled, then reset
```

Both are context managers: the hook is removed and the cache is reset on exit,
including when generation raises. Two concurrent prefills use separate caches.

## Testing status

* Hidden injection is proven against a deterministic in-process decoder
  (`tests/test_torch_decoder_backend.py`, `tests/test_injection_engine.py`).
* KV prefill ordering/cleanup is proven against a llama.cpp binding double
  (`tests/test_gguf_capabilities.py`). A real GGUF artifact is **not** present
  in this environment; that integration remains blocked in
  `reports/completion-evidence.json`.
