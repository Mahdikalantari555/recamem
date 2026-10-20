# Phase 0 Research: Droid × Raw Byte RTU Fusion

All findings below were verified against the current source on 2026-10-11. No open
technical unknowns remain; every decision has either a source citation or a
measured gate.

## Decisions

### D1 — Backend is a drop-in peer of `GgufBackend`

- **Decision**: The fused backend exposes exactly `generate(prompt, max_tokens,
  temperature, top_p) -> str` plus a trivial `tokenize()`, and is assigned to the
  existing `DroidEngine.gguf_backend` slot.
- **Rationale**: The slow path already calls one call shape —
  `self.gguf_backend.generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9)`
  at `src/model/droid.py:1153-1157`. A protocol-compatible peer needs zero
  `DroidEngine` surgery and inherits the existing exception fallback at
  `droid.py:1156-1157`.
- **Alternatives considered**: Adding a separate `byte_rtu_backend` code path
  inside `chat()`/`generate_with_memory()`. Rejected: it duplicates the backend
  selection logic and multiplies the state space (two backend attributes that
  must stay mutually exclusive), and the plan already requires the both-supplied
  error (FR-008) either way.

### D2 — Inference routes only through the frozen path of the byte-level model

- **Decision**: Every backend call uses `frozen=True` (model `__call__`, not
  `forward`).
- **Rationale**: `src/model/rtu.py:261-265` runs under `torch.no_grad()`, returns
  `(sampled_byte, stop)`, and mirrors detached state into `layer.states`
  (`rtu.py:245-246`). The non-frozen path (`rtu.py:254-311`) runs
  `self.optimizer.step()` at `rtu.py:309` and mutates `decay`/projection weights —
  it must be unreachable from inference. This is the construction-level proof of
  the frozen rule (FR-003, SC-004).
- **Alternatives considered**: Calling `model.step(..., frozen=True)` directly.
  Equivalent; the `__call__` signature is the one the standalone UI loop uses
  (`app.py:370`, `app.py:373`), so using it keeps test 1
  (`test_byte_rtu_backend_matches_raw_ui_loop`) a true equivalence check.

### D3 — Prompt stuffing is pinned, not auto-selected

- **Decision**: Fused profiles write `injection_strategy: "prompt"` into config.
- **Rationale**: `select_strategy` (`src/model/memory_injection.py:125-139`)
  already returns the config override verbatim when present and not `"auto"`
  (`:128-132`). The `hidden` strategy is a fixed random projection
  (`memory_injection.py:56`) and the `kv_cache` strategy degrades to prompt
  stuffing anyway (`:104`, `:114`) — both are stubs. Pinning makes the fast path
  deterministic.
- **Alternatives considered**: Leaving `"auto"` and catching stubs at runtime.
  Rejected: `select_strategy` returns `hidden` for 5–20 facts, which would make
  behavior depend on fact count and inject a random projection.

### D4 — Query goes last in the primed prompt

- **Decision**: Use `inject_prompt` unchanged; never reorder its
  `Context: … \n\nQuery: x` layout.
- **Rationale**: The RTU state is an EMA whose per-dimension half-lives span
  1…spread (`src/model/rtu.py:117-119`); the tail of the byte stream dominates
  the state. `inject_prompt` already ends with the query (`memory_injection.py:31-33`).
- **Alternatives considered**: Query-first. Rejected on decision 4 of the plan;
  the cache/prefix structure of this model has no attention fallback, so the
  question must own the final state.

### D5 — Persistence: one extra file in the profile dir, manifest written last

- **Decision**: `rtu.safetensors` (via `Model.save`, `rtu.py:327-353`, which is
  atomic through a temp file + `os.replace` at `:349-353`) plus three manifest
  fields (`rtu_checkpoint`, `injection_strategy`, `rtu_dim`/`rtu_layers`). Save
  order: knowledge → RTU checkpoint → `config.json` last. Load: after the memory
  load, re-attach the backend; raise when the manifest declares a checkpoint
  whose file is absent.
- **Rationale**: `Model.load` no-ops on a missing file (`rtu.py:356-357`), so a
  declared-but-absent checkpoint would silently run a random component — the
  exact silent-degradation class the spec forbids (FR-007).
- **Alternatives considered**: Reusing `memory.safetensors` (shared tensor
  namespaces across two different `nn.Module`s) — rejected; separate files keep
  ownership clean (`memory.safetensors` = plastic memory, `rtu.safetensors` =
  generative component).

### D6 — Associative bonus re-ranks behind the dense gate

- **Decision**: After `q_emb` normalization (`droid.py:521`), compute
  `q_assoc = normalize(memory.retrieve(q_emb))` inside a guard; per candidate,
  add `0.15 * max(0, dot(q_assoc, fact_emb))` to the existing similarity.
- **Rationale**: `retrieve` exists (`rtu.py:626-656`: `S @ q` per head plus
  `0.1 * h_trace`) but is never called. With an empty `S` it returns
  `0.1 * h_trace`, so the bonus is naturally near zero on a cold start (test:
  no-regression, FR-011). Applying it only to candidates that already pass
  `sim >= 0.48` (`droid.py:564`) keeps the gate authoritative (FR-010, SC-010).
- **Alternatives considered**: Full two-channel plastic re-ranker with fitted
  features (09 Phase 6). Deferred by scope; this is the minimal channel-A wiring.

### D7 — Dimensional contract

- **Decision**: Fused backend uses `dim=384, layers=8, spread=32`.
- **Rationale**: 384 matches the MiniLM anchor output and `h_trace` width, so the
  associative bonus (D6) and any state bridging are dimensionally well-defined.
  The standalone raw mode keeps its own dim/layers widgets (`app.py:209-215`) —
  untouched.

### D8 — Blocker fix borrowed from plan 09 Phase 0 (one item only)

- **Decision**: Add `clear_memory()` to `DroidEngine` (calls
  `self.knowledge.clear_all()` at `knowledge_store.py:755`, `self.memory.reset()`,
  zeroes `step_count`) plus a `clear_knowledge = clear_memory` alias.
- **Rationale**: `scripts/verify_remote_sensing_e2e.py:18` and `app.py:198` both
  call `clear_knowledge()`, which does not exist on `DroidEngine` (verified by
  grep). The E2 verification gate (SC-007) cannot run without this. It is the
  first task of plan 09 Phase 0 and is scope-free: it adds an entry point, it
  renames and restructures nothing.
- **Alternatives considered**: Patching the E2E script and WebUI to call
  `memory.reset()` + manual knowledge clear inline. Rejected: it leaves two
  broken public entry points in place and the fix is smaller than the workaround.

### D9 — Latency caps with a measured gate

- **Decision**: Cap primed prompt bytes and generated output bytes in the
  backend; measure one fused `/slow` answer; if > 10 s, lower
  `max_prompt_bytes` and record the ceiling in the plan file.
- **Rationale**: Priming is a Python loop over bytes × layers of `frozen=True`
  calls. The byte cost per call is small but the loop is interpreted; the 10 s
  budget (SC-008) is the acceptance gate, and the plan file must carry the
  measured number before the feature is called done.
- **Alternatives considered**: Vectorized prefill. Rejected: the state recurrence
  is sequential by construction (EMA over the byte stream); no correctness-safe
  batching exists without changing the model.

### D10 — Verification compares state, not text

- **Decision**: Equality assertions use parameter/state checksums; content
  assertions use taught-fact substring presence.
- **Rationale**: `Model.sample` is stochastic (`rtu.py:228-233`, temperature
  modulated by measured entropy), so text is non-deterministic. The frozen rule
  (FR-003/SC-004) is provable only on tensors; grounding (FR-004/SC-001) is
  provable on substring presence.

## Open items

None blocking. Three plan-level open questions remain (prompt-byte ceiling,
`stop_threshold` configurability, future rename target) — all are recorded as
assumptions or plan-file notes, not spec clarifications.
