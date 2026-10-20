# Design

## Context

This change is the "correctness pass, no rename" slice of plan 09 Phase 0. It touches
three live paths in `DroidEngine`: the teach path (plastic write), the recall path
(candidate ranking), and the profile path (save/load). It also touches the anchor
fallback and the knowledge-store writer.

Line references below were re-verified against this checkout. Several plan-09 and
plan-10 citations are off by a few lines because they were taken before T001–T011
landed; the code is authoritative.

| Concern | Current code | Observed behavior |
|---|---|---|
| Effective LR | `src/model/droid.py:455` (compute), `:524` (report) | `update_associative_memory()` has no learning-rate input |
| Plastic writes | `src/model/rtu.py:548-620` | All updates run under `torch.no_grad()` |
| Dead retrieval | `src/model/rtu.py:626-655` | No `DroidEngine` call site |
| Shadowed block | `src/model/droid.py:164-203`, import at `:209` | Local class wins over `rtu.RTUMemoryBlock` |
| Duplicate error vector | `src/model/rtu.py:539`, `:544` | Second assignment always overwrites |
| Capacity | `src/model/droid.py:346` (field), `:796` (saturation only) | No eviction anywhere |
| Ranking | `src/model/droid.py:531-611` | Hardcoded `0.25` / `0.2` / `0.3` / `0.2` / `0.2` terms |
| Anchor fallback | `src/model/onnx_anchor.py:103-110`, `:241-244` | `use_mock = True`, no warning |
| UI badge | `app.py:114` | Hardcoded "ONNX MiniLM (INT8, 22MB) Active" |
| Adapter field | `src/model/droid.py:1268-1281` (write), `:1360-1372` (read) | Write exists; read of `adapter_kwargs` has no writer |
| Store writes | `src/model/knowledge_store.py:62`; `memory_lifecycle.py` daemon | `check_same_thread=False`, no lock |
| Dependencies | `requirements.txt` | `psutil` imported by `scripts/verify_reproduction.py`, absent |
| Test count | `README.md:122` | Says 43; suite collects 68 |

## Goals / Non-Goals

**Goals**

- Every plastic update receives and applies the effective learning rate; the applied
  value is recorded per teach call.
- One live `RTUMemoryBlock`, one `error_vec`, and a decided fate for `retrieve()`.
- `K_max` is enforced by a named eviction policy, and evictions are counted.
- Recall ranking is a documented, unit-tested scorer with named signals and sources.
- The mock anchor is impossible to mistake for the real one.
- Profiles persist and restore the adapter type that the loader actually reads.
- Concurrent writers to `KnowledgeStore` serialize through one lock.
- `requirements.txt` covers every imported runtime dependency; README claims match the
  code.

**Non-Goals**

- No rename (`RTU`, `DroidEngine`, `DroidManager`, `droids/` stay).
- No new embedder, no ANN index, no cross-encoder reranker.
- No power-law forgetting, no utility pruning, no sleep consolidation.
- No change to `Model.forward`'s return signature.
- No new profile manifest fields beyond `plastic_adapter` (the fusion change owns the
  engine/checkpoint fields).

## Decisions

### Decision 1 — The schedule scales the write; it is not a report field

`effective_lr` is passed into the plastic update as an explicit argument. For
`PlasticAssociativeRTU` the update becomes a scaled write:
`S_t = S_{t-1} G_t + M_t` where the momentum term is multiplied by
`(effective_lr / base_lr)` and the trace update keeps its own `0.9/0.1` constants so the
1D summary is not rescaled twice. The engine keeps `base_lr = 0.05` as the reference
scale, so `effective_lr / base_lr == 1` at `step_count == 0` and decays from there.

Alternatives rejected:

- *Create an optimizer over the associative state.* The state is a buffer, not a
  parameter, and every existing test asserts the closed-form update.
- *Multiply the whole state by the rate.* That is a decay, not a learning rate, and it
  would shrink old associations on every teach call.

The applied rate is added to the teach result and to the `train_log.json` entry as
`applied_lr`, next to the existing `effective_lr` key (kept for compatibility).

### Decision 2 — Keep `retrieve()`, call it from `recall()`, and delete only proven dead code

`retrieve()` is the only existing semantic read of the plastic state
(`S @ W_Q(q) + 0.1 h_trace`). Deleting it would force the fusion change to reintroduce
the same math. It is called once per recall (guarded, `None` on failure) and exposed to
the scorer as the `associative` signal with a documented weight of `0.0` in this change.
Zero weight keeps this change behavior-preserving; the fusion change raises the weight
and owns the re-ranking scenario. If review finds the call carries no signal even at
weight 0 (that is, it is still unreferenced after the change), the fallback is deletion
as 2.3 allows.

The shadowing `RTUMemoryBlock` in `droid.py` is deleted, and the import at `:209` stays.
The duplicate `error_vec` assignment is collapsed to `e_norm - hat_e` with the reasoning
comment retained once.

### Decision 3 — Capacity eviction is weakest-key-first, counted, and observable

`K_max` bounds the knowledge store today. The associative update additionally keeps a
key register: on `update_associative_memory`, the incoming key's projection norm is
recorded; when the register reaches `K_max`, the key with the smallest projection norm
(and therefore the weakest stored association) is removed by attenuating its direction,
and `evictions` is incremented. The counter is exposed as
`memory.eviction_count` and surfaced by `get_memory_health()["evictions"]`.

Alternatives rejected:

- *FIFO eviction.* Ignores association strength; the oldest fact is not the weakest.
- *Hard delete of `S` rows.* Destroys correlations sharing the row direction; the
  attenuation reuses the existing Householder-like erasure path.

### Decision 4 — A named, weighted scorer with a pinned order

Ranking becomes a pure function:

```python
def score_candidate(signals: dict[str, float], weights: "ScoreWeights") -> float
```

Each signal carries its weight and source in one table (module constant
`RECALL_SCORE_WEIGHTS`), so a reader can audit the composition without reading `recall()`:

| Signal | Weight | Source |
|---|---|---|
| `dense` | 1.0 | `KnowledgeStore.recall()` dense similarity (`dense_similarity`) |
| `lexical` | 0.0–0.5, capped | query/fact content-word overlap, stop-list filtered |
| `definitional` | 0.25 | query classified `definitional` by `RetrievalPolicy` and fact matches a copula pattern |
| `associative` | 0.0 | `PlasticAssociativeRTU.retrieve(q_emb)` dot product (fusion change raises it) |
| `recency` | 0.0 | fact timestamp half-life (documented exponent, disabled at weight 0) |
| `confidence` | 0.0 | stored fact novelty/confidence (disabled at weight 0) |
| `contextual` | multiplicative `(1 + 0.2 · max(0, c))` | dot of normalized `h_trace` with fact embedding |

Weights that are zero in this change exist as named slots so the fusion change can set
them without inventing a second scorer. `tests/test_recall_scoring.py` pins the order
for a fixed fixture: a definitional fact beats a higher-dense non-definitional fact;
two facts with equal dense and equal lexical overlap keep insertion order; a query with
no overlap never ranks a below-gate fact above the `sim >= 0.48` dense gate.

The `sim < 0.48` gate stays authoritative and is applied to the dense signal before any
other signal contributes.

### Decision 5 — Adapter type is written and read as the same field

Save already writes `plastic_adapter` (`droid.py:1268-1281`). Load now reads that field
first, then falls back to the current class-name inference when it is absent. The
non-existent `adapter_kwargs` read is replaced by a read of `adapter_kwargs` **only if
the config contains it** (new saves write the kwargs they were constructed with), so a
profile can round-trip a non-default adapter without silently reverting to
`associative_rtu`.

### Decision 6 — Mock anchor is a first-class state

`OnnxMiniLM` sets `self.is_mock = False` in `__init__`, raises a `warnings.warn(...)`
and sets `self.is_mock = True` on every fallback path, and keeps `use_mock` as an alias
for existing callers. `app.py` prints a success badge only when `is_mock` is false; the
mock path prints a warning badge that names the reason. The engine records
`self.anchor_is_mock` so CLI and MCP output can carry it too.

### Decision 7 — One writer lock per store, not per call site

`KnowledgeStore` owns `self._write_lock = threading.RLock()`. Every method that opens a
write transaction acquires it. `ContinuousStream`'s daemon calls the same public store
methods, so it inherits the lock instead of inventing a second one.

Alternatives rejected:

- *Lock in `ContinuousStream` only.* Other writers (Streamlit handlers, MCP server,
  CLI) still race.
- *Serialize by opening one connection per thread.* SQLite WAL allows it, but the dense
  vector cache (`_vec_cache`, `_id_cache`) is shared mutable state and would still race.

Test 2.14 starts one writer on the main thread and one on the stream thread, writes a
known number of facts, and asserts the count and that no exception escaped either
thread.

### Decision 8 — Dependencies and README are part of correctness

`psutil` and `numpy` are declared in `requirements.txt` (`numpy` is imported directly by
the anchor path; `psutil` by `scripts/verify_reproduction.py`). The README test count is
replaced by the collected count, and each unsupported claim is either corrected or
removed. The claims reviewed for 2.17 are: the "43 tests" count, the "zero catastrophic
collapse" claim (no forgetting measurement exists), "sub-15ms CPU" (not measured in this
repo), and "22 MB on disk" (the anchor cache size is environment-dependent).

## Risks / Trade-offs

- **Ranking churn.** Even a faithful port of the old terms into a scorer can reorder
  results, because the old code added the definitional term before the gate and the new
  code applies the gate to the dense signal first. Mitigation: the gate ordering matches
  the old intent, the fixture pins the documented order, and the E2E script runs before
  the change closes. Any regression is reported as a finding.
- **Learning-rate application destabilizes long teaching runs.** Mitigation: scaling is
  relative to `base_lr`, the trace constants are unchanged, and the applied rate is
  logged so a run can be replayed.
- **Eviction destroys a usable association.** Mitigation: evict only at the bound, use
  the existing attenuation operator, count every eviction, and test that a key below the
  bound is never evicted.
- **Lock contention on the Streamlit thread.** The lock is held per write, not per
  session; the expensive work (embedding) happens outside it.
- **The `adaptation/plastic-memory-titans` base spec is structurally invalid today.**
  It contains a `## MODIFIED Requirements` header inside the main spec, so its last
  requirement is invisible to `validate`/`archive` (pre-existing; this change does not
  edit `openspec/specs/`). The delta for "Capacity bound enforcement" is therefore a
  MODIFIED delta against a requirement that the main spec does not currently parse.
  Archive will refuse this capability until the base spec is repaired; that repair is
  recorded here as a follow-up, not performed in this change.
- **`droid/profile-management` and `model/dual-mode-generation` are also modified by the
  fusion change.** Both changes touch the same requirements from different angles
  (adapter field vs engine/checkpoint fields; scorer vs associative bonus). The ordering
  2 → 1 is mandatory; the later change must preserve the earlier requirement text.
