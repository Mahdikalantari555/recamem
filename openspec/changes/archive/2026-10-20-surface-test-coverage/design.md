# Design

## Context

`pytest.ini` sets `pythonpath = .`; there is no `conftest.py` and no shared fixture
module. Tests import from `src.model.*` directly and construct small objects with
`dim=32` or `64`. The suite runs on CPU in the `ai` conda environment.

The modules this change covers are unverified today. Their public surfaces differ: two
are pure functions over tensors (`confidence`, `decision_head`), one is a registry
(`plastic_adapter`), three are storage/format helpers (`memory_lifecycle`,
`droid_package`, `propositions`), two are process boundaries (`cli`, `mcp_server`), one
is a third-party wrapper (`gguf_backend`), and one is a prompt formatter
(`memory_injection`).

The contract bug is concrete: `tests/test_rtu.py:106-120` unpacks `(sample, stop)` as
`(_, loss)` and then compares `model.last_loss` to that stop value.

## Goals / Non-Goals

**Goals**

- Every module under `src/` has a test file that exercises its public contract.
- The `ce_only` test asserts what the code actually returns.
- A guard test makes a new untested module a failing test, not a review miss.

**Non-Goals**

- No production code changes; bugs found are reported to the owning change.
- No Streamlit UI tests; `app.py` is validated manually in change 1 and change 5.
- No network, no model downloads, no `llama-cpp-python` requirement.
- No coverage-percentage target; the unit is "has a contract test", not line coverage.
- No refactor of the modules under test to make them testable; where a module needs a
  seam, the test uses monkeypatching at the module boundary.

## Decisions

### Decision 1 — Module-to-file matrix

| Module | Test file | Contract under test |
|---|---|---|
| `src/model/confidence.py` | `tests/test_confidence.py` | component scores in `[0,1]`, harmonic-mean fusion matches the documented formula, tier boundaries at 0.5/0.8, low tier carries the uncertainty prefix |
| `src/model/decision_head.py` | `tests/test_decision_head.py` | intent argmax over prototypes, entropy confidence bounded, below-threshold query returns the ambiguous status, structured output keys are JSON-serializable |
| `src/model/memory_lifecycle.py` | `tests/test_memory_lifecycle.py` | snapshot create/restore/list metadata, rollback equality on state and facts, branch independence, merge conflict detection and each policy, stream pause/resume and rate accounting |
| `src/model/memory_injection.py` | `tests/test_memory_injection.py` | `prompt` renders `Context:` then `Query:` last, `tokenize()` fallback for a byte backend, strategy function returns `prompt` for a pinned config, hidden/KV paths are marked stubbed rather than silently wrong |
| `src/model/propositions.py` | `tests/test_propositions.py` | sentence split, length/word filters, anaphora resolution, stable output for a fixed paragraph |
| `src/model/droid_package.py` | `tests/test_droid_package.py` | export/import round trip in `tmp_path`, manifest digests present, tampered member rejected, `../` member rejected |
| `src/model/gguf_backend.py` | `tests/test_gguf_backend.py` | constructor argument handling, `generate()` call shape with a fake `llama` module, download path failure returns a clear error, no import-time crash when `llama_cpp` is absent |
| `src/model/plastic_adapter.py` | `tests/test_plastic_adapter.py` | registry registration/listing/instantiation, unknown name raises, state dict round trip |
| `src/cli.py` | `tests/test_cli.py` | subcommand dispatch table, `eval` routes to its handler and returns an exit code, unknown subcommand exits non-zero, stdin teach path with a fake engine (the eval metric format is asserted by the change that implements `cmd_eval`) |
| `src/mcp_server.py` | `tests/test_mcp_server.py` | `initialize`, `tools/list` name set, `tools/call` for each tool with a temp profile, JSON-RPC error envelope for an unknown tool |

Each file seeds `torch.manual_seed` where tensors are random, and each uses `tmp_path`
for anything that writes.

### Decision 2 — Assert the return contract, not a convenient unpacking

`Model.forward` returns `(sampled_byte, stop_value)`. The CE-only test becomes:

```python
sample, stop = model(10, nextb=11, end=False, frozen=False, ce_only=True)
assert isinstance(sample, int) and 0 <= sample < model.vocab_size
assert 0.0 <= stop <= 1.0
assert model.last_loss == pytest.approx(expected_ce, rel=1e-6)
assert model.step_count == 1
assert not torch.allclose(model.blocks[0].states, before_states)
```

`expected_ce` is computed independently from `compute_losses(..., ce_only=True)` on the
same tensor, so the test pins the number rather than echoing the attribute.

Why not change the signature to return the loss: the MLX reference returns
`(sample, stop)` at `main_mlx.py:114` and `main_mlx.py:172`, the byte-level call sites
depend on the pair, and `last_loss` already exists for the training surface. Changing the
signature would break the "1:1 contract-verified port" claim for a test's convenience.

### Decision 3 — The coverage guard is a mapping, not a filename heuristic

`tests/test_surface_coverage.py` holds an explicit `MODULE_TESTS` dict mapping each
`src/` module to the test file that covers it (the matrix above plus the existing
files: `rtu.py` → `test_rtu.py`, `droid.py` → `test_droid_engine.py`, `droid_manager.py`
→ `test_droid_profiles.py`, `knowledge_store.py` → `test_knowledge_store.py`,
`onnx_anchor.py` → `test_onnx_anchor.py`, `teacher.py` → `test_teacher.py`, `data.py` →
`test_data.py`, `byte_rtu_backend.py` → `test_droid_rtu_fusion.py`).

The test walks `src/**/*.py`, skips `__init__.py`, and fails with a message listing any
module missing from the mapping:

```
Untested modules (add a test file or an entry to MODULE_TESTS):
  src/model/new_thing.py -> expected tests/test_new_thing.py
```

It also asserts that every mapped test file exists and contains at least one
`def test_`. A filename heuristic was rejected: `test_droid_engine.py` covers
`droid.py`, and a heuristic would force renames with no benefit.

### Decision 4 — Test boundaries for process and third-party modules

- `gguf_backend`: monkeypatch `sys.modules["llama_cpp"]` with a stub exposing
  `Llama`. No download, no file larger than a few bytes.
- `mcp_server`: drive `handle_initialize`, `handle_tools_list`, `handle_tools_call`
  directly; do not start the stdio loop. One test serializes a response and asserts it
  is valid JSON-RPC.
- `cli`: call `cmd_*` functions with a `SimpleNamespace` argparse result or run `main`
  with `monkeypatch.setattr(sys, "argv", ...)`; no subprocess, no daemon.
- `memory_lifecycle` stream: use a fake clock and a small rate limit rather than
  `sleep`.

### Decision 5 — Determinism and time budget

Random tensors are seeded; assertions use tolerances (`rtol=1e-5` for float32 cosine
math) rather than exact equality except for integer counts and shapes. No wall-clock
assertion is added in this change. The target is under 60 seconds for the ten new files
combined on the reference machine; a slow test is rewritten, not marked slow.

## Risks / Trade-offs

- **Behavior-pinning tests.** A test written against a 614-line module can freeze a bug.
  Mitigation: each test names the requirement or invariant it asserts in its docstring,
  and findings are reported rather than enshrined.
- **Mocking too much.** A GGUF test that stubs `llama_cpp` proves the call shape, not
  inference. That is the honest limit and is stated in the test docstring.
- **Coverage guard churn.** Every new module needs a mapping entry. That is the intended
  cost: the failure teaches the reviewer where the test belongs.
- **Optional imports.** `src/model/gguf_backend.py` imports `llama_cpp` lazily; a test
  that imports the module must not require the package. If the import is not lazy
  today, that is a finding for change 5 rather than a production edit here.
