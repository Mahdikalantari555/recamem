# Proposal

## Why

The suite is 16 files and 68 test functions. Ten shipped modules have no test file at
all:

| Module | Lines | Role |
|---|---|---|
| `src/model/confidence.py` | 127 | fused retrieval/generation confidence |
| `src/model/decision_head.py` | 218 | System-1 routing |
| `src/model/memory_lifecycle.py` | 614 | snapshots, merge, semantic store, stream |
| `src/model/memory_injection.py` | 140 | prompt/hidden/KV injection strategies |
| `src/model/propositions.py` | 154 | proposition extraction and normalization |
| `src/model/droid_package.py` | 181 | `.droid` archive export/import |
| `src/model/gguf_backend.py` | 184 | GGUF load/generate/download |
| `src/model/plastic_adapter.py` | 133 | adapter registry and base contract |
| `src/cli.py` | 483 | headless entrypoint |
| `src/mcp_server.py` | 402 | MCP stdio server |

Those modules are reachable from the product surface the other changes in this set
touch: the CLI and MCP server (change 5), the profile archive, the injection strategy
(change 1), and the health/merge UI. A change that edits them without a test has no
regression signal.

One existing test also asserts the wrong contract.
`tests/test_rtu.py::test_model_forward_accepts_ce_only` destructures the second return
value of `Model.forward` as a loss:

```python
_, loss = model(10, nextb=11, end=False, frozen=False, ce_only=True)
assert model.last_loss == loss
```

`Model.forward` returns `(sampled_byte, stop_value)` (`src/model/rtu.py:236-320`); the
loss lives in `model.last_loss`. The assertion compares the loss to a stop probability
and either fails or passes by coincidence.

## What Changes

- Add one test file per untested module: confidence, decision head, memory lifecycle,
  memory injection, propositions, droid package, GGUF backend, plastic adapter, CLI, MCP
  server (3.1–3.10).
- Fix `test_model_forward_accepts_ce_only` to assert `model.last_loss` against the
  cross-entropy term, keeping the `step_count == 1` and trace-buffer assertions
  (3.11–3.12).
- Do not change the return signature of `Model.forward`; the MLX reference returns
  `(sample, stop)` at `main_mlx.py:114` and `main_mlx.py:172` (3.13).
- Add a guard test that lists modules without a test file and fails when the list grows
  (3.14).

## Capabilities

### New Capabilities

- `verification/surface-coverage`: a module-to-test coverage contract, and a conformance
  rule for tests that assert a return contract.

### Modified Capabilities

None.

## Impact

- **Code areas:** `tests/` only. No production module changes in this change; a defect
  discovered by a new test is reported and fixed in the change that owns the module
  (1, 2, or 5), not silently patched here.
- **Test count:** 68 → at least 100, and the README count is owned by
  `correctness-hardening` (2.16); this change updates the number again if it lands
  first.
- **Runtime:** every new test must be CPU-only, offline, deterministic, and use
  temporary directories; the full suite stays under two minutes on the reference
  machine.
- **Ordering:** after `droid-rtu-fusion-completion`, because the new tests cover modules
  that change touches, except `rtu.py` and `byte_rtu_backend.py`, which get their own
  tests in change 1.

## Risks

- Tests over a 614-line module can encode current behavior rather than intended
  behavior. Mitigation: assert public contracts and documented invariants, not internal
  call sequences.
- The anchor and GGUF paths touch optional dependencies. Mitigation: mock at the module
  boundary; no network, no model download, no `llama-cpp-python` import requirement.
- A coverage guard can become a maintenance tax. Mitigation: one explicit mapping table,
  a failure message that names the module and the expected filename, and an empty
  allowlist at close.
- Fixing the `ce_only` test may reveal that the CE-only path does not populate
  `last_loss` as documented. That finding is reported with the test fix rather than
  weakening the assertion.
