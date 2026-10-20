# Tasks

Task IDs match this change's numbering. Group headers are descriptive so the validator
does not read them as a second numbering scheme.

## Module coverage

- [x] 3.1 Add `tests/test_confidence.py` for `src/model/confidence.py`. Cover component bounds, the harmonic-mean fusion formula, the 0.5/0.8 tier boundaries, and the low-confidence clarification prefix.
- [x] 3.2 Add `tests/test_decision_head.py` for `src/model/decision_head.py`. Cover prototype argmax, entropy-derived confidence bounds, out-of-domain rejection, and JSON-serializable structured output.
- [x] 3.3 Add `tests/test_memory_lifecycle.py` for `src/model/memory_lifecycle.py` (614 lines). Cover snapshot create/restore/list, rollback equality, branch independence, merge conflict detection with each policy, and stream pause/resume with a fake clock.
- [x] 3.4 Add `tests/test_memory_injection.py` for `src/model/memory_injection.py`. Cover the prompt layout with the query last, the byte-backend `tokenize()` fallback, a pinned config returning `prompt`, and explicit handling of the hidden/KV stubs.
- [x] 3.5 Add `tests/test_propositions.py` for `src/model/propositions.py`. Cover sentence splitting, length/word filters, anaphora resolution, and stable output for a fixed paragraph.
- [x] 3.6 Add `tests/test_droid_package.py` for `src/model/droid_package.py`. Cover export/import round trip in `tmp_path`, manifest digests, tampered-member rejection, and `../` rejection.
- [x] 3.7 Add `tests/test_gguf_backend.py` for `src/model/gguf_backend.py`. Use a stub `llama_cpp` module: cover constructor handling, `generate()` call shape, download failure, and import without the package installed.
- [x] 3.8 Add `tests/test_plastic_adapter.py` for `src/model/plastic_adapter.py`. Cover registry registration, listing, construction by name, unknown-name error, and state round trip.
- [x] 3.9 Add `tests/test_cli.py` for `src/cli.py`. Cover subcommand dispatch, `eval` output shape, unknown subcommand exit code, and the stdin teach path with a fake engine.
- [x] 3.10 Add `tests/test_mcp_server.py` for `src/mcp_server.py`. Cover `initialize`, the `tools/list` name set, `tools/call` for each tool against a temp profile, and the JSON-RPC error envelope.

## Contract conformance

- [x] 3.11 Fix `tests/test_rtu.py::test_model_forward_accepts_ce_only`. The test destructures the second return value of `Model.forward` as a loss. `Model.forward` returns `(sampled_byte, stop_value)`. The loss lives in `model.last_loss`.
- [x] 3.12 Assert `model.last_loss` equals the cross-entropy term. Keep the assertion on `step_count == 1` and on the trace buffers. Compute the expected CE independently from `compute_losses(..., ce_only=True)` and compare with `pytest.approx`.
- [x] 3.13 Do not change the return signature of `Model.forward`. The MLX reference returns `(sample, stop)` at `main_mlx.py:114` and `main_mlx.py:172`. Record that reasoning in the test docstring so the next reader does not "fix" it.
- [x] 3.14 Add a test that lists the modules that have no test file. Fail the test when the list grows. Put `MODULE_TESTS` in `tests/test_surface_coverage.py`, assert every mapped file exists and contains a test, and fail with the untested module and its expected filename.


## Implementation notes

Recorded on completion (2026-10-20); the tests themselves live in `tests/`.

- **3.3 (stream):** pause/resume/stop, queue accounting and drop behaviour are exercised
  without sleeping; the worker loop's exit path is driven directly and the one threaded
  test waits on a condition with a hard deadline. No fake clock is needed because the
  rate limiter only resets on a one-minute boundary that the tests do not cross.
- **3.5 (propositions):** `src/model/propositions.py` has no anaphora rewrite or word-count
  filter - the anaphora regex lives in `src/model/droid.py` and rewrites at teach time.
  The tests pin the implemented split/length/marker contract and say so in their docstring
  rather than inventing a behaviour the module does not provide.
- **Defects found (not patched here; this change is tests-only, per the proposal):**
  1. `PlasticAssociativeRTU.state_dict()` calls `super().state_dict()`, which resolves to
     the abstract `PlasticAdapter.state_dict` (returns `None`) instead of
     `nn.Module.state_dict`, so it raises `AttributeError`. Covered by a strict xfail.
  2. `LegacyRTUAdapter.load_state_dict` copies the exported `states` entry into `h_trace`
     and ignores the exported `h_trace` entry, losing a trace advanced by `update()`.
     Covered by a strict xfail.
  3. `MCPServer._tool_decide_action` builds `DecisionHead(prototypes=...)` without `dim`,
     so anchors whose embedding width is not 384 fail with a `torch.mv` size mismatch.
     Covered by a strict xfail.
  4. `inject_hidden`'s error branch itself calls `torch.norm(h_trace)`, so a non-tensor
     trace re-raises instead of returning the documented prompt fallback.
  5. `RetrievalPolicy.classify_query`'s `ambiguous` branch is unreachable for any query
     containing `?` (the `specific_fact` rule catches those first).

## Integration note (2026-10-20)

PRs #5 and #9 were reconciled together, retaining complementary suites.
The associative-state, legacy-trace and MCP-width expected failures mentioned
above are now fixed and run as ordinary tests. Parallel scan is mapped by the
coverage guard. See `reports/2026-10-20-pr-integration.md` for current validation;
original counts and pre-existing-defect notes above are historical.
