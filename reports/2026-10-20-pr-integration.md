# PR integration verification — 2026-10-20

Integrated PR heads #8 (correctness hardening), #5 and #9 (surface coverage)
with merge commits on `arena/c284cedc-test-model-thing-torch`, based on main
`7b09561`. Both coverage implementations are retained: #5's canonical files
and #9's complementary `test_*_additional.py` files. The coverage guard maps
the already-merged parallel scan module.

## Conflict decisions

- Preserve fusion weights: dense 1.0, lexical 0.70, associative 0.15,
  recency 0.02, confidence 0.05, contextual 0.20. Definitional prior remains
  part of the lexical signal. Production recall uses the shared pure scorer;
  dense-gate status is the primary ordering key. A production-path regression
  checks this, in addition to scorer and existing fusion tests.
- Preserve profile artifact ordering, checkpoint validation, fused UI, and
  replacement-fact validation. Add the hardening write lock around superseding.
- Keep one mock-anchor state plus its compatibility alias and fallback warning.
- Update obsolete zero-weight/count assertions to the combined contracts;
  README uses a collection command rather than an inevitably stale count.
- Remove the already-fixed associative-state strict xfail. Fix legacy adapter
  trace restoration and non-384 MCP option routing, promoting both expected
  failures to normal regression tests.
- Fix negative decision confidence uncovered by the combined suites; add
  antipodal-vector tests with positive, zero, and negative rejection thresholds.

## Reproduced verification

Sandbox: Python 3.11.2, PyTorch 2.2.2, NumPy 1.26.4, CPU. The workstation's
Conda installation was unavailable; a disposable virtualenv was provisioned
at the repository-prescribed `/home/asus/miniforge3/envs/ai` path. No environment
or model weights are committed.

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  /home/asus/miniforge3/envs/ai/bin/pytest tests -q
# 441 passed, 1 skipped; no xfails (7.12 seconds)

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 PYTHONPATH=. \
  /home/asus/miniforge3/envs/ai/bin/python scripts/verify_remote_sensing_e2e.py
# ALL END-TO-END VERIFICATION CHECKS PASSED

npx --yes @fission-ai/openspec validate --all --no-interactive
# 35 items passed, 0 failed; long-requirement warnings only
```

The skipped test requires `RTU_CHECKPOINT`; trained grounding was not verified.
The anchor used deterministic mock embeddings, with explicit warnings. These
results do not establish real-ONNX/GGUF quality, general no-forgetting claims,
or target-device performance. Original PR bodies/archive results are historical
and their test counts are superseded by this report.

## Remaining work, not certified by this merge

- Three active feature proposals: delta/sparse memory, autonomous sleep/MAS,
  edge retrieval/model catalog/test-time compute.
- Historical archived requirements need a task-to-code/test evidence ledger;
  unchecked archived tasks must not be silently declared complete.
- Adapter API defaults/nn.Module inheritance and strict custom-adapter identity,
  LR capability negotiation, error propagation, and persisted capacity metadata
  need completion beyond the default-adapter tests.
- Hidden/KV injection still uses placeholder behavior; fallback error handling
  and query-classifier ambiguity require explicit contracts and tests.
- Live model, checkpoint-grounding, concurrency/lifecycle, and benchmark gates
  must be recorded separately from offline contract tests.
