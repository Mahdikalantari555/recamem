# Tasks

## 1. Supersede helper
- [x] 1.1 Add `_policy_update_targets(text)` extracting (soil, phase) from a proposition.
- [x] 1.2 In `_teach_impl`, after inserting facts, if proposition is a policy update, supersede prior active facts matching soil+phase+threshold.
- [x] 1.3 Ensure superseded facts remain in audit trail.

## 2. Benchmark harness
- [x] 2.1 Create `scripts/run_vela_benchmark.py` with conditions A/B/C/D/E.
- [x] 2.2 Implement teaching of T1-T10 and V1-V12 evaluation.
- [x] 2.3 Implement decision extraction and scoring.
- [x] 2.4 Write JSON report with provenance.

## 3. Tests
- [x] 3.1 Add `tests/test_vela_benchmark.py` for supersede update and scoring helpers.

## 4. Verification
- [x] 4.1 Run harness on MiniCPM and inspect report.
- [x] 4.2 Run `PYTHONPATH=. python -m pytest tests/test_vela_benchmark.py -q`.
