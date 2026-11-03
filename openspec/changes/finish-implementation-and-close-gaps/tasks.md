# Tasks

> **Progress (2026-11-03).** Phase 1 of this change is implemented: the
> completion inventory and guard, all offline correctness/repository-hygiene
> fixes, core-config parity and the real-model runbook. 19 of 62 inventory
> rows are `closed` with tests and evidence in
> `reports/evidence/2026-11-03-offline-closure.json`. The remaining rows need
> model weights, a trained checkpoint, benchmark hardware or a compiled
> desktop shell; `check_completion_gaps.py` still exits 1 by design. A checked
> box below is not evidence; the inventory rows are.

All boxes start unchecked. A checked box is not evidence: every task names the
command or artifact that must exist, and `scripts/check_completion_gaps.py`
rejects a `closed` row without a test and an evidence record.

## 1. Baseline and inventory

- [x] 1.1 Create `reports/completion-gaps.json` with one row per known gap: every unresolved ledger row, every unchecked task in the two active changes, every open task in `2026-10-28-post-merge-completion`, every unmet target in `docs/release-matrix.md`, every Vela open question, every doc defect found, and every unchecked item in `docs/surfaces.md`; each row carries id, source, owner, tier, verify_command, artifact, test and status
- [x] 1.2 Create `scripts/check_completion_gaps.py` that validates the inventory schema, rejects duplicate ids, rejects a `closed` row whose artifact or test is missing, and exits 1 when any row is not closed; add `tests/test_completion_gaps.py`
- [x] 1.3 Run `PYTHONPATH=. python -m pytest tests -q` on a clean `pip install -r requirements-core.txt` environment and record the full result (count, failures, commit, environment) in `reports/evidence/<date>-offline-baseline.json`
- [x] 1.4 Reconcile the "10 pre-existing failures" note in `recamem-language-learning/tasks.md#7.2` against the fresh run: triage each failure into a real defect, an environment requirement, or a stale note, and add an inventory row for each real defect
- [x] 1.5 Re-run `scripts/check_release_evidence.py` and `--require-complete` and record both outputs as the pre-change baseline
- [x] 1.6 Add the inventory guard to `.github/workflows/offline-contracts.yml` and to `scripts/offline_gate.py`'s documented check list

## 2. Offline correctness and repository hygiene

- [x] 2.1 Fix every real test failure found in 1.3/1.4 and add one regression test per fix that fails on the old behaviour; record the fixed test ids in the inventory
- [x] 2.2 Repair `notebooks/recamem_colab.ipynb` into valid `nbformat` JSON (remove the trailing heredoc marker) and add a test that `json.load` succeeds and that the notebook has the install, configure, train and push cells
- [x] 2.3 Remove the trailing shell text from `GLOSSARY.md` and add a Markdown hygiene test that rejects stray `EOF`/`echo` lines in tracked `.md` files
- [x] 2.4 Fix `plans/README.md` to reference `05_phase9_gate.md` and add a link-existence test for every `plans/*.md` backtick reference
- [x] 2.5 Replace hardcoded developer-machine absolute interpreter paths in `AGENTS.md`, `docs/release-evidence.md` and `plans/*.md` with portable commands, and add a test rejecting machine-specific absolute paths in agent-facing docs
- [x] 2.6 Remove `|| true` from the E2E smoke step in `.github/workflows/ci.yml`, keep the informational `hnswlib` step explicitly non-gating, and add a test that no verification command in a workflow is masked
- [x] 2.7 Resolve the stale "one decision is still open and blocks Phase A" note in `plans/README.md` against `plans/01_source_analysis.md` and record the resolution
- [x] 2.8 Reconcile `docs/testing.md`'s claim that core tests run without `psutil` with `requirements-core.txt` (which declares it), and fix whichever side is wrong
- [x] 2.9 Fix the `/mnt/d/AI/.lmstudio/...` provenance path and `bde46c1` commit reference in `docs/vela-poc-results.md` / `reports/vela-poc-final.json` so the report names a reachable artifact or explicitly says the run was performed elsewhere
- [x] 2.10 Verify `pyproject.toml` package list (duplicate `src.model.tournament` entry, missing `recamem.bench` subpackages if any) against the importable tree and pin it with a packaging test

## 3. Core config and evidence parity

- [x] 3.1 Write a parity test comparing `configs/recamem_selected_core.json` to `reports/recamem-core-selection.md` and `reports/evidence/recamem-core-tournament.json` for selection status, winning stack ids, mode, calibration, tier and dataset provenance
- [x] 3.2 Regenerate `configs/recamem_selected_core.json` from the tournament evidence so `selection_status`, `memory_arm_id` and `multitask_batching_mode` match the selection report, and keep `target_17m_hyperparameters.trained` false until a real run exists
- [x] 3.3 Add a `src/model/recamem_core.py` loader regression test that pins the generated config contract and fails on silent reinterpretation of `memory_arm_id`
- [x] 3.4 Document the config-as-generated-artifact rule in `docs/core-candidates.md` and `docs/architecture-tournament.md`
- [ ] 3.5 Verify `recamem-language-learning` tasks 1.2, 4.1-4.6, 5.3-5.5, 6.2-6.5 and 7.4 are complete with passing evidence, and reference their evidence records from the inventory
- [ ] 3.6 Verify `recamem-core-architecture-tournament` has passing evidence for its checked tasks and archive it only after `openspec validate` passes
- [ ] 3.7 Archive `recamem-language-learning` after G2 (BPB below trigram) and G3/G4 are measured, not merely after the code lands

## 4. Real-model and grounding tiers

- [x] 4.1 Write `docs/real-model-runbook.md` with the exact download, build, checksum and verification commands for ONNX MiniLM, a named GGUF and `llama-cpp-python`, including the hardware/RAM budget
- [ ] 4.2 On a machine with model access, run `scripts/fetch_real_model.py` and record a `downloaded` evidence record with bytes, sha256 and source repo; no weights are committed
- [ ] 4.3 Build `llama-cpp-python` and run `pytest tests -q -k "onnx or gguf"` against the real artifacts, publishing binding version, model sha256, seed, command, commit and hardware
- [ ] 4.4 Verify real GGUF generation, streaming and KV prefill (tasks 3.4, 3.7, 8.1) with a named model path, and assert no `MockLlama` or mock anchor appears in the answer path
- [ ] 4.5 Run `scripts/verify_remote_sensing_e2e.py` with a trained `RTU_CHECKPOINT` for the `grounding` tier and record the raw result, or record `blocked` with the missing checkpoint
- [ ] 4.6 Name the 8B reference model and matched budget for task 7.10, run the 1B+TTC comparison with a statistical test, and publish raw scores and uncertainty; withhold superiority claims if unverified
- [ ] 4.7 Make `recamem doctor` exit 0 only with a real anchor and a named generation backend, and add a test that it exits 1 with the explicit mock
- [ ] 4.8 Close tasks 9.2, 9.4 and 9.5 with cross-feature integration evidence in all five tiers

## 5. Performance targets

- [ ] 5.1 Re-measure all targets in `docs/release-matrix.md` on a declared machine with a provenance-stamped report before changing any implementation
- [ ] 5.2 Build `hnswlib` (or an equivalent declared ANN index) and re-measure 50k/100k retrieval p95 and recall@5 against exact MVM
- [ ] 5.3 Optimize the lexical/fused path at 100k facts (FTS-first candidate generation, fused-scan pruning) and re-measure lexical p95
- [ ] 5.4 Re-measure Matryoshka 64-D speedup and delta 20-task memory loss with the documented seeds and ablations; fix the delta adapter if the loss remains above 0.10
- [ ] 5.5 Optimize merge 2x10k facts and surprise-gate insert reduction and publish before/after measurements
- [ ] 5.6 For any target still missed, publish the measured result as `partial` and, only if a reviewed decision approves it, add a requirement revision that records old target, new target, measurement and reason; never delete a case or lower a threshold silently
- [ ] 5.7 Update `docs/release-matrix.md` and `reports/evidence/*.json` so every row has an outcome and a provenance record

## 6. Scientific claims and surfaces

- [ ] 6.1 Add a hidden-injection capability probe to `scripts/run_vela_benchmark.py` per `docs/backend-contracts.md`, and report `blocked` with the missing capability when no backend supports it
- [ ] 6.2 Run conditions C and D on the same model, seed, context budget and V1-V12 item set, with at least three repeats where generation is nondeterministic, and report a paired comparison with a confidence interval
- [ ] 6.3 Re-run the Vela conditions with a more instruction-following model and held-out generated rule sets; publish the result even when C does not beat B
- [ ] 6.4 Publish `docs/vela-poc-results.md` v2 with the negative or positive finding and full provenance, keeping the original report intact
- [ ] 6.5 Compile the Tauri shell or record the exact missing toolchain and WebView; make `recamem ui --dry-run` report `shell=tauri` with the binary present when it is built
- [ ] 6.6 Implement or explicitly remove the Memory Explorer, Benchmarks, Training, Architecture Lab and Settings nav entries that currently say "not available yet", with a test that no page shows invented data
- [ ] 6.7 Verify MCP tool list, CLI subcommand set and `recamem serve --rest` loopback-only rules still pass `scripts/offline_gate.py` after all changes

## 7. Release closure

- [ ] 7.1 Resolve the three `partial` rows (`edge-slm-optimization` 1.4, 2.5, 5.2) with implementation, passing tests and evidence, or an approved requirement revision
- [ ] 7.2 Resolve the two `blocked` rows (`edge-slm-optimization` 4.3, 4.4) with the matched-budget comparison or an approved replacement requirement
- [ ] 7.3 Add passing evidence records for `offline`, `real-model`, `grounding`, `concurrency-migration` and `benchmark` tiers, each with commit, command, versions, configuration, seed where applicable, hardware and outcome
- [ ] 7.4 Confirm `scripts/check_release_evidence.py --require-complete` exits 0 with the guard unchanged (diff the guard against its baseline)
- [ ] 7.5 Update `README.md`, `docs/release-evidence.md`, `docs/release-matrix.md`, `docs/testing.md` and `docs/surfaces.md` so no shipped claim exceeds the evidence, with no hand-written test count
- [ ] 7.6 Re-run the full verification set (`pytest tests -q`, `offline_gate.py`, `check_completion_gaps.py`, `validate_openspec.py`, `check_release_evidence.py --require-complete`) and record one final evidence bundle
- [ ] 7.7 Archive this change only after every inventory row is `closed` and every checked task references a passing evidence record
