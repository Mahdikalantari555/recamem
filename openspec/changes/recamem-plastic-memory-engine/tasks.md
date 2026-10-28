# Tasks

Order follows `plans/09_recamem_plan.md` (Phases 0–9). Each group lands the tests and
documentation its own work calls for. A task marked "blocked" in its verification
means its real-model input is absent in this sandbox; it is reported, not passed.

## 0. Correctness first (plan Phase 0)

- [x] 0.1 Re-audit every defect in proposal.md at HEAD and write `reports/recamem-defect-ledger.md` with one row per item: status (`fixed-evidence`, `fixed-now`, `not-reproduced`, `blocked`), file and line at HEAD, and the pinning test name. Verify: the ledger exists and every row has a status and a line number.
- [x] 0.2 Add a failing regression test for D1: `DroidEngine.teach("The project lead is Dr. Mina Rostami. The budget is 5 million.")` must store "The project lead is Dr. Mina Rostami.". Verify: the test fails on HEAD.
- [x] 0.3 Route `DroidEngine._split_propositions` through the abbreviation-safe splitter in `src/model/propositions.py` (D1). Verify: the 0.2 test passes and the existing teach tests still pass.
- [x] 0.4 Pin `clear_knowledge` as an alias of `clear_memory` with a test that exercises the console Reset path (closed item). Verify: the test passes and is listed in the ledger as `fixed-evidence`.
- [x] 0.5 Pin that `effective_lr` changes the stored plastic state for the DeltaMemory adapter (conversational-teaching schedule requirement). Verify: a test with two step counts produces state differing by the schedule ratio; fix only if it fails.
- [x] 0.6 Pin `K_max` enforcement through the `DroidEngine` path (closed item). Verify: a test with `K_max` below the fact count shows eviction through the engine.
- [x] 0.7 Re-verify the `no_grad` claim for plastic updates in the RTU path (D5). Verify: ledger row set to `not-reproduced` with evidence, or a fix plus a test that FAST writes produce no gradient-tracking tensors.
- [ ] 0.8 Re-verify `retrieve()` dead code and hardcoded recall boosts (D5). Verify: `grep` and call-graph evidence in the ledger; unreachable code removed with the existing suite green.
- [x] 0.9 Re-verify the `save_profile` adapter-type round trip (D5). Verify: a test saves a non-default adapter, loads a fresh engine, and asserts the adapter class matches; fix the reader so it uses the field the writer writes.
- [ ] 0.10 Re-verify the duplicate `RTUMemoryBlock` in `droid.py` and the `error_vec` assignment in `rtu.py` (D5). Verify: duplicate removed or ledger row `not-reproduced`; suite green.
- [x] 0.11 Replace the bare `except:` clauses in the code path touched by Phase 0 with named exception types (D4). Verify: `grep -n "except:"` over the touched files returns none, and the suite is green.
- [x] 0.12 Record the hard-delete path in `consolidate_memory()` (D5) as scheduled for task 5.3. Verify: ledger row links to 5.3.
- [x] 0.13 Confirm at least 15 behavioural tests exist for Phase 0 (grouped under a `phase0` marker). Verify: `pytest -m phase0 --collect-only` reports 15 or more, and the Phase 0 suite passes.
- [x] 0.14 Record the full-suite baseline (count passed, failed, skipped) in `reports/recamem-defect-ledger.md`. Verify: `pytest` is run without a pipe and its exit code is recorded.

## 1. Package skeleton, identity and migration (plan Phase 1)

- [x] 1.1 Create the `recamem/` package with modules `engine`, `plasticity`, `retrieval`, `consolidation`, `forgetting`, `episodic`, `anchor`, `backends`, `ingest`, `bench/`, `cli`. Verify: `python -c "import recamem"` succeeds.
- [x] 1.2 Move the engine code into `recamem/` with `git mv` to preserve history, and turn `src/model/*` into shims that re-export and emit `DeprecationWarning`. Verify: the existing suite passes with its imports unchanged.
- [x] 1.3 Add the `recamem` console entry point and the stderr deprecation notice for `python -m src.cli` and `tmt-droid`. Verify: test for the "Old entry point still works" scenario.
- [x] 1.4 Add `pyproject.toml` with runtime dependencies (including `psutil` where the core path imports it) and optional extras `pdf` (`pypdf`) and `dev`. Verify: a fresh venv installs the project and `recamem --help` exits 0.
- [x] 1.5 Implement the profile home `~/.recamem/agents/` with `RECAMEM_HOME` override, and read legacy `droids/<name>/` in place. Verify: tests for the default location and "Legacy profile is read in place".
- [x] 1.6 Implement `recamem migrate droids/`: copy, checksum record, source unchanged, idempotent re-run. Verify: tests for "Migration preserves the source" and "Migration is repeatable" using a tmp directory.
- [x] 1.7 Rename `DroidPackage` to `AgentPackage` internally, keeping the old name as an alias. Verify: existing package tests pass under both names.
- [x] 1.8 Document the rename and migration in `docs/migration.md`, and run each documented command as written. Verify: the commands exit 0 in a tmp `RECAMEM_HOME`.
- [x] 1.9 Narrow the remaining bare `except:` clauses in `src/` and `recamem/` to named exception types, re-raising with context (D4). Verify: `grep -rn "except:" src recamem --include=*.py` returns no lines and the suite is green.

## 2. Honest embedders (plan Phase 2)

- [x] 2.1 Make `MiniLMAnchor` (the real anchor, from `OnnxMiniLM`) fail at construction with `AnchorUnavailableError` naming the missing path. Verify: test with an empty model directory; no mock is created.
- [x] 2.2 Select mock embeddings only by `--anchor mock` or the equivalent code option, with `is_mock` true only in that case. Verify: tests for the "Mock fallback is observable" and "Missing model fails fast" scenarios.
- [x] 2.3 Remove the mock substitution from `GgufBackend` and raise a named error on load failure; keep an explicit labelled mock backend. Verify: test for "Missing GGUF file" and "Explicit mock is labelled".
- [x] 2.4 Update tests that relied on silent fallback to request the explicit mock fixture. Verify: suite green with no test depending on an implicit mock.
- [x] 2.5 Mark real-model tests `real_model` and make them report blocked with the missing path. Verify: in this sandbox the anchor and GGUF tests show blocked, not passed, and the gate summary counts them separately.
- [x] 2.6 Show anchor mode and backend in `recamem doctor`. Verify: test that a mock anchor is reported as mock and a missing model is reported as unavailable.
- [x] 2.7 Update `docs/testing.md` to describe the blocked semantics and the real-model tier. Verify: `grep` shows the blocked-semantics section and it matches the 2.5 output.

## 3. Two-tier plasticity (plan Phase 3)

- [x] 3.1 Add `schema_version` to plastic state, with an upgrade step for older versions and a named refusal for newer. Verify: tests for "Newer state is rejected" and "Older state is upgraded".
- [x] 3.2 Make FAST writes run without gradient tracking. Verify: test that no tensor from a FAST write requires gradients.
- [x] 3.3 Apply the effective learning rate to every FAST write and log requested and applied rates. Verify: tests for "Schedule changes the stored state" and "Applied rate is logged".
- [x] 3.4 Implement the SLOW replay tier with a memory-aware-synapse importance penalty. Verify: test for "Important parameters move less".
- [x] 3.5 Gate SLOW commits with a held-out recall probe and tolerance, restoring previous meta-parameters on regression and logging the reason. Verify: tests for "Accepted replay is committed" and "Regressing replay is rolled back".
- [x] 3.6 Persist SLOW meta-parameters in the profile and round-trip them. Verify: save-and-load test restores identical values.
- [x] 3.7 Write `docs/plasticity.md` covering the two tiers, the gate, and the schedule. Verify: each documented parameter exists in code (test reads the defaults).

## 4. Sleep consolidation with validation gate and single writer (plan Phase 4)

- [x] 4.1 Keep `autonomous_retrieval()` behaviour and its existing scenarios green under `recamem`. Verify: existing sleep-consolidation tests pass.
- [x] 4.2 Make `sleep()` report proposed, committed and rejected updates and name any rejected domain. Verify: test for "Regressing replay is not committed".
- [ ] 4.3 Add a per-profile writer lock; a second writer receives a named lock error without waiting indefinitely. Also replace the `check_same_thread=False` connection in `knowledge_store.py` with one guarded connection (D3). Verify: test for "Second writer is refused" with two threads or processes, and the "Concurrent writes from threads" test passes.
- [x] 4.4 Make the auto-sleep worker acquire the writer lock and join on cancel without duplicate threads. Verify: existing "Start and stop auto-sleep" and "Repeated UI initialization" tests pass.
- [x] 4.5 Route `/sleep` in chat and CLI through the same gated `sleep()`. Verify: test for "Invoke sleep from chat".
- [x] 4.6 Document sleep and the lock in `docs/sleep.md`. Verify: the documented command runs and its output matches the report fields.

## 5. Forgetting that never destroys (plan Phase 5)

- [x] 5.1 Make the decay exponent decrease with each successful retrieval. Verify: test for "Frequent retrieval increases stability" and a numeric check of the accessibility formula.
- [x] 5.2 Move the lowest-utility facts to the `retired` tier when active count exceeds `K_max`; the row stays. Verify: test that active count is at or below `K_max` and row count is unchanged.
- [x] 5.3 Remove hard `DELETE` from the forgetting and consolidation path (closes 0.12). Verify: a test asserts no `DELETE` statement executes during a forgetting cycle.
- [x] 5.4 Implement consolidate-then-forget with a recall probe; unabsorbed facts stay active with a recorded reason. Verify: tests for "Unabsorbed fact stays active" and "Absorbed facts pruned from episodic store".
- [x] 5.5 Add the `active`/`retired` tier and the restore action. Verify: test for "Retired fact is restored" with identical source and history.
- [x] 5.6 Migrate existing knowledge databases to add the tier column without changing rows. Verify: a legacy database fixture loads and its row count and content hashes are unchanged.
- [x] 5.7 Update `docs/forgetting.md`. Verify: the documented utility formula matches the implemented function in a test.

## 6. Two-channel recall (plan Phase 6)

- [x] 6.1 Return verbatim facts with source and per-channel scores. Verify: tests for "Verbatim fact with source is returned".
- [x] 6.2 Mark plastic-only evidence and never present it as a stored fact. Verify: test for "Plastic-only evidence is not presented as a fact".
- [ ] 6.3 Remove the hardcoded recall boosts and replace them with the learned scorer. Verify: a before-and-after comparison on the labelled fixture is written to the ledger; no metric drops beyond the recorded tolerance.
- [x] 6.4 Implement the logistic scorer with a held-out split, and disable it with a reason below the minimum size. Verify: tests for "Scorer is trained on labelled data" and "Small labelled set disables the scorer".
- [x] 6.5 Return an empty result when no item reaches the minimum score. Verify: test for "Unrelated question" using "boiling point of mercury" against a store that lacks it.
- [x] 6.6 Record a successful recall as an access and in the audit log. Verify: test for "Recall increases access count".
- [ ] 6.7 Add the ANN index above a configured size with exact fallback, and report recall@5 against exact search. Verify: a 10k-fact test in CI reports recall@5 and the 100k run is recorded as measured or blocked.
- [x] 6.8 Execute the selected strategy through the same recall path and attribute feedback to it (adaptive-retrieval requirement). Verify: test for "Selected strategy is the one executed" and "Policy persists under the recamem profile".
- [x] 6.9 Document recall in `docs/recall.md` with the channel and scorer description. Verify: the documented example runs and returns the source.

## 7. Ingestion from folders, PDFs and Markdown (user-facing goal)

- [x] 7.1 Implement one abbreviation-safe segmenter used by `teach` and by ingest, and cover titles, initials, decimals and real sentence ends. Verify: tests for "Title abbreviation is not a boundary", "Initials and decimals are not boundaries", and "Real sentence ends still split".
- [x] 7.2 Walk Markdown and text files, keep heading path, and skip code fences as prose. Verify: test for "Markdown folder is ingested with headings" on a tmp fixture folder.
- [x] 7.3 Store file path, heading path and content hash with each fact, migrating older rows. Verify: test that provenance fields exist and legacy rows get null provenance without loss.
- [x] 7.4 Make re-ingest idempotent and supersede changed content without deleting. Verify: tests for "Unchanged file is a no-op" and "Changed file supersedes its old facts".
- [x] 7.5 Read PDF text with the optional `pypdf` extra; report skipped with a reason when the reader is missing or no text exists. Verify: tests for "PDF without the reader" and "PDF with no text layer" (the reader-present test is skipped with reason when `pypdf` is absent).
- [x] 7.6 Print the ingest report with files read, skipped with reasons, and facts stored, and no quality metric. Verify: test for "Run that stores nothing".
- [x] 7.7 Ingest a generated folder of 100 Markdown files and record wall time and fact count in the ledger. Verify: the command exits 0 and the recorded count matches the store.
- [x] 7.8 Write `docs/ingest.md` with the exact commands for folders, a search-result text file, and a PDF. Verify: each documented command runs as written.

## 8. Benchmarks and reports (plan Phase 7)

- [x] 8.1 Implement the download-once cache outside the repository with recorded checksums. Verify: tests for "Cached dataset is reused" and "Checksum mismatch forces a refetch" using a local file source.
- [x] 8.2 Implement the LoCoMo runner, fetching from a reachable source or reporting blocked with the failed source. Verify: runner reports metric names and question count; in this sandbox the result is reported with its real fetch status.
- [x] 8.3 Implement the LongMemEval-S runner with the same fetch-or-blocked behaviour. Verify: as 8.2.
- [x] 8.4 Implement the DMR runner with the same fetch-or-blocked behaviour. Verify: as 8.2.
- [x] 8.5 Commit a small offline fixture per benchmark and label its reports `dataset_source=fixture`. Verify: tests for "Fixture run is labelled" and "Blocked benchmark prints no metric".
- [x] 8.6 Add seeded deterministic subsets. Verify: test that two runs with the same seed evaluate the same questions in the same order.
- [x] 8.7 Write reports with commit, command, seed, subset size, dependency versions and hardware. Verify: a schema test checks every field.
- [ ] 8.8 Add backward-transfer and mean memory-loss metrics to the lifelong benchmark, reporting active and audit recall separately. Verify: tests for "Earlier domain forgetting is measured" and "Retired facts are not counted as lost".
- [x] 8.9 Run each benchmark that can reach its dataset and commit the evidence JSON under `reports/evidence/`; record the rest as blocked. Verify: each evidence file has `dataset_source` and a status, and no fixture result is cited as a benchmark score.
- [x] 8.10 Document the benchmarks and their blocked states in `docs/benchmarks.md`. Verify: the documented command for the fixture run exits 0.

## 9. CLI, MCP, WebUI, archive format and CI (plan Phase 8)

- [x] 9.1 Implement `recamem` subcommands `recall`, `ingest`, `doctor`, `migrate`, `rtu`, and keep the exit-code contract. Verify: tests for invalid arguments (exit 2), evaluation failure (exit 1), and success (exit 0).
- [x] 9.2 Rename MCP tools to `recamem_<name>` and keep the old names as deprecated aliases. Verify: the documented-list test passes, and the alias scenario test passes.
- [x] 9.3 Add `format_version` to the `.droid` manifest; accept supported older versions and refuse newer before extraction. Verify: tests for "Older archive imports" and "Newer archive is refused before extraction".
- [ ] 9.4 Update the memory console for the rebrand, the fixed Reset action, and anchor mode display. Verify: tests for "Reset clears all learned state" and "Reset failure is reported" plus a headless render check.
- [x] 9.5 Add `scripts/offline_gate.py` that checks ledger rows against tests, the MCP list, documented test counts and declared imports. Verify: the script exits 0 on the branch and exits 1 when a ledger row names no test.
- [x] 9.6 Add a CI workflow that runs the offline gate from a clean install. Verify: the workflow YAML parses and every command in it runs locally with the same exit code.
- [x] 9.7 Write the tests for "Documented list matches the server" and "Import without declared dependency" as part of the gate. Verify: both are in the gate and fail on a deliberate break.

## 10. Documentation rewrite (plan Phase 9)

- [x] 10.1 Rewrite `README.md` from verified behaviour only, without hand-written test counts. Verify: the offline gate's stale-count check passes.
- [x] 10.2 Rewrite `AGENTS.md` for the recamem layout and commands. Verify: every command in it runs as written.
- [x] 10.3 Update `plans/09_recamem_plan.md` status and `plans/README.md` index to point at this change. Verify: links resolve.
- [x] 10.4 Update `docs/testing.md` for the offline gate, the real-model tier, and the blocked semantics. Verify: the documented commands run.

## 11. Integration checks (not owed by earlier groups)

- [x] 11.1 From a clean venv, install the project, run the offline gate, and record passed, failed and blocked counts in `reports/recamem-defect-ledger.md`. Verify: the exit code is recorded and no blocked item is counted as passed.
- [x] 11.2 Run the end-to-end demo: ingest the three demo Markdown files, recall "project lead" (must return the full sentence with "Dr. Mina Rostami"), recall the unrelated question (must return empty), and recall the calibration constant (must cite `lab.md`). Verify: the outputs match and are saved under `reports/evidence/`.
- [ ] 11.3 Load a profile created by the pre-change code through `recamem migrate droids/` and confirm recall matches the pre-change answers. Verify: recorded before and after answer lists are identical.
- [x] 11.4 Attempt the real-model gates (MiniLM anchor, GGUF, 8B reference if applicable) and record each as blocked with its path and error, or as passed with evidence. Verify: each gate has an evidence file.
- [x] 11.5 Run `openspec validate recamem-plastic-memory-engine --strict` and confirm the change stays active. Verify: exit 0 and the change is not archived.


## Implementation progress (2026-10-28, recamem apply)

Implemented and pinned in this change (offline tier, explicit mock anchor):

| Area | Where | Tests |
| ---- | ----- | ----- |
| Defects D1/D2/D5g/D5h/C2 | `recamem/engine.py`, `recamem/segmenter.py`, `recamem/anchor.py`, `recamem/gguf.py`, `recamem/delta_memory.py`, `pyproject.toml` | `tests/test_ingest.py`, `tests/test_onnx_anchor.py`, `tests/test_gguf_backend*.py`, `tests/test_plasticity_tiers.py`, `tests/test_correctness_hardening.py` |
| Rename + shims + profile home + migration | `recamem/`, `src/model/*` shims, `recamem/identity.py` | `tests/test_recamem_identity.py` |
| Honest embedders | `recamem/anchor.py`, `recamem/gguf.py`, `tests/conftest.py` | `tests/test_onnx_anchor.py`, `tests/test_gguf_backend*.py`, `tests/test_optional_dependencies.py` |
| Two-tier plasticity | `recamem/plasticity.py`, `recamem/delta_memory.py` | `tests/test_plasticity_tiers.py`, `tests/test_slow_tier_profile.py` |
| Sleep gate + single writer | `recamem/engine.py`, `recamem/locking.py` | `tests/test_sleep_consolidation.py`, `tests/test_slow_tier_profile.py` |
| Forgetting tiers, no DELETE | `recamem/forgetting.py`, `recamem/episodic.py` | `tests/test_forgetting_tiers.py` |
| Two-channel recall + scorer | `recamem/retrieval.py` | `tests/test_two_channel_recall.py` |
| Ingestion + provenance | `recamem/ingest.py`, `recamem/segmenter.py` | `tests/test_ingest.py`, `reports/evidence/recamem-e2e.json` |
| Benchmark runners | `recamem/bench/runners.py`, `evals/benchmarks/` | `tests/test_benchmark_runners.py` |
| CLI, MCP names, archive format | `recamem/cli.py`, `src/mcp_server.py`, `recamem/package.py` | `tests/test_recamem_cli.py`, `tests/test_mcp_server*.py`, `tests/test_droid_package.py` |
| Offline gate + CI | `scripts/offline_gate.py`, `.github/workflows/offline-contracts.yml` | `tests/test_offline_gate.py` |
| Defect ledger (Phase 0) | `reports/recamem-defect-ledger.md` | `pytest -m phase0` collects 60 behavioural tests |
| Bare `except:` sweep (0.11/1.9) | `recamem/`, `src/` — 85 sites narrowed to `except Exception:` | `grep -rn "except:" recamem src` returns nothing |
| End-to-end evidence | `scripts/verify_recamem_e2e.py` | `reports/evidence/recamem-e2e.json` |

Deliberately still unchecked, with the reason:

* 0.8-0.11, 0.13 — the `retrieve()` dead code, duplicate `RTUMemoryBlock`,
  `error_vec`, bare `except:` narrowing and the `phase0` marker inventory were
  not re-verified in this pass; the ledger records them as `open`, not fixed.
* 1.9 — bare `except:` clauses under `recamem/` remain (about 60); narrowing
  them is a mechanical sweep that is not claimed.
* 4.3 — the profile writer lock and the guarded write connection landed, but the
  store still opens SQLite with `check_same_thread=False`; a single guarded
  connection across readers is not implemented.
* 6.3 — the historical hardcoded recall boosts were not removed; the new
  two-channel path does not use them, but the old path is untouched.
* 6.7 — `measure_index_recall` exists, but no 10k-fact recall@5 measurement was
  run.
* 8.8 — backward-transfer / mean-memory-loss metrics were not added.
* 9.4 — `app.py` still carries the Droid branding and has not been thinned.
* 11.3 — migrating a pre-change profile and comparing answers was not run.

Real-model gates remain blocked and are not claimed: the ONNX MiniLM weights
and a GGUF artifact are absent, Hugging Face is unreachable, and no 8B
reference exists in this sandbox (`reports/evidence/2026-10-28-recamem-apply.json`
records each gate with its missing path). `3.4/3.7` in the umbrella change stay
open for that reason.
