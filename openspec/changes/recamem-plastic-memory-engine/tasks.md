# Tasks

Order follows `plans/09_recamem_plan.md` (Phases 0–9). Each group lands the tests and
documentation its own work calls for. A task marked "blocked" in its verification
means its real-model input is absent in this sandbox; it is reported, not passed.

## 0. Correctness first (plan Phase 0)

- [ ] 0.1 Re-audit every defect in proposal.md at HEAD and write `reports/recamem-defect-ledger.md` with one row per item: status (`fixed-evidence`, `fixed-now`, `not-reproduced`, `blocked`), file and line at HEAD, and the pinning test name. Verify: the ledger exists and every row has a status and a line number.
- [ ] 0.2 Add a failing regression test for D1: `DroidEngine.teach("The project lead is Dr. Mina Rostami. The budget is 5 million.")` must store "The project lead is Dr. Mina Rostami.". Verify: the test fails on HEAD.
- [ ] 0.3 Route `DroidEngine._split_propositions` through the abbreviation-safe splitter in `src/model/propositions.py` (D1). Verify: the 0.2 test passes and the existing teach tests still pass.
- [ ] 0.4 Pin `clear_knowledge` as an alias of `clear_memory` with a test that exercises the console Reset path (closed item). Verify: the test passes and is listed in the ledger as `fixed-evidence`.
- [ ] 0.5 Pin that `effective_lr` changes the stored plastic state for the DeltaMemory adapter (conversational-teaching schedule requirement). Verify: a test with two step counts produces state differing by the schedule ratio; fix only if it fails.
- [ ] 0.6 Pin `K_max` enforcement through the `DroidEngine` path (closed item). Verify: a test with `K_max` below the fact count shows eviction through the engine.
- [ ] 0.7 Re-verify the `no_grad` claim for plastic updates in the RTU path (D5). Verify: ledger row set to `not-reproduced` with evidence, or a fix plus a test that FAST writes produce no gradient-tracking tensors.
- [ ] 0.8 Re-verify `retrieve()` dead code and hardcoded recall boosts (D5). Verify: `grep` and call-graph evidence in the ledger; unreachable code removed with the existing suite green.
- [ ] 0.9 Re-verify the `save_profile` adapter-type round trip (D5). Verify: a test saves a non-default adapter, loads a fresh engine, and asserts the adapter class matches; fix the reader so it uses the field the writer writes.
- [ ] 0.10 Re-verify the duplicate `RTUMemoryBlock` in `droid.py` and the `error_vec` assignment in `rtu.py` (D5). Verify: duplicate removed or ledger row `not-reproduced`; suite green.
- [ ] 0.11 Replace the bare `except:` clauses in the code path touched by Phase 0 with named exception types (D4). Verify: `grep -n "except:"` over the touched files returns none, and the suite is green.
- [ ] 0.12 Record the hard-delete path in `consolidate_memory()` (D5) as scheduled for task 5.3. Verify: ledger row links to 5.3.
- [ ] 0.13 Confirm at least 15 behavioural tests exist for Phase 0 (grouped under a `phase0` marker). Verify: `pytest -m phase0 --collect-only` reports 15 or more, and the Phase 0 suite passes.
- [ ] 0.14 Record the full-suite baseline (count passed, failed, skipped) in `reports/recamem-defect-ledger.md`. Verify: `pytest` is run without a pipe and its exit code is recorded.

## 1. Package skeleton, identity and migration (plan Phase 1)

- [ ] 1.1 Create the `recamem/` package with modules `engine`, `plasticity`, `retrieval`, `consolidation`, `forgetting`, `episodic`, `anchor`, `backends`, `ingest`, `bench/`, `cli`. Verify: `python -c "import recamem"` succeeds.
- [ ] 1.2 Move the engine code into `recamem/` with `git mv` to preserve history, and turn `src/model/*` into shims that re-export and emit `DeprecationWarning`. Verify: the existing suite passes with its imports unchanged.
- [ ] 1.3 Add the `recamem` console entry point and the stderr deprecation notice for `python -m src.cli` and `tmt-droid`. Verify: test for the "Old entry point still works" scenario.
- [ ] 1.4 Add `pyproject.toml` with runtime dependencies (including `psutil` where the core path imports it) and optional extras `pdf` (`pypdf`) and `dev`. Verify: a fresh venv installs the project and `recamem --help` exits 0.
- [ ] 1.5 Implement the profile home `~/.recamem/agents/` with `RECAMEM_HOME` override, and read legacy `droids/<name>/` in place. Verify: tests for the default location and "Legacy profile is read in place".
- [ ] 1.6 Implement `recamem migrate droids/`: copy, checksum record, source unchanged, idempotent re-run. Verify: tests for "Migration preserves the source" and "Migration is repeatable" using a tmp directory.
- [ ] 1.7 Rename `DroidPackage` to `AgentPackage` internally, keeping the old name as an alias. Verify: existing package tests pass under both names.
- [ ] 1.8 Document the rename and migration in `docs/migration.md`, and run each documented command as written. Verify: the commands exit 0 in a tmp `RECAMEM_HOME`.
- [ ] 1.9 Narrow the remaining bare `except:` clauses in `src/` and `recamem/` to named exception types, re-raising with context (D4). Verify: `grep -rn "except:" src recamem --include=*.py` returns no lines and the suite is green.

## 2. Honest embedders (plan Phase 2)

- [ ] 2.1 Make `MiniLMAnchor` (the real anchor, from `OnnxMiniLM`) fail at construction with `AnchorUnavailableError` naming the missing path. Verify: test with an empty model directory; no mock is created.
- [ ] 2.2 Select mock embeddings only by `--anchor mock` or the equivalent code option, with `is_mock` true only in that case. Verify: tests for the "Mock fallback is observable" and "Missing model fails fast" scenarios.
- [ ] 2.3 Remove the mock substitution from `GgufBackend` and raise a named error on load failure; keep an explicit labelled mock backend. Verify: test for "Missing GGUF file" and "Explicit mock is labelled".
- [ ] 2.4 Update tests that relied on silent fallback to request the explicit mock fixture. Verify: suite green with no test depending on an implicit mock.
- [ ] 2.5 Mark real-model tests `real_model` and make them report blocked with the missing path. Verify: in this sandbox the anchor and GGUF tests show blocked, not passed, and the gate summary counts them separately.
- [ ] 2.6 Show anchor mode and backend in `recamem doctor`. Verify: test that a mock anchor is reported as mock and a missing model is reported as unavailable.
- [ ] 2.7 Update `docs/testing.md` to describe the blocked semantics and the real-model tier. Verify: `grep` shows the blocked-semantics section and it matches the 2.5 output.

## 3. Two-tier plasticity (plan Phase 3)

- [ ] 3.1 Add `schema_version` to plastic state, with an upgrade step for older versions and a named refusal for newer. Verify: tests for "Newer state is rejected" and "Older state is upgraded".
- [ ] 3.2 Make FAST writes run without gradient tracking. Verify: test that no tensor from a FAST write requires gradients.
- [ ] 3.3 Apply the effective learning rate to every FAST write and log requested and applied rates. Verify: tests for "Schedule changes the stored state" and "Applied rate is logged".
- [ ] 3.4 Implement the SLOW replay tier with a memory-aware-synapse importance penalty. Verify: test for "Important parameters move less".
- [ ] 3.5 Gate SLOW commits with a held-out recall probe and tolerance, restoring previous meta-parameters on regression and logging the reason. Verify: tests for "Accepted replay is committed" and "Regressing replay is rolled back".
- [ ] 3.6 Persist SLOW meta-parameters in the profile and round-trip them. Verify: save-and-load test restores identical values.
- [ ] 3.7 Write `docs/plasticity.md` covering the two tiers, the gate, and the schedule. Verify: each documented parameter exists in code (test reads the defaults).

## 4. Sleep consolidation with validation gate and single writer (plan Phase 4)

- [ ] 4.1 Keep `autonomous_retrieval()` behaviour and its existing scenarios green under `recamem`. Verify: existing sleep-consolidation tests pass.
- [ ] 4.2 Make `sleep()` report proposed, committed and rejected updates and name any rejected domain. Verify: test for "Regressing replay is not committed".
- [ ] 4.3 Add a per-profile writer lock; a second writer receives a named lock error without waiting indefinitely. Also replace the `check_same_thread=False` connection in `knowledge_store.py` with one guarded connection (D3). Verify: test for "Second writer is refused" with two threads or processes, and the "Concurrent writes from threads" test passes.
- [ ] 4.4 Make the auto-sleep worker acquire the writer lock and join on cancel without duplicate threads. Verify: existing "Start and stop auto-sleep" and "Repeated UI initialization" tests pass.
- [ ] 4.5 Route `/sleep` in chat and CLI through the same gated `sleep()`. Verify: test for "Invoke sleep from chat".
- [ ] 4.6 Document sleep and the lock in `docs/sleep.md`. Verify: the documented command runs and its output matches the report fields.

## 5. Forgetting that never destroys (plan Phase 5)

- [ ] 5.1 Make the decay exponent decrease with each successful retrieval. Verify: test for "Frequent retrieval increases stability" and a numeric check of the accessibility formula.
- [ ] 5.2 Move the lowest-utility facts to the `retired` tier when active count exceeds `K_max`; the row stays. Verify: test that active count is at or below `K_max` and row count is unchanged.
- [ ] 5.3 Remove hard `DELETE` from the forgetting and consolidation path (closes 0.12). Verify: a test asserts no `DELETE` statement executes during a forgetting cycle.
- [ ] 5.4 Implement consolidate-then-forget with a recall probe; unabsorbed facts stay active with a recorded reason. Verify: tests for "Unabsorbed fact stays active" and "Absorbed facts pruned from episodic store".
- [ ] 5.5 Add the `active`/`retired` tier and the restore action. Verify: test for "Retired fact is restored" with identical source and history.
- [ ] 5.6 Migrate existing knowledge databases to add the tier column without changing rows. Verify: a legacy database fixture loads and its row count and content hashes are unchanged.
- [ ] 5.7 Update `docs/forgetting.md`. Verify: the documented utility formula matches the implemented function in a test.

## 6. Two-channel recall (plan Phase 6)

- [ ] 6.1 Return verbatim facts with source and per-channel scores. Verify: tests for "Verbatim fact with source is returned".
- [ ] 6.2 Mark plastic-only evidence and never present it as a stored fact. Verify: test for "Plastic-only evidence is not presented as a fact".
- [ ] 6.3 Remove the hardcoded recall boosts and replace them with the learned scorer. Verify: a before-and-after comparison on the labelled fixture is written to the ledger; no metric drops beyond the recorded tolerance.
- [ ] 6.4 Implement the logistic scorer with a held-out split, and disable it with a reason below the minimum size. Verify: tests for "Scorer is trained on labelled data" and "Small labelled set disables the scorer".
- [ ] 6.5 Return an empty result when no item reaches the minimum score. Verify: test for "Unrelated question" using "boiling point of mercury" against a store that lacks it.
- [ ] 6.6 Record a successful recall as an access and in the audit log. Verify: test for "Recall increases access count".
- [ ] 6.7 Add the ANN index above a configured size with exact fallback, and report recall@5 against exact search. Verify: a 10k-fact test in CI reports recall@5 and the 100k run is recorded as measured or blocked.
- [ ] 6.8 Execute the selected strategy through the same recall path and attribute feedback to it (adaptive-retrieval requirement). Verify: test for "Selected strategy is the one executed" and "Policy persists under the recamem profile".
- [ ] 6.9 Document recall in `docs/recall.md` with the channel and scorer description. Verify: the documented example runs and returns the source.

## 7. Ingestion from folders, PDFs and Markdown (user-facing goal)

- [ ] 7.1 Implement one abbreviation-safe segmenter used by `teach` and by ingest, and cover titles, initials, decimals and real sentence ends. Verify: tests for "Title abbreviation is not a boundary", "Initials and decimals are not boundaries", and "Real sentence ends still split".
- [ ] 7.2 Walk Markdown and text files, keep heading path, and skip code fences as prose. Verify: test for "Markdown folder is ingested with headings" on a tmp fixture folder.
- [ ] 7.3 Store file path, heading path and content hash with each fact, migrating older rows. Verify: test that provenance fields exist and legacy rows get null provenance without loss.
- [ ] 7.4 Make re-ingest idempotent and supersede changed content without deleting. Verify: tests for "Unchanged file is a no-op" and "Changed file supersedes its old facts".
- [ ] 7.5 Read PDF text with the optional `pypdf` extra; report skipped with a reason when the reader is missing or no text exists. Verify: tests for "PDF without the reader" and "PDF with no text layer" (the reader-present test is skipped with reason when `pypdf` is absent).
- [ ] 7.6 Print the ingest report with files read, skipped with reasons, and facts stored, and no quality metric. Verify: test for "Run that stores nothing".
- [ ] 7.7 Ingest a generated folder of 100 Markdown files and record wall time and fact count in the ledger. Verify: the command exits 0 and the recorded count matches the store.
- [ ] 7.8 Write `docs/ingest.md` with the exact commands for folders, a search-result text file, and a PDF. Verify: each documented command runs as written.

## 8. Benchmarks and reports (plan Phase 7)

- [ ] 8.1 Implement the download-once cache outside the repository with recorded checksums. Verify: tests for "Cached dataset is reused" and "Checksum mismatch forces a refetch" using a local file source.
- [ ] 8.2 Implement the LoCoMo runner, fetching from a reachable source or reporting blocked with the failed source. Verify: runner reports metric names and question count; in this sandbox the result is reported with its real fetch status.
- [ ] 8.3 Implement the LongMemEval-S runner with the same fetch-or-blocked behaviour. Verify: as 8.2.
- [ ] 8.4 Implement the DMR runner with the same fetch-or-blocked behaviour. Verify: as 8.2.
- [ ] 8.5 Commit a small offline fixture per benchmark and label its reports `dataset_source=fixture`. Verify: tests for "Fixture run is labelled" and "Blocked benchmark prints no metric".
- [ ] 8.6 Add seeded deterministic subsets. Verify: test that two runs with the same seed evaluate the same questions in the same order.
- [ ] 8.7 Write reports with commit, command, seed, subset size, dependency versions and hardware. Verify: a schema test checks every field.
- [ ] 8.8 Add backward-transfer and mean memory-loss metrics to the lifelong benchmark, reporting active and audit recall separately. Verify: tests for "Earlier domain forgetting is measured" and "Retired facts are not counted as lost".
- [ ] 8.9 Run each benchmark that can reach its dataset and commit the evidence JSON under `reports/evidence/`; record the rest as blocked. Verify: each evidence file has `dataset_source` and a status, and no fixture result is cited as a benchmark score.
- [ ] 8.10 Document the benchmarks and their blocked states in `docs/benchmarks.md`. Verify: the documented command for the fixture run exits 0.

## 9. CLI, MCP, WebUI, archive format and CI (plan Phase 8)

- [ ] 9.1 Implement `recamem` subcommands `recall`, `ingest`, `doctor`, `migrate`, `rtu`, and keep the exit-code contract. Verify: tests for invalid arguments (exit 2), evaluation failure (exit 1), and success (exit 0).
- [ ] 9.2 Rename MCP tools to `recamem_<name>` and keep the old names as deprecated aliases. Verify: the documented-list test passes, and the alias scenario test passes.
- [ ] 9.3 Add `format_version` to the `.droid` manifest; accept supported older versions and refuse newer before extraction. Verify: tests for "Older archive imports" and "Newer archive is refused before extraction".
- [ ] 9.4 Update the memory console for the rebrand, the fixed Reset action, and anchor mode display. Verify: tests for "Reset clears all learned state" and "Reset failure is reported" plus a headless render check.
- [ ] 9.5 Add `scripts/offline_gate.py` that checks ledger rows against tests, the MCP list, documented test counts and declared imports. Verify: the script exits 0 on the branch and exits 1 when a ledger row names no test.
- [ ] 9.6 Add a CI workflow that runs the offline gate from a clean install. Verify: the workflow YAML parses and every command in it runs locally with the same exit code.
- [ ] 9.7 Write the tests for "Documented list matches the server" and "Import without declared dependency" as part of the gate. Verify: both are in the gate and fail on a deliberate break.

## 10. Documentation rewrite (plan Phase 9)

- [ ] 10.1 Rewrite `README.md` from verified behaviour only, without hand-written test counts. Verify: the offline gate's stale-count check passes.
- [ ] 10.2 Rewrite `AGENTS.md` for the recamem layout and commands. Verify: every command in it runs as written.
- [ ] 10.3 Update `plans/09_recamem_plan.md` status and `plans/README.md` index to point at this change. Verify: links resolve.
- [ ] 10.4 Update `docs/testing.md` for the offline gate, the real-model tier, and the blocked semantics. Verify: the documented commands run.

## 11. Integration checks (not owed by earlier groups)

- [ ] 11.1 From a clean venv, install the project, run the offline gate, and record passed, failed and blocked counts in `reports/recamem-defect-ledger.md`. Verify: the exit code is recorded and no blocked item is counted as passed.
- [ ] 11.2 Run the end-to-end demo: ingest the three demo Markdown files, recall "project lead" (must return the full sentence with "Dr. Mina Rostami"), recall the unrelated question (must return empty), and recall the calibration constant (must cite `lab.md`). Verify: the outputs match and are saved under `reports/evidence/`.
- [ ] 11.3 Load a profile created by the pre-change code through `recamem migrate droids/` and confirm recall matches the pre-change answers. Verify: recorded before and after answer lists are identical.
- [ ] 11.4 Attempt the real-model gates (MiniLM anchor, GGUF, 8B reference if applicable) and record each as blocked with its path and error, or as passed with evidence. Verify: each gate has an evidence file.
- [ ] 11.5 Run `openspec validate recamem-plastic-memory-engine --strict` and confirm the change stays active. Verify: exit 0 and the change is not archived.
