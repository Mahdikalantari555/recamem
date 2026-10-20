# Design

## Context

The surfaces this change completes already have backends; what is missing is wiring,
reporting, and one wrong argument. Verified state:

| Seam | Location | State |
|---|---|---|
| `eval` | `src/cli.py:371-374`, parser at `:461-462` | Prints a notice, returns 0 |
| MCP tools | `src/mcp_server.py:56-129` | Six tools including `generate_with_memory` |
| MCP spec | `openspec/specs/mcp-server/spec.md:23-35` | Lists five |
| Supersede | `app.py:538-545` | `supersede(fact_id_input, 0)` |
| Bare excepts | `app.py:109`, `:196`, `:675` | Swallow and print `N/A` / pass |
| Snapshots/merge/stream | `src/model/memory_lifecycle.py` | No UI entry point |
| Confidence | `src/model/confidence.py:112` | `confidence_tier()` exists, unused in UI |
| Decision | `src/model/droid.py` `decide_path()` | No UI entry point |
| Packages | `src/model/droid_package.py:27,100,169` | CLI only |
| GGUF download | `src/model/gguf_backend.py:137` | CLI only (`download-model`) |

## Goals / Non-Goals

**Goals**

- `eval` performs a real evaluation and prints machine-readable output with honest exit
  codes.
- The documented MCP tool list equals the exposed list, and a test keeps them equal.
- Supersede records a real successor.
- No bare `except:` remains in `app.py`; every caught failure is named and reported.
- Every memory-lifecycle and portability action named in this change is reachable from
  the console.
- Destructive actions are confirmation-gated.

**Non-Goals**

- No new engine, store, or package API beyond the small helpers named below.
- No new UI framework, no page routing, no client-side state.
- No automated Streamlit tests; panels are validated manually and recorded.
- No packaged distribution (no `pyproject.toml`, no console entry point) — plan 09
  Phase 8 stays out of scope.

## Decisions

### Decision 1 — `eval` measures retrieval, and says on what

`cmd_eval` gains arguments:

```
python -m src.cli eval --name <droid> [--queries <jsonl>] [--top-k 5] [--sample 50]
```

Two modes, and the output states which one ran:

- **Labeled** (`--queries`): each line is `{"query": "...", "fact_ids": [..]}`. The
  command runs `engine.recall(query, top_k=k)` and computes precision@k, recall@k, and
  MRR against the labeled ids.
- **Self-recall** (no `--queries`): samples `--sample` stored facts, queries with the
  fact text, and measures whether the fact itself appears in the top-k. This is a
  sanity check of the index, not a quality benchmark, and the output says so.

Output (stdout, no ANSI, stable order):

```
mode=self_recall
profile=droid-rs
facts=128
samples=50
top_k=5
precision_at_k=0.840
recall_at_k=0.840
mrr=0.912
```

Errors (missing profile, unreadable query file, empty store) go to stderr prefixed
`error:`, and the exit code is `2` for invalid input, `1` for an evaluation failure, `0`
for success. The current stub's "success with nothing measured" is the specific behavior
being removed.

### Decision 2 — Tool/spec parity is a test over both spec locations

`tests/test_mcp_server.py` asserts `set(handle_tools_list()["tools"] names) ==
set(documented_names)`, where `documented_names` is parsed from
`openspec/specs/mcp-server/spec.md` plus every
`openspec/changes/*/specs/mcp-server/spec.md` delta. Reading both locations makes the
test correct before and after archive: the delta adds the sixth tool now, the main spec
gains it at archive, and the union is the same in both states.

The spec change itself is the delta
`specs/mcp-server/spec.md` → MODIFIED "Exposed MCP tools", which adds
`generate_with_memory(query, strategy, droid_name)`. `openspec/specs/` is not edited
directly.

### Decision 3 — Supersede takes a real successor

The console offers two distinct actions:

- **Supersede with replacement** — the user enters the replacement text; the UI inserts
  it (`knowledge.insert_fact(...)` with a fresh embedding), then calls
  `supersede(old_id, new_id)`. The pointer names the fact that replaced it.
- **Mark superseded (no replacement)** — calls the store's supersede path with an
  explicit "no successor" value (`None`), not `0`. If the store rejects `None`
  today, the task adds a documented `mark_superseded(fact_id)` helper rather than
  passing a sentinel id.

`supersede(fact_id, 0)` is removed. A store-level test asserts `superseded_by` is either
a real facts id or `NULL`, never `0`.

### Decision 4 — Named exceptions, reported where they happen

Each bare handler becomes a named one, with the narrowest exception that the code path
can actually raise:

- `app.py:109` (`memory.states` norm): `AttributeError` (a backend without `states`),
  reported as `st.warning` once, and the metric shows `N/A` with the reason.
- `app.py:196` (memory reset): `(AttributeError, RuntimeError)`, reported with
  `st.error(f"Memory reset failed: {e}")` and the step count is not zeroed.
- `app.py:675` (memory norm in the console): `AttributeError`, same treatment as :109.

The rule is enforced by a structural test: `app.py` contains no line matching
`^\s*except\s*:\s*$`. The test fails with the line number, so the next bare handler is
caught in review.

### Decision 5 — Console panels map to existing APIs

| Panel | API |
|---|---|
| Snapshots / rollback / branch | `SnapshotManager.create_snapshot(droid, version_id)`, `restore_snapshot`, `branch_from`, `list_snapshots` |
| Merge with policy | `MergeEngine.detect_conflicts`, `merge_memories(store_a, store_b, policy)`; policies `keep_newest`, `keep_highest_novelty`, `keep_both_as_alternatives`, `require_manual` |
| Continuous stream | `ContinuousStream.start/pause/resume/stop`, `get_status`, `get_stats` |
| `decide_path` | `engine.decide_path(query)` |
| Confidence tier | `confidence_tier(c)` from `confidence.py`, plus `estimate_confidence` |
| Package export/import | `DroidPackage.export_droid`, `list_contents`, `import_droid(..., verify=True)` |
| GGUF download | `GgufBackend.download_model(repo_id, filename, local_dir)` |

The stream panel keeps its worker in `st.session_state` keyed by profile name and reuses
it on rerun, so a Streamlit rerun cannot start a second daemon. Rollback and merge take
a snapshot before they mutate state and require an explicit confirmation checkbox.

### Decision 6 — The confidence tier is labelled with what it measured

Every assistant answer shows a tier chip: `high` (>0.8), `medium` (0.5–0.8), `low`
(<0.5), with the numeric confidence and the label `retrieval-only` when no generation
confidence exists (extractive and fallback answers). For fused and LLM answers, the tier
uses `estimate_confidence(retrieval_conf, generation_conf)`; when a backend cannot
produce a generation confidence, the chip says `retrieval-only` instead of inventing a
second number. Low-tier answers keep the existing clarification prefix.

Placement: beside the answer-path badge from change 1, in the same caption row, so the
two never contradict each other (`recall` + `retrieval-only`, `fused` + fused tier).

### Decision 7 — Packages: show the digest before the import

Export writes `<name>-<timestamp>.droid` under `exports/` and then displays
`DroidPackage.list_contents()`: member names, sizes, and the manifest SHA-256 values.

Import shows the same table, requires the confirmation checkbox, and calls
`import_droid(..., verify=True)`. A digest mismatch aborts with the member name and both
digests; no partial extraction is presented as success. The archive is written outside
the profile directory so an export can never overwrite the profile it reads.

### Decision 8 — GGUF download shows path and size first

The panel asks for `repo_id`, `filename`, and `local_dir`, resolves the target path,
and shows it with the size before the download starts. The size comes from the Hub's
file metadata when reachable; when it is not, the panel says `size unknown` rather than
guessing. The download starts only after the user confirms, runs with a spinner, and
reports the returned path. A failed download is reported with the exception message —
the CLI's current `print(...)` + silent fallback pattern is not copied into the UI.

## Risks / Trade-offs

- **Self-recall overstates quality.** A store can return every fact for its own text and
  score 1.0. Mitigation: the mode is named in the output, the metric is documented as an
  index sanity check, and labeled evaluation is the recommended path.
- **UI-only behavior is untested by pytest.** Mitigation: the panel logic that can be
  wrong (successor id, policy name, digest verification) lives in store/model calls that
  do have tests; the widgets are thin.
- **Merge and rollback can destroy state.** Snapshot-before-action plus explicit
  confirmation; the merged target is always a new profile name, never an existing one.
- **Streamlit reruns and daemons.** The session-state key is the guard; a test cannot
  cover it, so the panel is validated manually and the key is documented in code.
- **Spec-parsing test fragility.** The parity test parses a narrow bullet pattern. If a
  future spec reformats the list, the test fails with the file name rather than silently
  passing.
- **`eval` adds a CLI surface without a benchmark claim.** No score is compared to an
  external baseline; the numbers describe this profile's retrieval on this input.
