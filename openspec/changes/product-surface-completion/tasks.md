# Tasks

Task IDs match this change's numbering. Group headers are descriptive so the validator
does not read them as a second numbering scheme.

## CLI and MCP

- [ ] 5.1 Implement `cmd_eval` (`src/cli.py:371-374`). It currently prints a notice and returns 0. Add `--name`, `--queries`, `--top-k`, `--sample`; evaluate labeled precision@k/recall@k/MRR when a query file is given, otherwise run the documented self-recall sanity check.
- [ ] 5.2 Print the evaluation result in the Unix-friendly format from the `cli` capability. Emit stable `key=value` lines to stdout with no ANSI, send errors to stderr with an `error:` prefix, and return 0 success / 1 evaluation failure / 2 invalid input.
- [ ] 5.3 Add `generate_with_memory` to the tool list in `openspec/specs/mcp-server/spec.md`. The server exposes 6 tools (`src/mcp_server.py`). The spec lists 5. Make the edit through this change's `specs/mcp-server/spec.md` delta (the main spec updates at archive), documenting the tool's arguments and result.
- [ ] 5.4 Add a test that asserts the tool list matches the spec. Parse the tool names from `openspec/specs/mcp-server/spec.md` plus active `openspec/changes/*/specs/mcp-server/spec.md` deltas and compare with `handle_tools_list`; fail naming the file and the extra or missing tool.

## Console correctness

- [ ] 5.5 Replace `supersede(fact_id_input, 0)` (`app.py:541`). The call must pass the id of the new successor fact. Add "supersede with replacement" (insert the replacement, then supersede with its id) and "mark superseded" (explicit no-successor, never `0`); test at store level that `superseded_by` is a real id or `NULL`.
- [ ] 5.6 Replace the bare `except:` at `app.py:109`, `app.py:196`, and `app.py:675`. Each handler must name the exception and report it. Use the narrowest exception the path can raise, show the message to the user, and do not zero the step count when a reset fails. Add a structural test that `app.py` contains no bare `except:`.

## Console actions

- [ ] 5.7 Add the snapshot, rollback, and branch actions to the memory console. Use `SnapshotManager.create_snapshot/restore_snapshot/branch_from/list_snapshots`; snapshot before a rollback and require confirmation.
- [ ] 5.8 Add the merge action with a conflict policy selector. Offer `keep_newest`, `keep_highest_novelty`, `keep_both_as_alternatives`, and `require_manual`; show detected conflicts before merging; write the result to a new profile name, never an existing one.
- [ ] 5.9 Add the continuous stream panel with a start control and a stop control. Key the worker in `st.session_state` per profile so a rerun cannot start a second daemon; expose pause/resume and the status/stats readouts.
- [ ] 5.10 Add the `decide_path` panel with the structured decision output. Show `action`, the probability distribution, `confidence`, and `latency_ms` from `engine.decide_path(query)`.
- [ ] 5.11 Add the confidence tier display. Show the tier beside every answer. Use `confidence_tier`; label extractive and fallback answers `retrieval-only`; keep the low-tier clarification prefix.

## Portability

- [ ] 5.12 Add a `.droid` package export in the UI. The archive must contain the profile files. Export outside the profile directory, then show member names, sizes, and manifest digests from `DroidPackage.list_contents`.
- [ ] 5.13 Add a `.droid` package import in the UI. Verify the digest before the import. Show the manifest table, require confirmation, call `import_droid(..., verify=True)`, and abort with the member name and both digests on a mismatch.
- [ ] 5.14 Add a GGUF download action. Show the target path and the size before the download starts. Resolve the path from `repo_id`/`filename`/`local_dir`, show `size unknown` when the Hub gives no metadata, require confirmation, run with a spinner, and report the returned path or the exception message.
