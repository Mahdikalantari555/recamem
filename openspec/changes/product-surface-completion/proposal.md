# Proposal

## Why

The product surfaces the specs describe are only partly present, and several of them
fail silently rather than reporting a problem:

- **`cmd_eval` is a stub.** `src/cli.py:371-374` prints "Eval subcommand not yet
  implemented" and returns `0`, so a script that runs `eval` sees success with no
  evaluation.
- **The MCP spec undercounts the server.** `src/mcp_server.py:56-129` exposes six tools,
  including `generate_with_memory`; `openspec/specs/mcp-server/spec.md` lists five. An
  agent reading the spec cannot discover the sixth tool.
- **The console supersedes to fact `0`.** `app.py:541` calls
  `supersede(fact_id_input, 0)`, recording a successor id that does not exist. The fact
  is marked superseded with a dangling pointer.
- **Three handlers swallow every exception.** `app.py:109`, `:196`, and `:675` use a
  bare `except:`. A failed reset or a missing state buffer is reported as `N/A` or
  ignored, with no message and no cause.
- **Half the memory lifecycle has no UI.** Snapshots, rollback, branching, merging with
  a policy, and the continuous stream exist in `src/model/memory_lifecycle.py` and are
  unreachable from the console. `decide_path`, the confidence tier, `.droid`
  export/import, and the GGUF download exist in code and specifications and are likewise
  unreachable.

## What Changes

- **CLI.** Implement `cmd_eval` with a real evaluation, print the result in the
  Unix-friendly `key=value` format, and use exit codes that scripts can branch on
  (5.1, 5.2).
- **MCP.** Add `generate_with_memory` to the `mcp-server` spec delta and add a test that
  asserts the tool list matches the spec (5.3, 5.4).
- **Console correctness.** Pass the id of the new successor fact to `supersede`, and
  replace the three bare `except:` handlers with named exceptions that are reported to
  the user (5.5, 5.6).
- **Console actions.** Snapshot, rollback, and branch; merge with a conflict-policy
  selector; continuous stream with start/stop; `decide_path` with structured output;
  confidence tier beside every answer (5.7–5.11).
- **Portability.** `.droid` export and import in the UI, with digest verification before
  the import; GGUF download with the target path and size shown before the download
  starts (5.12–5.14).

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `cli` — MODIFIED "Subcommand routing", "Unix-friendly output"
- `mcp-server` — MODIFIED "Exposed MCP tools"
- `droid/memory-console` — MODIFIED "Memory Console Tab - Fact CRUD", "Memory Console
  Tab - Fact Browser"
- `memory/versioning` — MODIFIED "Point-in-time rollback", "Branching from snapshot"
- `memory/merge-engine` — MODIFIED "Memory merge with conflict detection",
  "Configurable conflict resolution policies"
- `model/confidence-estimation` — MODIFIED "Confidence thresholds for action"
- `system1-decision` — MODIFIED "Structured decision output"
- `portable-brain` — MODIFIED "ZIP archive format", "Integrity verification"
- `model/gguf-backend` — MODIFIED "Per-profile model config"

## Impact

- **Code areas:** `src/cli.py`, `src/mcp_server.py`, `app.py`,
  `openspec/changes/product-surface-completion/specs/mcp-server/spec.md` (the spec delta;
  `openspec/specs/` is not edited directly), and tests for the CLI, MCP server, package,
  and versioning surfaces.
- **Compatibility:** no engine or storage contract changes. The console additions call
  existing APIs (`SnapshotManager`, `MergeEngine`, `ContinuousStream`,
  `DroidPackage`, `GgufBackend.download_model`, `confidence_tier`, `decision_head`);
  where an API is missing a value the UI needs, the gap is named in the task rather than
  invented in the widget.
- **Destructive actions:** rollback, merge, and import can replace state. Every one
  requires an explicit confirmation and takes a snapshot first where the underlying API
  supports it.
- **Ordering:** after `droid-rtu-fusion-completion`, whose engine field, answer-path
  badge, and profile manifest these surfaces read; after `surface-test-coverage`, whose
  CLI/MCP/package tests this change extends.

## Risks

- **A stub replaced by a wrong number is worse than a stub.** `cmd_eval` must report the
  metric it actually measured and the sample it measured it on; it must not print a
  plausible score from an unlabeled run.
- **UI state and reruns.** Streamlit reruns the script on every interaction; a stream
  started twice would double-ingest. The panel keys the worker in `st.session_state` and
  reuses it, mirroring the existing single-worker rule.
- **Destructive console actions.** Rollback, merge, and import can overwrite a profile.
  Each is confirmation-gated, and rollback snapshots the current state first.
- **The spec/tool parity test couples two files.** Parsing the spec is the point; the
  test reads the main spec plus any active delta so it works before and after archive.
  A future spec edit that changes formatting could break the parse, so the parser is
  narrow and the failure message names the file.
