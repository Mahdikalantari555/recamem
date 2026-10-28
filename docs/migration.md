# Migrating to `recamem`

The engine code now lives in the `recamem/` package and the command is
`recamem`. The old import paths and command keep working for one release.

## What moved

| Before | After |
| ------ | ----- |
| `src/model/droid.py` | `recamem/engine.py` (`DroidEngine`, also exported as `MemoryEngine`) |
| `src/model/droid_manager.py` | `recamem/manager.py` (`DroidManager`, also `AgentManager`) |
| `src/model/onnx_anchor.py` | `recamem/anchor.py` (`OnnxMiniLM`, also `MiniLMAnchor`) |
| `src/model/knowledge_store.py` | `recamem/episodic.py` |
| `src/model/sleep_consolidation.py` | `recamem/consolidation.py` |
| `src/model/propositions.py` | `recamem/segmenter.py` |
| `src/model/gguf_backend.py` | `recamem/gguf.py` |
| `src/model/byte_rtu_backend.py` | `recamem/byte_rtu.py` |
| `src/model/memory_lifecycle.py` | `recamem/lifecycle.py` |
| `src/model/memory_injection.py` | `recamem/injection.py` |
| `src/model/droid_package.py` | `recamem/package.py` (`DroidPackage` / `AgentPackage`) |
| `src/model/benchmarks.py` | `recamem/bench/metrics.py` |
| `src/model/continual_bench.py` | `recamem/bench/continual.py` |
| `src/cli.py` | `recamem/cli.py` (the old module still runs) |

Every `src/model/*` module that moved is a shim: it re-exports the real module
and emits a `DeprecationWarning`. The shim replaces itself in `sys.modules`, so
`monkeypatch.setattr("src.model.droid.DroidEngine", ...)` still patches the real
class. `src/model/rtu.py`, `src/model/cores/`, `src/model/tournament/` and
`src/model/recamem_core.py` stay in place: they are the training/research tree,
not the engine surface.

## Profile home

Profiles live under `~/.recamem/agents/<name>/`. `RECAMEM_HOME` overrides the
root. A legacy `droids/<name>/` directory is read in place when it exists, so
nothing breaks on the first run.

```bash
export RECAMEM_HOME=/data/recamem        # optional
recamem migrate droids/                  # copy legacy profiles into the home
```

`recamem migrate` copies, records a SHA-256 checksum of every file in
`<home>/agents/migration.json`, verifies the copy, and leaves the source
byte-identical. Re-running it with an unchanged source prints
`status=already_migrated` and writes nothing.

## Compatibility window

* `python -m src.cli <command>` prints a deprecation notice naming `recamem` and
  behaves identically.
* `tmt-droid` (installed script) prints the same notice.
* MCP tools are `recamem_<name>`; the old names stay callable and are listed
  with `deprecated: true`.
* `.droid` archives gain `format_version`; version 1 archives still import, a
  newer archive is refused before extraction.

## Rollback

There is nothing to roll back in the data: migration copies, it never moves or
deletes. To return to the previous code, check out the previous commit — the
`droids/` profiles are untouched and the old modules still read `config.json`
without `schema_version` (it is treated as version 1 and upgraded on the next
save).
