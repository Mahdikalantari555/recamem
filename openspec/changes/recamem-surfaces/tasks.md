# Tasks — RecaMem Surfaces

## 1. Standalone runner
- [ ] 1.1 Implement `recamem/runner.py` with `run_once(prompt, engine)` returning a dict and printing one JSON line.
- [ ] 1.2 Implement `run_loop(engine, stdin)` reading lines and emitting JSON per line.
- [ ] 1.3 Add `recamem run` subcommand: `--prompt`, `--name`, `--json` (default). Verify: `recamem run --prompt "hello" --name demo` prints JSON and exits 0.
- [ ] 1.4 Add `scripts/run_recamem.py` thin wrapper invoking the runner.
- [ ] 1.5 Test: runner works without Streamlit installed (core deps only).

## 2. WebUI rebrand
- [ ] 2.1 Create `recamem/webui/` package with `launch()`.
- [ ] 2.2 Implement tabs: Chat, Teach, Recall, Memory health, Bench.
- [ ] 2.3 Sidebar: anchor mode (real/mock), selected core id, memory stats.
- [ ] 2.4 Wire Reset Memory button to `engine.clear_memory()`; verify no AttributeError.
- [ ] 2.5 Rebrand strings "Droid" → "agent"/"RecaMem" throughout.
- [ ] 2.6 Add `recamem ui` subcommand launching Streamlit on the new app.
- [ ] 2.7 Keep `app.py` as thin launcher or remove after migration.

## 3. CLI accessibility
- [ ] 3.1 Add `recamem doctor` reporting anchor mode, model path, load time, memory stats; exit 0 only if ready.
- [ ] 3.2 Document quick start in README: `recamem run`, `recamem ui`, `recamem doctor`.
- [ ] 3.3 Add console entry point `recamem` in pyproject.toml.

## 4. Optional UI dependency
- [ ] 4.1 Add `[ui]` extra to pyproject.toml pulling Streamlit.
- [ ] 4.2 Test: core install (no [ui]) imports `recamem` and runs `recamem doctor` without Streamlit.

## 5. Verification
- [ ] 5.1 `recamem run --prompt "..."` exits 0 and prints parseable JSON.
- [ ] 5.2 `recamem ui` launches (manual).
- [ ] 5.3 `recamem doctor` exits 0 on a ready profile.
- [ ] 5.4 Reset Memory button no longer raises AttributeError.
- [ ] 5.5 CI: runner smoke test in GitHub Actions.
