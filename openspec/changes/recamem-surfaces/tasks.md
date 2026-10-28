# Tasks — RecaMem Surfaces (Tauri desktop + Tailwind/shadcn)

## 1. Standalone runner (Python, core deps only)
- [ ] 1.1 Implement `recamem/runner.py` with `run_once(prompt, engine)` returning a dict and printing one JSON line.
- [ ] 1.2 Implement `run_loop(engine, stdin)` reading lines and emitting JSON per line.
- [ ] 1.3 Add `recamem run` subcommand: `--prompt`, `--name`. Verify: prints JSON, exits 0.
- [ ] 1.4 Add `scripts/run_recamem.py` thin wrapper.
- [ ] 1.5 Test: runner works without Node/Tauri.

## 2. Python API backend
- [ ] 2.1 Implement `recamem/api.py` (FastAPI or stdlib) exposing `/api/v1/health`, `/models`, `/models/load`, `/inference`, `/jobs`, `/experiments`, `/clear`, `/bench`.
- [ ] 2.2 Wire endpoints to MemoryEngine methods; reuse `serve --rest`.
- [ ] 2.3 Add `recamem serve --web` starting API + static frontend.

## 3. Frontend scaffold (Vite + React + TS + Tailwind + shadcn)
- [ ] 3.1 Scaffold `web/` with Vite + React + TS + Tailwind.
- [ ] 3.2 Add shadcn/ui components (button, card, tabs, input, sidebar, dialog).
- [ ] 3.3 Define design tokens (colors, typography, spacing, radius) per `docs/ui-design.md` §4.
- [ ] 3.4 Implement app shell: left nav rail, main workspace, status area, collapsible nav.
- [ ] 3.5 Implement typed API client in `src/lib/api/`.
- [ ] 3.6 Build script producing `web/dist/`.

## 4. Pages (progressive)
- [ ] 4.1 Playground: input, submit, output, cancel, decision inspector (show unavailable if not provided).
- [ ] 4.2 Minimal Experiments view: list runs, open details.
- [ ] 4.3 Runtime status in sidebar (connected/disconnected, model, device, active op).
- [ ] 4.4 (Later) Architecture Lab, Training, Benchmarks, Memory Explorer, Settings.

## 5. Tauri desktop shell
- [ ] 5.1 Scaffold `src-tauri/` with Tauri 2, configure to load `web/dist`.
- [ ] 5.2 Desktop integration: window, tray, native menus (minimal).
- [ ] 5.3 `recamem ui` launches Tauri app (build if needed); browser fallback if Tauri absent.

## 6. CLI accessibility
- [ ] 6.1 Add `recamem ui` subcommand.
- [ ] 6.2 Add `recamem doctor` (anchor mode, model path, load time, memory stats; exit 0 if ready).
- [ ] 6.3 Console entry point in pyproject.toml.

## 7. Reset Memory + rebrand
- [ ] 7.1 Wire Reset Memory button to `/clear` → `engine.clear_memory()`; verify no AttributeError.
- [ ] 7.2 Rebrand "Droid" → "agent"/"RecaMem" in all UI strings.
- [ ] 7.3 Remove or thin-launcher `app.py`.

## 8. Optional dependencies
- [ ] 8.1 Add `[web]` extra (fastapi, uvicorn) to pyproject.toml.
- [ ] 8.2 Document Node + pnpm + Rust/Tauri requirement in README.
- [ ] 8.3 Test: core install works without [web]/Node; `recamem doctor` runs.

## 9. Accessibility / performance / security (per design §16-18)
- [ ] 9.1 Keyboard navigation + visible focus states.
- [ ] 9.2 Loopback-only binding, path validation, no secrets in logs.
- [ ] 9.3 Reduced-motion support.

## 10. Verification
- [ ] 10.1 `recamem run --prompt "..."` exits 0, prints JSON.
- [ ] 10.2 `recamem ui` launches desktop app (manual).
- [ ] 10.3 `recamem doctor` exits 0 on ready profile.
- [ ] 10.4 Reset Memory no AttributeError.
- [ ] 10.5 CI: runner smoke test.
- [ ] 10.6 Definition of done (design §20) checklist.
