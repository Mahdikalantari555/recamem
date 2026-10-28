# Tasks — RecaMem Surfaces (Tauri desktop + Tailwind/shadcn)

## 1. Standalone runner (Python, core deps only)
- [x] 1.1 Implement `recamem/runner.py` with `run_once(prompt, engine)` returning a dict and printing one JSON line.
- [x] 1.2 Implement `run_loop(engine, stdin)` reading lines and emitting JSON per line.
- [x] 1.3 Add `recamem run` subcommand: `--prompt`, `--name`. Verify: prints JSON, exits 0.
- [x] 1.4 Add `scripts/run_recamem.py` thin wrapper.
- [x] 1.5 Test: runner works without Node/Tauri.

## 2. Python API backend
- [x] 2.1 Implement `recamem/api.py` (FastAPI or stdlib) exposing `/api/v1/health`, `/models`, `/models/load`, `/inference`, `/jobs`, `/experiments`, `/clear`, `/bench`.
- [x] 2.2 Wire endpoints to MemoryEngine methods; reuse `serve --rest`.
- [x] 2.3 Add `recamem serve --web` starting API + static frontend.

## 3. Frontend scaffold (Vite + React + TS + Tailwind + shadcn)
- [x] 3.1 Scaffold `web/` with Vite + React + TS + Tailwind.
- [x] 3.2 Add shadcn/ui components (button, card, tabs, input, sidebar, dialog).
- [x] 3.3 Define design tokens (colors, typography, spacing, radius) per `docs/ui-design.md` §4.
- [x] 3.4 Implement app shell: left nav rail, main workspace, status area, collapsible nav.
- [x] 3.5 Implement typed API client in `src/lib/api/`.
- [x] 3.6 Build script producing `web/dist/`.

## 4. Pages (progressive)
- [x] 4.1 Playground: input, submit, output, cancel, decision inspector (show unavailable if not provided).
- [x] 4.2 Minimal Experiments view: list runs, open details.
- [x] 4.3 Runtime status in sidebar (connected/disconnected, model, device, active op).
- [ ] 4.4 (Later) Architecture Lab, Training, Benchmarks, Memory Explorer, Settings.

## 5. Tauri desktop shell
- [x] 5.1 Scaffold `src-tauri/` with Tauri 2, configure to load `web/dist`.
- [x] 5.2 Desktop integration: window, tray, native menus (minimal).
- [x] 5.3 `recamem ui` launches Tauri app (build if needed); browser fallback if Tauri absent.

## 6. CLI accessibility
- [x] 6.1 Add `recamem ui` subcommand.
- [x] 6.2 Add `recamem doctor` (anchor mode, model path, load time, memory stats; exit 0 if ready).
- [x] 6.3 Console entry point in pyproject.toml.

## 7. Reset Memory + rebrand
- [x] 7.1 Wire Reset Memory button to `/clear` → `engine.clear_memory()`; verify no AttributeError.
- [x] 7.2 Rebrand "Droid" → "agent"/"RecaMem" in all UI strings.
- [x] 7.3 Remove or thin-launcher `app.py`.

## 8. Optional dependencies
- [x] 8.1 Add `[web]` extra (fastapi, uvicorn) to pyproject.toml.
- [x] 8.2 Document Node + pnpm + Rust/Tauri requirement in README.
- [x] 8.3 Test: core install works without [web]/Node; `recamem doctor` runs.

## 9. Accessibility / performance / security (per design §16-18)
- [x] 9.1 Keyboard navigation + visible focus states.
- [x] 9.2 Loopback-only binding, path validation, no secrets in logs.
- [x] 9.3 Reduced-motion support.

## 10. Verification
- [x] 10.1 `recamem run --prompt "..."` exits 0, prints JSON.
- [ ] 10.2 `recamem ui` launches desktop app (manual).
- [ ] 10.3 `recamem doctor` exits 0 on ready profile.
- [x] 10.4 Reset Memory no AttributeError.
- [x] 10.5 CI: runner smoke test.
- [x] 10.6 Definition of done (design §20) checklist.


## Implementation notes (2026-10-28)

Shipped in this change:

| Area | Where | Tests |
| ---- | ----- | ----- |
| Headless runner (1.1–1.4) | `recamem/runner.py`, `scripts/run_recamem.py` | `tests/test_runner.py`, `tests/test_surfaces_cli.py` |
| REST API (2.1–2.3) | `recamem/api.py` (stdlib, loopback-only), `recamem serve --rest/--web` | `tests/test_api.py` |
| Frontend (3.1–3.6) | `web/` (Vite + React + TS + Tailwind, shadcn-style components in `src/components/ui`), tokens from `docs/ui-design.md` §4, committed `web/dist` | `tests/test_frontend_build.py`, `npm run build` |
| Pages (4.1–4.3) | Playground with decision inspector and cancel, Experiments listing, sidebar runtime status | `tests/test_frontend_build.py`, `tests/test_api.py` |
| Tauri shell (5.1–5.3) | `src-tauri/` (window, Refresh/Quit menu, tray, `core:default` capabilities), `recamem ui` with browser fallback | `tests/test_frontend_build.py`, `tests/test_surfaces_cli.py` |
| CLI (6.1–6.3) | `run`, `ui`, `serve --web`, extended `doctor` (anchor load time, profiles, facts) | `tests/test_surfaces_cli.py`, `tests/test_recamem_cli.py` |
| Reset + rebrand (7.1–7.3) | `POST /api/v1/clear` → `clear_memory()`, UI confirmation, `app.py` is a thin launcher, console moved to `recamem/console_legacy.py` with RecaMem branding | `tests/test_api.py`, `tests/test_console_lifecycle.py` |
| Optional deps (8.1–8.3) | `[web]` and `[console]` extras, README requirements, core-only runner tests | `tests/test_runner.py`, `tests/test_surfaces_cli.py` |
| Accessibility/security (9.1–9.3) | skip link, roving tabindex, `aria-current`, focus rings, `prefers-reduced-motion`, loopback-only binding, name/path validation, no bodies in logs | `tests/test_api.py`, `tests/test_frontend_build.py` |
| Verification (10.1, 10.4–10.6) | runner smoke in CI, Reset test, checklist in `docs/surfaces.md` | `.github/workflows/offline-contracts.yml` |

Still open, with the reason:

* **4.4** — Architecture Lab, Training, Benchmarks, Memory Explorer and Settings
  are nav entries marked "not available yet". They are the explicitly deferred
  pages; shipping inert screens would be the fabrication the design forbids.
* **10.2** — `recamem ui` cannot launch the Tauri window here: no Rust toolchain
  and no platform WebView. `recamem ui --dry-run` reports `shell=browser`
  (`binary=... exists=no`), and the fallback path is exercised by tests.
* **10.3** — `recamem doctor` exits 0 only when a real anchor loaded. Without
  ONNX weights it reports the explicit mock and exits 1, which is the correct
  behaviour and not a passing gate.
