# Design — RecaMem Surfaces (Tauri desktop + Tailwind/shadcn)

## Decisions
- Desktop shell: Tauri 2 (Rust owns shell + desktop integration, not model logic).
- Frontend: Vite + React + TypeScript + Tailwind CSS + shadcn/ui (components copied in).
- Charts: Apache ECharts, only where real time-series/comparison data exists.
- Icons: Lucide.
- Backend: reuse `recamem serve --rest` (FastAPI preferred, stdlib fallback) exposing
  versioned loopback API `/api/v1/*` per `docs/ui-design.md` §14.
- Runner: pure Python, core deps only, no Node/Tauri.
- Reset Memory calls `engine.clear_memory()`.
- Anchor mode, selected core id, memory stats from `engine.health()` + `configs/recamem_selected_core.json`.
- Modularity: now (not deferred).
- Visual identity per `docs/ui-design.md` §4: dark charcoal + muted lime accent, Inter +
  monospace, 8pt spacing, 8–12px radius, no gradients/glassmorphism.

## Module layout
- `recamem/runner.py` — `run_once(prompt, engine)`, `run_loop(engine, stdin)`.
- `recamem/api.py` — FastAPI app exposing `/api/v1/health`, `/models`, `/models/load`,
  `/inference`, `/jobs`, `/experiments`, `/clear`, `/bench`.
- `recamem/webui/__init__.py` — `launch()` (builds/serves frontend via Tauri or browser).
- `web/` — Vite project: `src/pages/{Playground,Architecture,Training,Benchmarks,Memory,Experiments,Settings}.tsx`,
  `src/components/{ui,layout,status,charts}/`, `src/lib/{api,schemas,utils}/`.
- `src-tauri/` — Tauri 2 shell (Rust), desktop integration.
- `recamem/cli.py` — add `run`, `ui`, `doctor`, `serve --web`.

## Pages (progressive)
- First release: Playground + runtime status + minimal Experiments.
- Later: Architecture Lab, Training, Benchmarks, Memory Explorer, Settings.

## Data flow
- `recamem run --prompt "..."` → load engine → run_once → print JSON line.
- `recamem run` (no prompt) → run_loop reading stdin.
- `recamem ui` → build (if needed) + launch Tauri (or serve in browser fallback).
- `recamem doctor` → health() → print items → exit 0 if ready.
- `recamem serve --web` → start API + serve built frontend on loopback.

## Ponytail notes
- React hooks only; no global state library.
- shadcn components copied in (not a runtime dep); Tailwind via Vite plugin.
- ECharts lazy-loaded only where real data exists.
- One backend process serves API + static files.
- Reuse existing REST path; no new server framework.
- Do not move/rename Python package to match UI tree.
