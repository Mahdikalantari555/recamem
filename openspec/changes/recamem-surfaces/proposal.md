# Proposal — RecaMem Surfaces: Tauri desktop app (Tailwind+shadcn) + standalone runner

## Why

RecaMem's engine work is specified in `recamem-plastic-memory-engine`, but the
user-facing surfaces are still the legacy Streamlit `app.py` ("Droid" branding,
broken Reset Memory button, no anchor-mode or memory-health visibility) and a
Streamlit-only entry path. There is no headless way to run chat/teach/recall/decide
without Streamlit, so the engine cannot be used in scripts, CI, or constrained
environments. The project is beyond MVP, so the UI should be a proper local-first
desktop app (Tauri 2 + React + Vite + Tailwind + shadcn/ui) per `docs/ui-design.md`,
while the standalone runner stays a lean Python process on core deps only.

## What Changes

- **Standalone runner** (`recamem run` / `scripts/run_recamem.py`): headless loop
  that loads the selected core, runs chat/teach/recall/decide over stdin or a
  single prompt, prints parseable JSON. Core deps only (no Node, no Tauri).
- **Desktop app** (`recamem ui`): Tauri 2 shell wrapping a Vite + React +
  TypeScript SPA styled with Tailwind + shadcn/ui, replacing `app.py`. Pages per
  `docs/ui-design.md`: Playground (first release), Architecture Lab, Training,
  Benchmarks, Memory Explorer, Experiments, Settings. First release ships
  Playground + runtime status + minimal Experiments; other pages follow as backend
  capabilities mature.
- **Python backend for the UI**: reuse `recamem serve --rest` exposing a versioned
  loopback API (`/api/v1/...`) per the contract in `docs/ui-design.md` §14.
  Tauri/Rust owns the shell and desktop integration, not model logic.
- **CLI accessibility**: console entry point `recamem` gains `run`, `ui`, `doctor`,
  `serve` subcommands.
- **Optional UI dependency**: core install stays lean; desktop app needs Node +
  pnpm + Rust/Tauri toolchain and is built separately (`web/` + `src-tauri/`).
  Python side needs only `fastapi`+`uvicorn` (or stdlib) for `serve`, via `[web]`.
- **Fix broken Reset Memory** (closes `clear_knowledge`/`clear_memory` gap).
- **Design tokens + visual identity** per `docs/ui-design.md` §4 (dark charcoal +
  muted lime accent, Inter + monospace, 8pt spacing, 8–12px radius).

## Ponytail (minimal) discipline
- No Redux/Zustand: React hooks + fetch to backend.
- shadcn components copied in (not a runtime dep); Tailwind via Vite plugin.
- ECharts added only where real time-series/comparison data exists.
- Progressive pages: ship Playground first; do not implement all pages before
  confirming runtime APIs.
- No fabrication of metrics/capabilities/decision traces; show "unavailable" instead.
- Adapt to repository: do not move/rename the Python package to match the UI tree.

## Capabilities

### New Capabilities
- `runner/standalone`: headless runner, core-deps-only, JSON output.
- `ui/desktop`: Tauri 2 + React + Vite + Tailwind + shadcn SPA, optional build,
  Python launcher, design tokens, progressive pages.

### Modified Capabilities
- `cli/accessibility`: add `run`, `ui`, `doctor`, `serve --web`.
- `serve/rest`: extend REST microdaemon with versioned `/api/v1/*` endpoints.

## Non-goals
- Cloud hosting or multi-user auth.
- Training a new foundation model.
- Visual architecture editor that silently changes model code.
- Plugin system before core workflows work.

## Impact
- Adds `web/` (frontend), `src-tauri/` (Tauri shell), `recamem/webui/` (launcher),
  `recamem/api.py` (REST).
- Modifies `app.py` to a thin launcher or removes it after migration.
- Updates `pyproject.toml` with `[web]` extra (fastapi, uvicorn).
- Depends on `recamem-plastic-memory-engine` Phase 0 (`clear_memory`) and Phase 1 (package rename).
- Requires Node + pnpm + Rust/Tauri on dev machines (documented in README).
- Design spec: `docs/ui-design.md`.
