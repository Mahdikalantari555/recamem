# Plan 11 — RecaMem Surfaces (Tailwind+shadcn WebUI + standalone runner)

Status: **plan (not implemented)**. Depends on `recamem-plastic-memory-engine` Phase 0 (`clear_memory`) and Phase 1 (package rename).

## Decisions (recorded via grill session)

| # | Decision |
|---|---|
| 1 | WebUI: Tauri 2 desktop shell + Vite + React + TS + Tailwind + shadcn/ui (not Streamlit). Per `docs/ui-design.md`. Project beyond MVP. |
| 2 | Standalone runner: full scope (chat + teach + recall + decide + doctor). |
| 3 | `app.py` modularization: now (not deferred). |
| 4 | Rename `tmt.plastic_adapters` → `recamem.plastic_adapters` with one-release alias. |
| 5 | Remove `.mimocode/` tooling. |
| 6 | GLOSSARY.md + ADRs: after Phase 1 rename (module paths stable). |
| 7 | Design spec: `docs/ui-design.md` (copied from user's design.md). Visual identity: dark charcoal + muted lime. |

## Design reference
`docs/ui-design.md` — full UI/UX spec (Tauri, React, Tailwind, shadcn, ECharts, 7 pages,
design tokens, a11y, perf, security, phased implementation, definition of done).

## Ponytail discipline
- React hooks only; no global state library.
- shadcn components copied in (not a runtime dep); Tailwind via Vite plugin.
- One backend process serves API + static files.
- Reuse existing REST microdaemon; no new server framework.

## Non-implementation todo (do before coding)
- [ ] Archive superseded OpenSpec changes: `delta-plastic-memory`, `autonomous-sleep-consolidation`, `edge-slm-optimization` (absorb into `recamem-plastic-memory-engine`).
- [ ] Add this plan + `09`/`10` to read-order index in `plans/README.md`.
- [ ] Create `.github/workflows/` CI (pytest + offline fixture gate) — config only, no code.
- [ ] Create `GLOSSARY.md` skeleton (deferred until Phase 1 rename).
- [ ] Remove `.mimocode/` directory.
- [ ] Confirm Node + pnpm availability on dev machines; document mirror (registry.npmmirror.com).

## OpenSpec change
`openspec/changes/recamem-surfaces/` — proposal, design, tasks, specs (runner/ui/cli-accessibility).
Supersedes the Streamlit-only WebUI assumptions in `recamem-plastic-memory-engine` Phase 8.4.
