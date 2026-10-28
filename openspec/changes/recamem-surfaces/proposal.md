# Proposal — RecaMem Surfaces: new UI and standalone runner

## Why

RecaMem's engine work (core selection, two-tier plasticity, forgetting, two-channel
recall) is specified in `recamem-plastic-memory-engine`, but the user-facing surfaces
are still the legacy Streamlit `app.py` ("Droid" branding, broken Reset Memory button,
no anchor-mode or memory-health visibility) and a Streamlit-only entry path. There is
no headless way to run chat/teach/recall/decide without Streamlit, so the engine cannot
be used in scripts, CI, or constrained environments. This change adds a first-class
standalone runner and a rebranded, functional WebUI, and makes both reachable through
the `recamem` CLI, while keeping the WebUI an optional dependency.

## What Changes

- **Standalone runner** (`recamem run` / `scripts/run_recamem.py`): headless loop that
  loads the selected core from `configs/recamem_selected_core.json`, runs chat, teach,
  recall, decide over stdin or a single prompt, and prints parseable JSON to stdout.
  Depends only on core deps (no Streamlit).
- **New WebUI** (`recamem ui`): rebranded Streamlit app replacing `app.py`'s legacy
  "Droid" surface. Tabs: Chat, Teach, Recall, Memory health, Bench. Sidebar shows
  anchor mode (real/mock), selected core id, memory stats. Reset Memory button wired
  to `clear_memory()`. Modularized into `recamem/webui/` (optional, may be deferred).
- **CLI accessibility**: console entry point `recamem` (already specified) gains
  `run`, `ui`, and `doctor` subcommands with documented examples; `recamem doctor`
  reports anchor mode, model path, load time, memory stats, and exits 0 only if ready.
- **Optional UI extra**: `pip install recamem[ui]` pulls Streamlit; core install stays
  lean. Runner works with core deps only.
- **Fix broken Reset Memory** (closes the `clear_knowledge`/`clear_memory` gap) so the
  WebUI button and the E2E script stop raising `AttributeError`.

## Capabilities

### New Capabilities
- `runner/standalone`: headless runner with stdin + single-prompt modes, parseable
  JSON output, core-deps-only.
- `ui/webui`: rebranded WebUI with tabs, anchor mode + memory health sidebar, working
  Reset Memory, optional dependency.

### Modified Capabilities
- `cli/accessibility`: add `run`, `ui`, `doctor` subcommands and documented quick-start.

## Non-goals
- Rewriting the UI into a SPA. Streamlit remains the web surface.
- Training a new foundation model.
- Cloud hosting or multi-user auth.

## Impact
- Adds `recamem/runner.py` and `recamem/webui/` (optional).
- Modifies `app.py` (rebrand + fix) or replaces it with a thin launcher.
- Updates `pyproject.toml` with `[ui]` extra.
- Depends on `recamem-plastic-memory-engine` Phase 0 (`clear_memory`) and Phase 1 (package rename).
