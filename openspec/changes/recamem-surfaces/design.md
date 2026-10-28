# Design — RecaMem Surfaces

## Decisions
- Runner is a pure-Python loop over `MemoryEngine` + selected core; no Streamlit import.
- WebUI stays Streamlit but is rebranded and modularized into `recamem/webui/` (pages per tab).
- UI is an optional extra (`recamem[ui]`); `recamem run` and `recamem doctor` work on core deps only.
- Reset Memory calls `engine.clear_memory()` (alias `clear_knowledge`), which clears knowledge store, resets memory, and zeroes step_count.
- Anchor mode, selected core id, and memory stats are read from `engine.health()` and `configs/recamem_selected_core.json`.

## Module layout
- `recamem/runner.py` — `run_once(prompt, engine)` and `run_loop(engine, stdin)`.
- `recamem/webui/__init__.py` — `launch()` entry for `recamem ui`.
- `recamem/webui/pages/` — chat.py, teach.py, recall.py, health.py, bench.py.
- `recamem/cli.py` — add `run`, `ui`, `doctor` subcommands.

## Data flow
- `recamem run --prompt "..."` → load engine → run_once → print JSON line.
- `recamem run` (no prompt) → run_loop reading stdin lines → JSON per line.
- `recamem ui` → check Streamlit installed → launch webui.
- `recamem doctor` → health() → print items → exit 0 if all ready.
