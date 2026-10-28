# Spec Delta — WebUI (Tauri desktop + Tailwind/shadcn)

## MODIFIED Requirements

### Requirement: Rebranded desktop WebUI
The UI SHALL be a Tauri 2 desktop app wrapping a Vite + React + TypeScript SPA
styled with Tailwind CSS and shadcn/ui, replacing the legacy Streamlit `app.py`.
First release SHALL ship Playground + runtime status + minimal Experiments; other
pages follow as backend capabilities mature. Visual identity per `docs/ui-design.md` §4
(dark charcoal + muted lime accent).

#### Scenario: Reset Memory works
- **WHEN** user clicks "Reset Memory"
- **THEN** the UI SHALL call `/api/v1/clear` → `engine.clear_memory()`
- **THEN** no `AttributeError` SHALL be raised

#### Scenario: Anchor mode visible
- **WHEN** the UI loads
- **THEN** the sidebar/status area SHALL show anchor mode (real or mock)

#### Scenario: No fabricated metrics
- **WHEN** the backend does not provide a metric
- **THEN** the UI SHALL show "unavailable", never an invented value

## ADDED Requirements

### Requirement: Optional web/desktop dependency
The desktop app and its build toolchain SHALL be optional. Core functionality
(runner, doctor) SHALL work without Node, Rust, or Tauri.

#### Scenario: Core-only install
- **WHEN** `recamem` is installed without `[web]` and without Tauri toolchain
- **THEN** `recamem run` and `recamem doctor` SHALL work
- **THEN** `recamem ui` SHALL report that the desktop build is required
