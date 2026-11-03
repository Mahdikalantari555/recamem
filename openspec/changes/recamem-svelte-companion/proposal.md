# Proposal

## Why

RecaMem can already be driven from a terminal, a loopback REST API, a React web
app and an uncompiled Tauri shell, but none of those makes the project *feel*
usable or easy to start:

- `docs/surfaces.md` lists Memory Explorer, Benchmarks, Training, Architecture
  Lab and Settings as "not available yet", `recamem serve --web` needs a Node
  build, `recamem ui` reports `shell=browser` because Rust/Tauri is not compiled,
  and `recamem doctor` exits 1 without ONNX weights.
- Training exists only as scripts: `scripts/train_lm.py`, `scripts/colab_train.py`
  and `recamem train-cloud`. `POST /api/v1/jobs` in `recamem/api.py` creates a
  **record**, not a running process, so the UI cannot actually start, watch or
  cancel a training run.
- `notebooks/recamem_colab.ipynb` is not valid JSON, so the documented Colab
  entry point cannot be opened.
- The "best model" is ambiguous today. `configs/recamem_selected_core.json`
  declares `selection_status: provisional_core` and
  `target_17m_hyperparameters.trained: false`, while
  `reports/recamem-core-selection.md` says `verified_winner`. No trained
  checkpoint is eligible to be called best, and no surface says so.
- Chat is turn-based `engine.chat()` with real teaching (`learn:` prefix),
  feedback (`approve` / `reject` / `correct`), sleep (`/sleep`) and memory
  health signals, but no surface exposes those as an ongoing, playful
  interaction. A companion pet is a good fit **only** if its reactions are
  derived from real engine state, otherwise it would be a decorative fake and
  would violate the repository's "every claim is measured" rule.

The user wants a Svelte application that starts with one command, lets them
train the best available model, and talk to it online like a game / virtual pet.
This change specifies that application as a new, additive surface. It does not
replace `web/` (React), `src-tauri/`, the CLI or the REST contract.

## What Changes

- Add a `companion/` Svelte 5 + Vite + TypeScript app with a committed
  `companion/dist` build, so no Node is needed at runtime.
- Add one launcher: `recamem companion` starts the existing stdlib REST API,
  serves `companion/dist` on the same origin at loopback, prints the URL and
  optionally opens the browser. A `--host 0.0.0.0` + explicit remote flag is
  available for preview environments; the default stays loopback-only.
- Extend the loopback API with additive companion endpoints: measured pet state,
  companion actions, feedback, checkpoint listing/selection, and a **real**
  training job runner (subprocess, progress, metrics, logs, cancellation) plus
  SSE progress with a documented polling fallback.
- Add a Training view: configuration form, live loss/BPB curves, G0-G4 gate
  badges, n-gram baselines, generated samples, resume/cancel, and a
  "best checkpoint" selector that only offers checkpoints with a valid
  manifest, provenance and measured gates. When none is eligible the UI says
  so instead of silently loading a random-init model.
- Add a Companion view: an SVG/CSS pet whose mood, energy, bond and knowledge
  are computed from measured engine signals; pet/feed/treat/scold/sleep/play
  actions that map to real `approve`, `teach`, `reject`, `correct`, `sleep` and
  `inference` calls; an online chat panel with the recalled evidence and answer
  path shown for every turn; and a before/after "what changed" log.
- Keep all honesty rules: explicit anchor/backend badges, no invented metrics,
  no cosmetic-only rewards, no token streaming claimed when the backend only
  returns a full turn, no `HF_TOKEN` in the browser or logs.
- Add contract tests and a headless smoke script so the surface is verifiable
  without Node, Rust or model weights.

## Capabilities

### New Capabilities
- `surfaces/companion-app`: One-command Svelte 5 companion surface, same-origin API, honest runtime disclosure, streaming-or-disclosed full turns, and loopback-only defaults.
- `training/companion-trainer`: In-app training job lifecycle, live metrics, gate badges, and a provenance-backed "best checkpoint" selector.
- `memory/companion-feedback`: Game-like petting actions mapped to real engine operations, with pet state derived from measured signals and an auditable before/after log.

### Modified Capabilities
- `verification/surface-coverage`: Add contract tests and a headless smoke for the companion surface without weakening the existing module-coverage rule.
- `training/cloud-huggingface`: Require checkpoint manifest/provenance eligibility for anything presented as best, a valid notebook, and server-side-only token handling.
- `droid/conversational-teaching`: Preserve teach/feedback/sleep semantics in the companion chat loop and disclose the answer path per turn.

## Impact

- **Affected code**: new `companion/` tree, `recamem/cli.py` (new `companion`
  command in `NEW_COMMANDS` and `USAGE`), `recamem/api.py` (additive `/api/v1/*`
  routes and a `TrainingJobRunner`), new `recamem/companion.py` (state derivation
  and action dispatch), `scripts/run_companion.py`, new tests under `tests/`.
- **Affected specs**: the capabilities listed above; `docs/surfaces.md` and
  `docs/ui-design.md` gain a companion section.
- **Dependencies**: Python core requirements only at runtime; Node/npm for
  building or developing the frontend; Svelte 5, Vite, TypeScript,
  `svelte-check` and Vitest as dev dependencies. No model weights, no GPU, no
  cloud account and no Rust toolchain required to *start* the app.
- **Compatibility**: additive only. `recamem serve --web` still serves
  `web/dist`, `src-tauri/` is untouched, and the MCP/CLI parity check in
  `scripts/offline_gate.py` stays green by registering `companion` in
  `NEW_COMMANDS` at the same time as the parser.
- **Non-goals**: no replacement of the React app or Tauri shell, no in-browser
  training, no new model architecture, no claim that an untrained checkpoint is
  trained, no fake pet emotions, no committed weights, no relaxation of the
  loopback-only or fail-fast anchor rules.
