# Tasks

> **Progress (2026-11-03).** The companion app is implemented and
> runnable: one-command launcher, committed Svelte build, additive measured
> state/action/feedback/checkpoint endpoints, a real cancellable training
> job with SSE, and the three Svelte views. Verification is recorded in
> `reports/evidence/2026-11-03-companion-offline.json`. Remaining open tasks:
> 3.5 (explicit elapsed/log-tail fields on the job record), 5.5 (live loss
> chart and generated samples) and 5.7 (Hub push/pull UI). Tasks 6.5/6.6
> are closed by the evidence record and the verification commands below.
> A checked box is not evidence; the evidence record is.

No box is checked until its command, test and evidence artifact exist. The
companion is additive: no existing route, payload, profile schema or stored
memory behaviour changes.

## 1. Scaffold and one-command launch

- [x] 1.1 Create `companion/` with Svelte 5, Vite, TypeScript, `svelte.config.js`, `vite.config.ts`, `tsconfig.json`, `package.json` and `package-lock.json`; pin versions and add `dev`, `build`, `preview`, `typecheck` and `test` scripts
- [x] 1.2 Configure Vite to proxy `/api` to the Python server in dev mode only, and to emit relative asset URLs so the built app works behind a proxy without any hard-coded localhost base URL
- [x] 1.3 Build the app and commit `companion/dist` following the `web/dist` convention; add a test that `companion/dist/index.html` exists and references existing hashed assets
- [x] 1.4 Add `companion` to `recamem.cli.NEW_COMMANDS`, `USAGE` and `build_parser()` in the same change so `scripts/offline_gate.py`'s CLI parity check remains green
- [x] 1.5 Implement `cmd_companion`: start the stdlib REST API with `companion/dist` as the static root, print the resolved URL, support `--port`, `--host`, `--name`, `--no-browser` and `--dry-run`, and keep loopback-only defaults with explicit remote opt-in
- [x] 1.6 Use a threading HTTP server (or an equivalent documented design) so an SSE connection cannot block unrelated API calls, and pin it with a concurrency test
- [x] 1.7 Add `scripts/run_companion.py`, a headless smoke that starts the server on a temporary `RECAMEM_HOME` with `RECAMEM_ANCHOR=mock`, checks `/api/v1/health` and the companion index, and exits non-zero with a JSON error on failure
- [x] 1.8 Add `tests/test_companion_launch.py` covering `--dry-run`, dist presence, host policy rejection and loopback default

## 2. Additive API and measured state

- [x] 2.1 Add `recamem/companion.py` with `companion_state(engine)` that derives mood, energy, bond and knowledge only from measured engine signals, returning each value with its source field and measurement timestamp
- [x] 2.2 Implement `GET /api/v1/companion/state` returning anchor mode, backend, profile, active/retired fact counts, retrieval hit rate, feedback counts, memory health, TTT status, last sleep time, selected checkpoint and model status
- [x] 2.3 Implement `GET /api/v1/feedback` (or reuse `get_feedback_log`) and `POST /api/v1/feedback` exposing `approve`, `reject` and `correct` with validated fact ids and text
- [x] 2.4 Implement `POST /api/v1/companion/act` for `pet`, `treat`, `feed`, `scold`, `correct`, `sleep` and `play`, mapping each to the real engine call and returning before/after measured values plus an event id
- [x] 2.5 Reject an action with no target fact or empty text with a JSON error and a reason, so no cosmetic-only reward can be awarded
- [x] 2.6 Add `tests/test_companion_state.py` asserting every displayed field is derived from an engine signal, and that a mock anchor is reported as mock in the state payload
- [x] 2.7 Add `tests/test_companion_actions.py` asserting pet calls `approve`, feed calls `teach`, scold calls `reject`, sleep calls `sleep`, and the before/after values differ when the engine state changes
- [x] 2.8 Keep every existing `/api/v1/*` route and response shape unchanged, pinned by the existing `tests/test_api.py` suite

## 3. Real training jobs

- [x] 3.1 Add `TrainingJobRunner` to `recamem/api.py` that launches `scripts/train_lm.py` as a subprocess with an explicit argv, working directory and environment, and rejects unknown args
- [x] 3.2 Parse stdout for tokens seen, tok/s, loss, held-out BPB, n-gram baselines, G0-G4 gate results and checkpoint writes; store them as job metrics with provenance
- [x] 3.3 Make `POST /api/v1/jobs` with `kind=train` start a real job while other job kinds keep their record-only semantics, and add a test that a train job cannot report `done` without a subprocess result
- [x] 3.4 Implement cancellation via `POST /api/v1/jobs/{id}/cancel` with SIGTERM, a persisted `cancelled` status and no orphan process, with a test
- [ ] 3.5 Add `GET /api/v1/jobs/{id}` log tail, elapsed wall clock, progress from actual tokens seen, and the produced checkpoint path; never compute progress from elapsed time alone
- [x] 3.6 Add `GET /api/v1/jobs/{id}/events` as SSE for training progress, with a documented polling fallback and a test that no synthetic progress event is emitted
- [x] 3.7 Support resume from a checkpoint with matching vocab size, architecture and seed, rejecting mismatched configs per `training/cloud-huggingface`
- [x] 3.8 Add `tests/test_companion_training.py` with a tiny fixture training command that starts, emits real events, writes a manifest and can be cancelled
- [x] 3.9 Ensure `HF_TOKEN` is read only server-side, never accepted in a request payload, never stored in job params and never written to logs or evidence, with a test

## 4. Best-checkpoint selection

- [x] 4.1 Define the checkpoint eligibility contract (sha256 manifest, config, seed, dataset identity, commit, command, measured held-out BPB, G0-G2 results) in `docs/surfaces.md` and `recamem/companion.py`
- [x] 4.2 Implement `GET /api/v1/checkpoints` listing local checkpoints with provenance, gate results and an `eligible` flag computed from the manifest
- [x] 4.3 Implement `GET /api/v1/checkpoints/best` returning the eligible checkpoint with the lowest held-out BPB and its provenance, or an explicit `none` result with the reason
- [x] 4.4 Implement `POST /api/v1/checkpoints/select` that verifies checksums and config before loading, refuses a random-init or unprovenanced checkpoint, and records the selection in the profile
- [x] 4.5 Add `tests/test_companion_checkpoints.py` covering ineligible, tied, missing and tampered-manifest cases
- [x] 4.6 Repair `notebooks/recamem_colab.ipynb` into valid JSON (coordinated with `finish-implementation-and-close-gaps` task 2.2) and add a notebook validity test referenced by `training/cloud-huggingface`

## 5. Svelte UI

- [x] 5.1 Create the app shell with three views (Companion, Training, Inspector), a runtime status bar showing anchor/backend/checkpoint/model status, and a visible `demo / recall-only` badge when no eligible checkpoint exists
- [x] 5.2 Implement the Companion view: SVG/CSS pet driven by `/api/v1/companion/state`, pet/feed/treat/scold/correct/sleep/play controls, and an inspector listing the raw measured values behind each pet attribute
- [x] 5.3 Implement the online chat panel: relative-URL calls to `POST /api/v1/inference`, per-turn recalled evidence and answer path, `learn:` teaching, and a before/after "what changed" log from action responses
- [x] 5.4 Implement chat progress as real SSE token events when the backend advertises streaming, otherwise an explicit generating state followed by a `full_turn` response; never animate fake tokens
- [ ] 5.5 Implement the Training view: config form bound to `train_lm.py` flags, start/resume/cancel, live loss/BPB chart from job metrics, n-gram baselines, G0-G4 badges and generated samples
- [x] 5.6 Implement the best-checkpoint selector UI showing provenance, BPB, gates and the reason when no checkpoint is eligible
- [ ] 5.7 Implement Hub push/pull controls that send only a repo id and a confirm flag, with confirmation before any expensive action and server-side token handling
- [x] 5.8 Add accessibility parity with `docs/ui-design.md`: skip link, keyboard navigation, visible focus, `aria-live` status, native dialog for destructive actions and `prefers-reduced-motion` support
- [x] 5.9 Add a Vitest suite for the API client, stores, state derivation and every view, using a fixture API so no weights or network are needed
- [x] 5.10 Add a test that no view renders an invented metric, a fake stream, or a success state for a `blocked` endpoint response

## 6. Documentation and CI

- [x] 6.1 Document `recamem companion`, the dev-mode Vite proxy, the remote-host opt-in and the recall-only state in `docs/surfaces.md`
- [x] 6.2 Add a companion section to `docs/ui-design.md` covering the pet state derivation table, action-to-engine mapping and streaming disclosure
- [x] 6.3 Update `README.md` with the one-command quick start, without a hand-written test count (offline gate constraint)
- [x] 6.4 Add a CI job that runs `svelte-check`, Vitest, `scripts/run_companion.py` and `pytest -q -k companion` on core deps plus Node
- [x] 6.5 Verify `scripts/offline_gate.py`, `scripts/validate_openspec.py` and `pytest tests -q` remain green after all additions
- [x] 6.6 Add an evidence record for the companion tier with commit, command, dependency versions, configuration and outcome, and reference it from the change's completion review
