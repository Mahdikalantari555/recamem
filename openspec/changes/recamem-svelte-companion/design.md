# Design

## Context

The engine already exposes everything a companion needs: `teach()`,
`chat()` (including `learn:`, `/sleep`, `/fast`, `/slow`), `recall()`,
`approve()` / `reject()` / `correct()`, `get_memory_health()`, `ttt_status()`,
`get_feedback_log()`, `list_snapshots()` and profile manifests. The stdlib API in
`recamem/api.py` exposes the first few of these but not feedback, health,
checkpoints or real job execution: `JobStore` explicitly says "Jobs are
*records*, not background threads".

`docs/ui-design.md` already asks for streaming "if the backend supports it" and
for progressive pages that never show invented data. `configs/recamem_selected_core.json`
and `scripts/train_lm.py` give us the training configuration surface, and
`docs/cloud-training.md` gives us checkpoint manifests with sha256 per file.

Goal: a Svelte app that a user can start with one command, train/select the best
*actually available* model in it, and chat with it continuously like a pet,
without lying about what is real.

## Goals / Non-Goals

**Goals:**
- One command to a working UI on a machine with only the core Python deps.
- Real training jobs (start, watch, cancel, resume) driven from the UI.
- A rigorous, provenance-backed definition of "best model".
- A playful pet whose state is a function of measured engine signals.
- Honest disclosure of anchor mode, backend, checkpoint and streaming support.
- Verifiable without Node, Rust, weights or network access.

**Non-Goals:**
- No new model architecture or training algorithm (owned by
  `recamem-language-learning`).
- No closing of the release gate (owned by
  `finish-implementation-and-close-gaps`), no real-model tier execution here.
- No replacing `web/` or `src-tauri/`, no desktop packaging.
- No training in the browser, no cloud account requirement, no weights in Git.
- No cosmetic-only rewards, no invented mood values, no fake token streaming.

## Decisions

### D1: Separate `companion/` tree, not a rewrite of `web/`
**Decision**: Svelte 5 + Vite + TypeScript lives in `companion/`; the React app
in `web/` keeps its contract and its committed `web/dist`.
**Rationale**: `recamem serve --web`, the Tauri shell and the existing frontend
tests depend on `web/dist`. A parallel surface is additive and reversible.
**Alternative**: Replace `web/` with Svelte. Rejected: it would break the shipped
surface, the desktop shell and the surface-coverage contract in one change.

### D2: One command, same origin, no runtime Node
**Decision**: `recamem companion` starts the stdlib API and serves the committed
`companion/dist` at `http://127.0.0.1:8000/companion/`; every API call is a
relative URL. Vite dev mode (`npm --prefix companion run dev`) proxies `/api`
to the Python server for development only.
**Rationale**: No CORS, no proxy in production, no browser-to-localhost network
call, and it works in a proxied preview host because URLs are relative.
**Alternative**: Two servers with a hard-coded `localhost:8000` base URL.
Rejected: browser-facing code must not call loopback from another origin.
**Host policy**: default loopback only; a remote bind requires the existing
`RECAMEM_API_ALLOW_REMOTE=1` plus an explicit `--host`, preserving the rule in
`docs/surfaces.md`.

### D3: "Best model" is an eligible checkpoint, not a config
**Decision**: The app distinguishes *selected architecture* (from
`configs/recamem_selected_core.json`) from *trained checkpoint* (weights +
`manifest.json`). A checkpoint is eligible only when it has a sha256 manifest,
a matching config, a recorded seed, dataset identity, commit and command, and a
measured held-out BPB with G0-G2 passed (G3/G4 displayed when available).
"Best" = lowest held-out BPB among eligible checkpoints, with provenance shown.
When none is eligible the UI shows `no eligible checkpoint` and chat runs in
explicit recall-only mode; it never loads random-init weights silently.
**Rationale**: The config currently says `trained: false` while the report says
`verified_winner`; calling either one "the best model" without evidence would
be a false claim.
**Alternative**: Pick the newest checkpoint. Rejected: recency is not quality.

### D4: Real training jobs, not job records
**Decision**: Add a `TrainingJobRunner` that launches `scripts/train_lm.py` as a
subprocess, captures stdout, parses tokens-seen / loss / BPB / gate events,
writes progress into `JobStore`, exposes a log tail, and cancels with SIGTERM.
`POST /api/v1/jobs` with `kind=train` becomes an executing job; other kinds
keep their existing record-only semantics.
**Rationale**: The current `JobStore` would let a UI show a progress bar for
work that never started — exactly the dishonesty the project forbids.
**Alternative**: Fire-and-forget thread inside the request handler. Rejected:
no cancellation, no log capture, and it would block the single-threaded server.

### D5: Streaming is capability-dependent and disclosed
**Decision**: `GET /api/v1/jobs/{id}/events` is SSE for training progress; chat
turns use SSE token events only when the selected backend advertises a
streaming callback (`model/gguf-backend`). Otherwise the chat shows an explicit
"generating" state and then the full turn, tagged `full_turn`, with a polling
fallback. No surface SHALL animate fake tokens.
**Rationale**: `engine.chat()` returns a string; a fake stream would be a lie.
**Alternative**: Always stream with a typing animation. Rejected.

### D6: Pet state is a pure function of measured signals
**Decision**: `recamem/companion.py` computes `mood`, `energy`, `bond` and
`knowledge` from `GET /api/v1/companion/state`: active/retired fact counts,
retrieval hit rate, approve/reject counts, `get_memory_health()`, `ttt_status()`,
time since last sleep, training progress and the selected checkpoint's status.
Each value is returned with its source field and the timestamp of the
measurement. The UI shows the raw numbers in an inspector.
**Rationale**: A Tamagotchi-like pet must react to real learning events; otherwise
"petting" is decoration.
**Alternative**: A local mood counter in the browser. Rejected: it would report
learning that never happened.

### D7: Every game action is a real engine call
**Decision**: pet/treat -> `approve(fact_id)` on the last recalled fact;
feed -> `teach(text)`; scold -> `reject(fact_id)`; correct -> `correct(...)`;
sleep -> `sleep()`; play -> `POST /api/v1/inference` chat. Each action returns
before/after measured values and an event id recorded in the feedback log. An
action with no target fact or no text SHALL be rejected with a reason rather
than awarding a cosmetic point.
**Rationale**: Keeps the game loop and the memory loop identical.
**Alternative**: Reward animations only. Rejected.

### D8: Secrets stay server-side
**Decision**: Hub push/pull runs in Python with `HF_TOKEN` read from the
environment or a prior CLI login. The token is never accepted from the browser,
never echoed in job params or logs, and never written to evidence. The UI only
sends a repo id and a confirm flag.
**Rationale**: `training/cloud-huggingface` already requires fail-fast auth
without partial uploads.
**Alternative**: Token field in the settings page. Rejected.

### D9: Verification without a browser
**Decision**: `scripts/run_companion.py` exercises the launcher and every new
endpoint against a temporary `RECAMEM_HOME` with an explicit mock anchor, and
asserts the built `companion/dist` is served. Vitest covers the Svelte stores
and components against a fixture API; `svelte-check` covers types. All CI checks
run on core Python deps plus Node for the frontend job only.
**Rationale**: The offline tier must stay runnable without weights or network.
**Alternative**: Playwright end-to-end. Rejected as a gating requirement: it
needs a browser and is brittle in CI.

## Risks / Trade-offs

- **Risk**: `BaseHTTPRequestHandler` is single-threaded; a long SSE connection
  could block. → Mitigation: `ThreadingHTTPServer` for the companion launcher,
  or SSE with short-lived connections plus polling; a concurrency test pins it.
- **Risk**: Training can take hours on CPU. → Mitigation: jobs are cancellable,
  resumable and show real elapsed time; no time estimate is invented.
- **Risk**: No eligible checkpoint will exist for a long time. → Mitigation:
  recall-only mode is a first-class, labelled state; the app is still useful for
  teaching, recall and feedback.
- **Risk**: Two frontends (`web/` and `companion/`) duplicate UI code. →
  Accepted for this change; a later consolidation may reuse the API client.
- **Risk**: Pet gamification could be read as a scientific claim. → Mitigation:
  every visible state links to its measured source and the docs call the pet a
  presentation layer, not an evaluation.

## Migration Plan

1. Add `companion/` with the Vite/Svelte scaffold, committed `dist`, and the
   API client using relative URLs.
2. Add additive routes and `TrainingJobRunner` to `recamem/api.py`; keep all
   existing routes and payloads unchanged.
3. Add `recamem/companion.py` for state derivation and action dispatch.
4. Add `recamem companion` to the CLI, `NEW_COMMANDS` and `USAGE` together so
   `scripts/offline_gate.py`'s CLI parity check stays green.
5. Add contract tests, the headless smoke script and the CI job.
6. Document in `docs/surfaces.md` and `docs/ui-design.md`; leave `web/dist` and
   `src-tauri/` untouched.

Rollback is deletion of the additive files plus the CLI entry; no profile
schema change, no data migration and no change to stored memory state.

## Verification / Definition of Done

- `recamem companion --dry-run` prints the resolved URL, host policy, dist path
  and anchor mode, and exits 0 without starting a server.
- `python scripts/run_companion.py` passes with core deps only and an explicit
  mock anchor.
- `PYTHONPATH=. pytest tests -q -k companion` passes; `offline_gate.py` and
  `validate_openspec.py` still exit 0.
- `npm --prefix companion run typecheck` and `npm --prefix companion test` pass.
- Training job smoke: a tiny fixture run starts, emits real progress events,
  writes a checkpoint with a manifest, and can be cancelled.
- Best-checkpoint selector refuses a checkpoint without provenance and shows
  `no eligible checkpoint` when the directory contains only random-init weights.

## Open Questions

- Should the companion eventually replace `web/`, or do the two surfaces stay
  side by side permanently?
- Which backend will be the first to expose a real streaming callback for chat?
- Should pet state persist across sessions in the profile, or be recomputed
  from engine signals on every load (current proposal: recomputed)?
- Is a GPU trainer adapter (Colab/Hub) needed in the first version, or is local
  `train_lm.py` with checkpoint push/pull sufficient?
