# Surfaces: runner, REST API, desktop app

RecaMem has three ways in. All of them use the same engine and the same
`RECAMEM_HOME`, and none of them can change what the engine remembers.

| Surface | Entry point | Dependencies | Status |
| ------- | ----------- | ------------ | ------ |
| Headless runner | `recamem run` / `scripts/run_recamem.py` | core only | shipped |
| REST API | `recamem serve --rest` | core only (stdlib) | shipped |
| Web app | `recamem serve --web` | core + built `web/dist` | shipped |
| Desktop app | `recamem ui` | core + Node (build) + Rust/Tauri (shell) | shell scaffolded, not compiled in CI |
| Legacy console | `streamlit run app.py --legacy-console` | `[console]` extra | deprecated, one release |

## Headless runner

```bash
recamem run --prompt "What is the calibration constant?" --name demo
printf 'hello\nstatus?\n' | recamem run --name demo
recamem run --mode teach --prompt "The gateway forwards readings every minute."
```

One JSON object per prompt, one per line, straight to stdout:

```json
{"anchor": "mock", "backend": "memory", "memory": [{"score": 0.62, "source": "lab.md", "text": "The calibration constant is 0.98."}], "profile": "demo", "prompt": "What is the calibration constant?", "response": "I do not have information ..."}
```

Rules the runner keeps:

* no Streamlit, no Node, no Tauri;
* `anchor` is always explicit (`mock` or `real`) — never a silent default;
* every answer carries the memory rows it was built from, or says it found
  nothing;
* failures exit non-zero and print a JSON `error` object, never a traceback.

## REST API

`recamem serve --rest` binds `127.0.0.1:8000` by default (loopback only; the
process refuses any other host unless `RECAMEM_API_ALLOW_REMOTE=1` is set) and
serves the versioned contract in `docs/ui-design.md` §14:

```
GET  /api/v1/health              GET  /api/v1/models
POST /api/v1/models/load         POST /api/v1/inference
POST /api/v1/teach               POST /api/v1/recall
POST /api/v1/clear               POST /api/v1/bench
GET  /api/v1/jobs                POST /api/v1/jobs
GET  /api/v1/jobs/{id}           POST /api/v1/jobs/{id}/cancel
GET  /api/v1/experiments         GET  /api/v1/experiments/{id}
```

* errors are JSON `{"error": "..."}` with a matching status code, never a
  traceback;
* profile names are validated (`[a-z0-9][a-z0-9._-]{0,63}`), so a request cannot
  walk out of the profile home;
* static files never escape `web/dist`;
* the request log records method, path and status only — request bodies hold
  user text.

`POST /api/v1/bench` answers `{"status": "blocked", "reason": "..."}` when no
dataset is configured. It never reports a metric it did not measure.

## Web app

```bash
npm --prefix web install
npm --prefix web run build       # emits web/dist
recamem serve --web              # API + frontend on http://127.0.0.1:8000/
```

`npm --prefix web run dev` runs Vite on `127.0.0.1:5173` with `/api` proxied to
the Python server. When `web/dist` is missing, `recamem serve --web` serves a page
that says the UI is not built and prints the build command — no blank screen.

Pages (`docs/ui-design.md` §5): Playground (prompt, answer, decision inspector,
cancel, Reset Memory), Experiments, and — since 2026-11-07 — Memory Explorer,
Benchmarks, Training, Architecture Lab and Settings. Each one is bound to a real
endpoint and shows nothing when the endpoint returns nothing:

| Page | Data source | Empty state |
| --- | --- | --- |
| Memory Explorer | `POST /api/v1/recall`, `POST /api/v1/teach` | "no stored fact matched" |
| Benchmarks | `POST /api/v1/bench` status, `GET /api/v1/experiments` | backend `blocked` reason, empty run list |
| Training | `GET /api/v1/jobs`, `POST /api/v1/jobs/{id}/cancel` | "no job has been created" |
| Architecture Lab | `GET /api/v1/health`, `GET /api/v1/models` | "unavailable"/"no profile" |
| Settings | `GET /api/v1/health` (read-only) | "unavailable" |

`tests/test_web_surfaces.py` fails if an entry is disabled again, if a page stops
calling the API client, if the former placeholder copy returns, or if a page
hardcodes a measurement.

Accessibility and motion: skip link, roving-tabindex tabs, `aria-current` on the
active nav item, visible focus rings from the design tokens, `aria-live` status
region, native `<dialog>` confirmation for destructive actions only, and a
`prefers-reduced-motion` block that disables animations.

## Desktop app

`src-tauri/` is a Tauri 2 shell that loads `web/dist`, owns the window, a
minimal menu (Refresh/Quit) and a tray icon, and talks to the Python API over
loopback. It contains no model logic.

```bash
recamem ui --dry-run     # prints shell=browser|tauri and the resolved paths
recamem ui               # Tauri binary if built, otherwise the browser fallback
```

The Rust toolchain and a platform WebView are not available in this repository's
CI environment, so the shell is scaffolded and reviewed but not compiled here.
`recamem ui --dry-run` names the missing pieces instead of implying a desktop
build:

```text
shell=browser
dist=/…/web/dist exists=yes
binary=/…/src-tauri/target/release/recamem exists=no
toolchain=missing cargo,rustc,pkg-config and a platform WebView
url=http://127.0.0.1:8000/
```

On 2026-11-07 the sandbox reports exactly that: `cargo`, `rustc` and
`pkg-config` are all absent and no WebView can be linked. `shell=tauri` is
printed only when the binary actually exists on disk.

## Definition of done (design §20)

- [x] `recamem run --prompt "..."` exits 0 and prints one JSON object.
- [x] `recamem serve --rest` answers `/api/v1/health` with the explicit anchor.
- [x] `recamem serve --web` serves the built frontend, or explains the build.
- [x] Reset Memory calls `clear_memory()` and is greyed-out-then-confirmed.
- [x] `recamem doctor` reports anchor, storage, selected core and generation
      backend; exit 0 only when the anchor really loaded.
- [x] Core install works without `[web]`, Node or Rust.
- [x] API is loopback-only and rejects path/name escapes.
- [x] Reduced motion, keyboard navigation and visible focus.
- [ ] `recamem ui` opening the Tauri window — blocked here: no `cargo`/`rustc`,
      no `pkg-config`, no platform WebView; `recamem ui --dry-run` prints
      `shell=browser` plus `toolchain=missing cargo,rustc,pkg-config and a
      platform WebView`.
- [ ] `recamem doctor` exit 0 — blocked here: no ONNX weights and no named
      generation backend, so the anchor is the explicit mock and doctor
      correctly exits 1. The command now names the expected anchor path and the
      `RECAMEM_GGUF`/`RECAMEM_RTU_CHECKPOINT` variables it looked for, and
      `_doctor_ready` pins "exit 0 only with a real anchor and a ready backend".

## Svelte companion app (additive surface)

`recamem companion` is the one-command entry point for the Svelte 5 companion
in `companion/`. It starts the same stdlib REST API and serves the committed
`companion/dist` build on the same origin, so the browser uses relative URLs and
needs no Node, Rust or model weights at runtime. The React app in `web/`, the
`serve --web` contract and `src-tauri/` are unchanged.

```bash
recamem companion                      # http://127.0.0.1:8000/
recamem companion --dry-run            # resolved URL, host policy, dist path, anchor
npm --prefix companion install         # development only
npm --prefix companion run dev        # Vite on 127.0.0.1:5173, /api proxied to Python
```

The default bind is loopback-only. A non-loopback bind requires both `--host`
and `RECAMEM_API_ALLOW_REMOTE=1`; the launcher refuses it otherwise. The UI
always displays the anchor mode, backend, profile and whether an eligible
trained checkpoint is loaded. With none available it shows `demo / recall-only`
and never presents random-init weights as a trained model.

Additive endpoints (existing routes and payloads are unchanged):

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/companion/state` | measured engine signals, derived pet state and recent events |
| `POST /api/v1/companion/act` | pet/treat/feed/scold/correct/sleep/play mapped to real engine calls |
| `GET /api/v1/feedback`, `POST /api/v1/feedback` | approve/reject/correct with a real fact id |
| `GET /api/v1/checkpoints`, `/checkpoints/best` | provenance-backed eligibility and lowest measured BPB |
| `POST /api/v1/checkpoints/select` | checksum-verified selection; refuses ineligible checkpoints |
| `POST /api/v1/jobs` (`kind=train`) | real `train_lm.py` subprocess, cancellable |
| `GET /api/v1/jobs/{id}/events` | SSE of real job log/metrics; polling is the fallback |

Chat turns are full-turn responses tagged `delivery=full_turn` until a backend
advertises a streaming callback; no synthetic tokens are animated. Training
progress is computed from parsed `tokens=` output, not elapsed time, and with no
real corpus the server uses a deterministic fixture corpus labelled
`dataset_source=fixture` — a pipeline smoke run, never a language-learning
result. `scripts/run_companion.py` verifies the whole surface headlessly.

The companion also ships an optional **Learning observatory** view: a Three.js
scene of the records above (memory count, recorded actions, jobs) with an
equivalent semantic list and table. It reads only the same-origin GET endpoints,
never starts training, and degrades to the 2D table when WebGL is unavailable.
Provenance, budgets, the Threlte decision and the unverified items are in
`docs/three-dimensional-studio.md`.
