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

Pages ship progressively (`docs/ui-design.md` §5): Playground (prompt, answer,
decision inspector, cancel, Reset Memory) and a minimal Experiments view are
live. Memory Explorer, Benchmarks, Training, Architecture Lab and Settings are
listed in the nav as *not available yet* rather than showing invented data.

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
`recamem ui` reports that honestly: `shell=browser` and
`binary=... exists=no`.

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
- [ ] `recamem ui` opening the Tauri window — blocked here: no Rust toolchain
      or WebView (`recamem ui --dry-run` prints `shell=browser`).
- [ ] `recamem doctor` exit 0 — blocked here: no ONNX weights, so the anchor is
      the explicit mock and doctor correctly exits 1.
