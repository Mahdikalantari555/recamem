# Three-dimensional learning studio

Status: **implemented, partially verified**. This page separates what was
measured from what still needs a real browser or target hardware. The OpenSpec
change is `openspec/changes/three-dimensional-learning-studio/`; several of its
tasks stay unchecked on purpose (see [Task status](#task-status)).

## What it is

The companion app has a fourth view, **Learning observatory** (`Observatory.svelte`),
reachable from the sidebar. It shows the records this machine already holds as a
small local scene:

| Layer | Marker | Source | Measured? |
| --- | --- | --- | --- |
| Memory | one marker per active fact, capped at 24 | `GET /api/v1/companion/state` → `state.facts_active` | count only |
| Actions | one marker per recorded companion action, capped at 12 | same endpoint → `events[]` | yes, one per record |
| Runs | one marker per job, capped at 8 | `GET /api/v1/jobs` | yes, one per record |
| Core, halo, orbit rings, shell rotation | — | none | **decoration** |

The panel beside the scene lists the same records as semantic HTML, and the
**Accessible 2D** mode replaces the canvas with a table that names each record's
source. Position and orbit speed carry no measurement: markers are placed on a
deterministic golden-angle spiral, not from embeddings.

What the API does *not* expose, and the scene therefore never shows: per-fact
text, embeddings, attention, weights, loss curves as geometry, or any evidence
that training happened. Facts are a **count projection** — 24 markers for 5 000
facts is stated in the UI, not drawn as if each were a fact.

## Data safety

- The view only issues `GET` requests to the app's own origin
  (`/api/v1/health`, `/learning`, `/companion/state`, `/jobs`). Component tests
  assert that every request is a `GET` on `/api/v1/`.
- Opening the view does not start training, research, evaluation or memory
  writes. The learning supervisor is untouched; a real job only starts through
  the existing Training lab controls.
- No third-party requests, no CDN, no analytics. Three.js is bundled and served
  from the same origin (`companion/dist/assets/three.module-*.js`).

## Honest status reporting

- Background learning status comes from `GET /api/v1/learning`; in this build it
  reports **blocked** and the UI says so, with the backend's own reason string.
- Freshness comes from `state.measured_at`; older than 30 s is labelled stale.
- A failed fetch shows `Backend unavailable: …`, an explicit "no measured memory
  signals" notice, and no data markers.
- Copy states that motion is not evidence of training.

## Lifecycle

`companion/src/lib/observatoryScene.ts` is framework-free and owns everything:

- render loop runs only while motion is on **and** the document is visible;
  otherwise rendering is on demand (pointer, resize, selection);
- `prefers-reduced-motion: reduce` starts paused; an in-page button toggles it;
- `ResizeObserver` resizes the canvas, camera aspect and pixel ratio (capped at 1.5);
- `webglcontextlost` stops the loop, dims the stage, keeps the record list
  usable and offers the 2D view; `webglcontextrestored` resumes only after the
  real event, and never claims a recovery that did not happen;
- `dispose()` cancels the frame, removes listeners and the observer, disposes
  every geometry/material and the renderer, and removes the canvas, so repeated
  mount/unmount cycles do not leak;
- if Three.js fails to import or a WebGL context cannot be created, the module
  resolves `supported: false` and the UI offers the 2D view. JSdom tests cover
  both the no-WebGL path and the store's absence.

## Dependency decision (task 1.4)

Threlte was prototyped out-of-tree with the same object count (3 shells, 24
markers) and the same Three.js version:

| Variant | Bundled JS (esbuild minified) | License |
| --- | --- | --- |
| Raw Three.js | 506.6 KB | MIT |
| Threlte 8.3.1 (+ `extras` 9.4.2) | 777.3 KB (+53.4 %) | MIT |

Decision: **stay on raw Three.js**. Threlte would add ~271 KB for this scene size,
brings `mitt` and a React-free but opinionated runtime, and its on-demand
rendering advantage is already reproduced with a manual dirty flag. Nothing was
added to `package.json`. This is a bundle-size comparison only; no frame-time
claim is made either way.

## Performance budgets (task 1.9, partially measured)

`python scripts/check_observatory_budget.py` measures the committed build and
exits non-zero when a ceiling is exceeded. Measured on this runner (Node 22,
Python 3.11):

| Metric | Value | Budget |
| --- | --- | --- |
| Entry JS, gzip | 41.6 KB | 64 KB |
| Three.js chunk, gzip (lazy) | 186.0 KB | 220 KB |
| CSS, gzip | 6.3 KB | 12 KB |
| All dist assets | 876 KB | 1 100 KB |
| Live WebGL contexts | 1 | 1 |

**Not measured here:** frame timing, interaction latency, memory growth over a
long session, and CPU-training/inference contention on a 16 GB device. This
sandbox has no runnable browser (no `libnss3`, no root to install packages) and
no display, so those numbers cannot be produced honestly from it. They remain
open items, not claims.

## Testing

```bash
npm --prefix companion test     # vitest: 26 tests (12 model, 6 component, 8 existing)
npm --prefix companion run typecheck
npm --prefix companion run build
python -m pytest tests/test_companion_launch.py -q   # includes the shipped-observatory check
python scripts/check_observatory_budget.py
```

The model tests pin down the mapping rules: one node per job/event, count-capped
memory markers, blocked vs paused, staleness, backend error, determinism and
shell placement. The component tests run in jsdom and pin the no-WebGL fallback,
the 2D table with sources, GET-only traffic, reduced-motion default, the blocked
notice, and the error state.

## Fallback and rollback

- Per user: the "Accessible 2D" toggle (persisted in `localStorage`), plus automatic
  fallback when WebGL is missing or the context is lost.
- Per build: delete `companion/src/components/Observatory.svelte`, the
  `observatory` route in `App.svelte` and the sidebar button; `LearningConstellation`
  (the illustrative hero art) is independent and keeps working.
- `recamem companion` is unchanged: same API, same loopback policy, same routes.

## Known limitations

- No real browser verification in this environment; screenshots and manual
  screen-reader/contrast review have not been performed.
- Mobile/touch: dragging and tap-to-select use pointer events; pinch-zoom and
  inertial rotation are deliberately absent.
- The scene reads no fact text, so markers cannot be labelled individually
  beyond their index and count.
- The dev server now binds `0.0.0.0` for sandbox previews; production still
  ships the static build served by the loopback API, so this does not widen the
  runtime attack surface. `recamem companion` keeps its loopback-only policy.

## Task status

Honest status for `openspec/changes/three-dimensional-learning-studio/tasks.md`:

| Task | Status |
| --- | --- |
| 1.1 audit real vs decorative | done — table above, `observatory.ts` comments |
| 1.2 interaction design incl. blocked/empty/stale/error | done — component implements all four states |
| 1.3 provenance-safe mapping | done — every node carries `provenance` + `measured` |
| 1.4 Threlte comparison | done — bundle measurement above; **no frame-time comparison** |
| 1.5 selected 3D view + semantic controls | done |
| 1.6 off switch, reduced motion, forced colours, no-WebGL, keyboard/SR parity | implemented and unit-tested; **manual SR review pending** |
| 1.7 visibility, resize, context loss, disposal | implemented; **long-session leak test pending** |
| 1.8 component/API/real-browser tests | component + API done; **real-browser tests blocked** (no runnable browser here) |
| 1.9 performance budgets | bundle budgets measured and gated; **frame/16 GB contention not measured** |
| 1.10 docs, privacy, rollback, manual a11y review | docs + privacy + rollback done; **manual a11y review pending** |

Tasks that depend on a runnable browser or target hardware are left unchecked.
