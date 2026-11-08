# Tasks

- [x] 1.1 Audit the existing constellation, workspace, API contracts and dependency licenses; document what is real data versus illustrative decoration.
- [x] 1.2 Produce and review a visual interaction design for the observatory and equivalent accessible 2D workflows, including blocked, empty, stale and error states.
- [x] 1.3 Define provenance-safe, versioned mapping of real job, memory and learning-status records to visual elements; avoid speculative model-internal claims.
- [ ] 1.4 Benchmark existing raw Three.js against an optional Svelte-native Threlte prototype; record compatibility, licensing, bundle size and adoption decision.
  Done for bundle size: raw Three.js 506.6 KB vs Threlte 777.3 KB (+53.4 %), decision: raw Three.js (`docs/three-dimensional-studio.md`). Frame-time comparison not performed (no runnable browser).
- [x] 1.5 Implement the selected 3D view with semantic HTML controls and source inspection, preserving current 2D functionality and default permissions.
- [x] 1.6 Implement off switch, reduced motion, forced-colors support, responsive layout, no-WebGL fallback and keyboard/screen-reader parity.
- [ ] 1.7 Handle visibility, offscreen rendering, resizing, context loss/restoration and disposal; test repeated mounting and long sessions for leaks.
  Visibility, resize, context loss/restore and disposal implemented in `observatoryScene.ts`; repeated-mount and long-session leak measurement still needs a real browser.
- [ ] 1.8 Add component, API-contract and real-browser tests for state truthfulness, data safety, interaction, fallbacks, context loss and accessibility.
  12 model tests and 6 jsdom component tests pass. Real-browser tests (WebGL, context loss, keyboard, responsive) are blocked: this sandbox has no runnable Chromium (missing system libraries, no root).
- [ ] 1.9 Set and measure performance budgets including bundle, startup, rendering and CPU-training/inference contention on representative and 16 GB target devices.
  Bundle budgets measured and enforced by `scripts/check_observatory_budget.py` (entry 41.6 KB gz, three 186.0 KB gz, CSS 6.3 KB gz). Frame timing and 16 GB contention unmeasured.
- [ ] 1.10 Document operation, limitations, privacy, fallback, dependency decisions, browser coverage and rollback; perform manual accessibility review.