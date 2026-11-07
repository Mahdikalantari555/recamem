# RecaMem roadmap and project TODO

This is a living, aspirational inventory, not a release promise or a claim that
unchecked work exists. Detailed acceptance criteria and dependencies live in
`openspec/changes/*`; reconcile this file when they change. The scope below is
intentionally broad and cannot enumerate every future task. Mark work done only
with code, tests and evidence. Target CPU-only operation around 16 GB RAM; no new
GGUF/llama.cpp dependency. Scratch and compatible pretrained models are both
candidates. Memory updates are not model weight updates.

## P0 — foundations, safety and truthfulness

- [ ] Inventory the active OpenSpec changes, existing specifications and task files; reconcile duplicated or contradictory requirements and retain history.
- [ ] Identify canonical APIs and ownership across `recamem/` and deprecated `src/model/` shims; finish planned migration without breaking users.
- [ ] Establish CI tiers: offline unit and integration, security, CPU smoke, frontend, packaging, and opt-in real-model/hardware evaluation.
- [ ] Make every UI status traceable to measured backend state, show blocked/error states and never silently substitute a mock, random model, or fabricated score.
- [ ] Define configuration schemas, versions, compatibility policy, migrations, recovery, telemetry and structured error conventions.
- [ ] Audit dependency licenses, upstream advisories, npm vulnerabilities and reproducible lockfiles; define a recurring update and patch policy.
- [ ] Establish supported Python, CPU architecture, OS and browser matrices; test clean installs and upgrades.

## P0 — independent incremental learning

- [ ] Specify user-approved data sources and privacy scope; add inspect, revoke, exclusion and influence/rollback guidance. No silent training on chat, files or web.
- [ ] Connect canonical teach/correct/ingest events to the durable intake ledger transactionally or via a tested idempotent outbox; preserve provenance.
- [ ] Implement a singleton supervisor with durable queue, leases, idempotency, crash recovery, bounded retries, cancellation, pause and observability.
- [ ] Enforce CPU thread, memory, disk, runtime and inference-contention budgets; report throttling and blocked states. Saved settings currently do not enforce limits.
- [ ] Automatically select among locally available, licensed and compatible model/tokenizer pairs and training plans; never silently replace or download a model.
- [ ] Establish a substantive, permission-verified base dataset and optionally a compatible pretrained CPU PyTorch baseline; record licensing, hashes, versions, splits and leakage checks.
- [ ] Implement bounded incremental training with replay, checkpoint/optimizer/RNG recovery, protected held-out data and atomic candidate artifacts.
- [ ] Evaluate model quality, retention, regressions, contradictions and resource budgets against prior versions and controls; promote only on explicit gates.
- [ ] Keep inference on an immutable active checkpoint while candidates train; add audited promotion, rollback, restart and disk-full recovery.
- [ ] Expose the same truthful lifecycle through CLI, API, MCP and Svelte, with sources, status, logs, pause and resource controls.
- [ ] Offer opt-in OS service installation with permission review, startup/shutdown, uninstall and recovery; never claim learning while the host/service is stopped.
- [ ] Test multi-process races, interruptions, consent revocation, duplicate events, memory corruption and failures; validate long-running behavior on 16 GB hardware.

## P0 — memory and unified agent

- [ ] Specify versioned request, response, evidence, error and provenance records shared by CLI, REST, MCP and GUI; add parity tests.
- [ ] Audit chat, teach, recall, correction, feedback, sleep and research routes; migrate through a real coordinator with backwards compatibility.
- [ ] Provide one logical memory interface with attribution, source trust, conflict status, supersession, correction and abstention.
- [ ] Benchmark episodic SQLite/FTS+dense, plastic associative memory and combinations with independent ablations before consolidating internals.
- [ ] Preserve canonical records and reversible migrations; test persistence, backups, export/import, crash consistency and no hard-delete of facts.
- [x] Add an opt-in experimental native generation adapter that passes budgeted recall evidence and linked provenance references.
- [ ] Validate grounding with memory-disabled and shuffled-memory controls and integrate a selected generator into normal chat and all surfaces.
- [ ] Test contradictory facts, revisions, poisoning, temporal validity, retrieval misses, confidence calibration and restart behavior.

## P0 — autonomous web research and network security

- [ ] Audit legacy fetch/search and fix DNS validation-to-connect TOCTOU, redirect, provider and auxiliary-request paths before declaring network isolation.
- [ ] Build a pinned-destination broker for HTTP/HTTPS with TLS hostname verification, proxy control, IPv4/IPv6 restrictions and per-redirect revalidation.
- [ ] Enforce opt-in, robots policy, per-host pacing, budgets, cancellation, bounded content, structured errors and no arbitrary script execution.
- [ ] Stage attributable, deduplicated, unverified findings; implement user-selected automatic saving or review mode using existing correction/lifecycle APIs.
- [ ] Isolate browser and all subresources behind independently tested mandatory egress controls; disable JS/browser mode if isolation cannot be demonstrated.
- [ ] Evaluate maintained library adapters and security advisories; do not assume a crawler, HTML parser or model enforces tool permissions.
- [ ] Test SSRF, rebinding, redirects, metadata IPs, IPv6 transition addresses, subresources, malicious pages, injection, budget exhaustion and no unauthorized writes.
- [ ] Add bounded planner integration only with a capable explicitly configured model; enforce permissions outside the model and report unavailable otherwise.
- [ ] Run explicitly authorized live-site evaluation and report reproducible provenance, citations, cost, failures, latency and resource usage.

## P1 — model, data and research

- [ ] Finish Muir equation/initialization audit, independent float64 gradient oracle, state/chunk tests, parameter-matched baselines and CPU evidence.
- [ ] Profile C4 and Muir training and streaming against representative workloads; optimize only demonstrated hotspots and retain reference parity.
- [ ] Explore PyO3/maturin, Tantivy, USearch, Candle, Optimum and orchestration libraries against actual requirements, licensing and benchmarks; avoid mandatory Rust.
- [ ] Verify substantive dataset source, license, revision, hashes, document splits, near-duplicate leakage and protected evaluation; keep large artifacts out of Git.
- [ ] Extend offline local corpus ingestion with streaming, robust document formats, provenance and near-duplicate detection; current exact-line check is limited.
- [x] Add a deterministic local byte-level held-out evaluation with unigram and bigram baselines and data hash checks.
- [ ] Measure novel compositions, calibration, retention, forgetting, corrections and multiple seeds on substantive independent data; do not use smoke evaluation as a promotion gate.
- [ ] Compare native scratch and licensed compatible pretrained CPU PyTorch models under matched hardware/time/memory budgets.
- [ ] Add CPU-specific quantization/export only where architecture and tokenizer parity, quality, hardware support and benchmarks justify it.
- [ ] Record model cards, training provenance, intended use, limitations, eval sets and reproducibility metadata; do not promise general intelligence from small data.

## P1 — product experience and accessibility

- [ ] Complete dataset, training, evaluation, evidence, teaching and correction workflows, with honest unavailable and dependency states.
- [ ] Add native data-source authorization and inspection controls, background-learning lifecycle, resource settings, event history and rollback UI.
- [ ] Present exactly what was retrieved, stored or trained and when; separate weight learning from memory storage and illustrative graphics.
- [ ] Test keyboard navigation, screen readers, color contrast, reduced motion, mobile/resizing, WebGL failure and context loss.
- [ ] Add component and browser end-to-end tests covering real API jobs, cancellation, refresh, errors, pause and stale/disconnected states.
- [ ] Improve local setup, first-run diagnostics, onboarding, accessibility preferences, logs and recovery instructions; retain loopback-by-default API.
- [ ] Review Three.js bundle size and lifecycle/performance, especially under CPU training; do not advertise decorative activity as a metric.

## P1 — performance, operations and reliability

- [ ] Benchmark end-to-end latency, memory peak, disk growth, contention, startup and shutdown on actual target devices with recorded hardware and configuration.
- [ ] Test clean installation, offline execution, packaging, reproducible builds, dependency lockfiles and optional-integration failures on supported platforms.
- [ ] Add schema migrations, integrity checks, backup/restore, atomic manifests, fault injection, disk-full handling and disaster-recovery runbooks.
- [ ] Address API authentication/origin protections for any explicitly enabled remote binding, request limits and credential/privacy boundaries.
- [ ] Define security response, supply-chain checks, dependency update cadence and vulnerability triage.
- [ ] Add privacy-safe diagnostics, audit events and user-visible errors; never log secrets or private training text by default.

## P2 — larger possibilities (research, not commitments)

- [ ] Explore document/file ingestion, local connectors, scheduled research, multimodal inputs and richer source attribution behind explicit authorization.
- [ ] Investigate personal knowledge graphs, temporal reasoning, conflict resolution and selective forgetting against measured memory baselines.
- [ ] Explore accessible multilingual interaction, internationalization and low-resource language datasets with appropriate rights and evaluations.
- [ ] Explore bounded planning, tool-use permissions, user approval policies and durable workflows only after the core safety boundary is in place.
- [ ] Explore user-controlled synchronization, encrypted backup, portability and multi-device conflict handling without changing the local-first default.
- [ ] Define community contribution guidance, reproducibility challenges, security disclosure, governance and long-term maintenance ownership.

## Current implementation notes

- C7/Muir is an experimental sequential CPU reference; C4 remains the baseline. No target-hardware improvement or general language ability is established.
- The included corpus is a tiny self-authored smoke fixture. Offline preparation of user-provided splits checks exact normalized lines only and does not independently verify rights.
- The basic Studio starts bounded manual smoke runs and shows a decorative Three.js constellation. It is not a continuous-learning worker.
- The learning ledger and saved preference exist, but intake is not connected to canonical teaching, resource limits are not enforced, and training/promotion are blocked.
- Autonomous browsing and safe browser isolation remain unimplemented; existing legacy fetching has a DNS-rebinding concern.
- The current session adapter does not unify all surfaces or memory mechanisms. Native checkpoint sampling remains separate from normal chat; an explicit experimental API and evaluation view can include recalled memory.
- The sandbox is not the 16 GB target and has no Rust toolchain. Real hardware, external-data and browser tests remain outstanding.

## Active OpenSpec changes

`autonomous-incremental-learning`, `autonomous-web-research`,
`complete-companion-pet-design`, `cpu-base-learning`, `cpu-learning-svelte`,
`cpu-library-acceleration`, `cpu-muir-core`,
`finish-implementation-and-close-gaps`, `recamem-core-architecture-tournament`,
`recamem-language-learning`, `recamem-smart-online-learning`,
`recamem-svelte-companion`, `unified-agent-experience`,
`three-dimensional-learning-studio`. Archive or consolidate
only after checking the current code and evidence against each change.

## P1 — 3D Learning Observatory proposal

See `openspec/changes/three-dimensional-learning-studio/` for the design,
requirements and detailed unchecked tasks. The existing constellation is
illustrative; this proposal is not implemented.

- [ ] Design a creative, optional 3D observatory and equivalent semantic 2D workflow without requiring architecture/tokenizer decisions.
- [ ] Map only actual authorized learning, job and memory records to data-driven visuals; label decoration and expose blocked/stale/empty states and provenance.
- [ ] Compare existing Three.js with a Svelte-native Threlte prototype using measured costs, licenses, compatibility and target hardware, not demo claims.
- [ ] Implement keyboard/screen-reader parity, reduced motion, motion/3D off switch, forced colors, responsive layout and non-WebGL fallback.
- [ ] Test context loss, recovery, visibility, offscreen behavior, disposal, browser compatibility and repeated mounts in real browsers.
- [ ] Benchmark bundle/load, frame responsiveness, CPU/GPU memory and contention with CPU training; keep ML CPU-only.
- [ ] Document privacy, same-origin behavior, dependency decisions and rollout/rollback; never enable research or learning by opening the scene.

## P0 — merge validation findings (2026-11-07)

- [ ] Register the existing Muir CPU tests in the shipped-module coverage map; the full suite currently reports `src/model/cores/muir.py` as unmapped.
- [ ] Investigate the reproducible remote-sensing end-to-end profile reload assertion failure before claiming persistence is fully validated.
- [ ] Investigate timing-test sensitivity: the benchmark ratio test was skipped under load in the full run but passed when rerun alone.
- [ ] Rerun the complete Python suite and required CI checks after the above issues are resolved; the latest run had 1,161 passed, 5 skipped and 4 failed.
