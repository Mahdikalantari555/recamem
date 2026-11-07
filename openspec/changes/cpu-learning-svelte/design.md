# Design

Extend `companion/` and the existing same-origin API. Organize the workflow into Dataset, Train, Evaluate, and Teach panels, reusing existing job management and chat rather than duplicating them. Dataset displays fixture versus substantive corpus, license, hash, splits, token count, and blocked status. Train exposes C4/Muir, CPU threads, sequence and token budgets, checkpoint location and measured pilot speed; calculate ETA from actual run data and label uncertain. Show honest queued/running/completed/failed/cancelled states and resume only validated checkpoints.

Evaluate shows held-out loss/BPB, n-gram baselines, tasks and evidence provenance; do not call a model intelligent or best based only on loss. Teach accepts explicit facts with source, confirmation and corrections, shows retrieved evidence, memory-disabled comparison and whether the answer is generated or retrieved. The UI must clearly separate memory changes from weight updates and show missing backend/corpus/checkpoint errors. Depend on real API responses; proposed but not implemented services remain visibly unavailable.

Default to loopback and same-origin relative API URLs, protect writes using existing origin/security controls, validate paths and avoid exposing secrets, support keyboard/screen reader use, responsive layout, progress polling fallback, and preserve accessibility of logs/errors. Add component tests, API integration tests, training cancellation/restart tests and an end-to-end CPU smoke workflow. Do not advertise GGUF as required.

The workspace may use a lazy-loaded local Three.js constellation as decorative
artwork, not as a model visualization or evidence of activity. It must remain
usable without WebGL, respect reduced-motion, pause when the page is hidden,
release GPU resources when unmounted and show real learning state in text.
