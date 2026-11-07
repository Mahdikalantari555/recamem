# Tasks

- [ ] 1.1 Audit existing language-learning scripts and open tasks; document reused functionality and gaps.
- [x] 1.2 Provide a self-authored offline smoke corpus with license, fixed held-out examples and deterministic preparation.
- [ ] 1.3 Verify and pin a substantive corpus source, rights, hashes and splits; implement a reproducible small subset with leakage checks, or report blocked.
- [ ] 1.4 Provide CPU-native training configuration, pilot resource estimate, checkpoint/resume and explicit architecture/tokenizer metadata.
- [x] 1.5 Implement or integrate native PyTorch checkpoint loading and generation without GGUF; fail clearly on missing or incompatible artifacts.
- [ ] 1.6 Evaluate held-out BPB, n-gram baselines, generation and novel compositional probes; capture seeds, hardware and training budget.
- [ ] 1.7 Integrate memory teaching and native generation using existing ingest, retrieval and correction APIs; preserve fact provenance across restart.
- [ ] 1.8 Test before/after learning, retrieval-disabled and shuffled-memory controls, contradiction handling and retention.
- [ ] 1.9 Define optional replay-based continued-training and rollback gate; test before exposing it as supported.
- [ ] 1.10 Compare C4 and optional Muir on equivalent budgets once Muir is available; report limitations and failures without asserting emergence.

## Local corpus preparation progress

- [x] 2.1 Prepare separately supplied UTF-8 splits offline, record user-asserted rights and hashes, and reject exact normalized document overlap; test the existing training data format.
- [ ] 2.2 Verify a substantive corpus, near-duplicate leakage and legal rights independently; user-supplied data and exact-line checks do not satisfy task 1.3.
- [x] 2.3 Add deterministic held-out byte-level model and unigram/bigram baseline evaluation with data hash verification and provenance; test corruption and reproducibility.
- [ ] 2.4 Evaluate novel compositional probes, multiple seeds, retention and substantive datasets; the new byte-level evaluation alone does not satisfy task 1.6.

## Experimental native memory integration

- [x] 3.1 Add an explicitly selected prompt-budgeted native generation adapter using existing recall, with evidence and no-evidence abstention; preserve legacy chat.
- [x] 3.2 Expose the experiment via a completed-job-only API and Svelte evaluation view; test a real smoke checkpoint, provenance references, corrections and profile reload.
- [ ] 3.3 Validate grounding and behavior against before/after, no-memory and shuffled-memory controls on representative data; semantic evidence currently exposes linked fact IDs rather than resolved attribution.
- [ ] 3.4 Integrate a validated generator with normal chat, CLI and MCP; the experimental path is not a replacement for the current agent.
