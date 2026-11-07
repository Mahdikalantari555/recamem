# Design

Begin with `recamem.runner.run_once`: introduce a single `AgentSession` entry point whose initial adapter delegates to the existing behavior. Keep legacy payloads, errors and mutations unchanged; CLI and API already use `run_once`, and MCP/Svelte paths must be audited before claiming parity. Distinguish requested intent, actual backend, evidence, confidence and unavailable state in a later versioned contract; never manufacture them.

Memory consolidation is incremental. Treat SQLite facts and provenance as the source of truth while exposing existing sparse/dense and plastic paths through one logical retrieval contract. Compare one-mechanism and hybrid alternatives with ablations on held-out recall, updates, contradiction, restart, resource use and latency; retain an explicit no-answer case and provenance. A physically unified learned state alone must not replace the auditable store until it demonstrably preserves inspectable facts, correction/supersession, retirement and safety. Versioned migration must have a reversible backup and compatibility gates. This is a research goal, not a predetermined winner.

For CPU language, evaluate Muir/C4 scratch training and separately compare pretrained PyTorch artifacts with verified licenses, provenance, tokenizer and CPU support; no GGUF or remote inference required. Estimate resources with a measured pilot on the user's hardware, targeting 16 GB RAM. A smoke-trained checkpoint is not a usable model and must not be silently promoted. The current native sampler is not yet integrated with memory chat.

User-selected web policy: automatic memory saving when autonomous research is explicitly enabled, retaining URL, timestamp and unverified status. Security and evidence controls live in `autonomous-web-research`; review mode remains available. Offline and existing profile defaults must not silently enable network or change behavior.

Gate progress on deterministic contract parity across surfaces, provenance and no-answer behavior, correctness tests, CPU measurement and rollback validation. Preserve legacy interfaces until migration is verified.
