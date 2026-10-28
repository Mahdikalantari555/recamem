# Glossary — RecaMem

Terms are domain-level only; no implementation details.

- **Agent**: a named RecaMem profile (formerly "droid") with its own memory state and knowledge store.
- **Fact**: a single proposition distilled from input, stored in episodic memory.
- **Memory (plastic)**: persistent adaptive state updated by the RTU from interactions and feedback.
- **Trace**: the recurrent state carrying past-experience influence into future reasoning.
- **Recall**: retrieving relevant facts/knowledge for a query (two-channel: plastic prior + episodic verbatim).
- **Consolidation (sleep)**: offline replay of experience into plastic state to strengthen retention.
- **Forgetting**: decay and eviction of weak, unreinforced memories (consolidate-then-forget).
- **Retired**: a cold-archived (not hard-deleted) memory entry, queryable with `--include-retired`.
- **Anchor**: the frozen embedding model (MiniLM INT8) providing feature space; mock is explicit opt-in.
- **Core**: the selected sequence-mixing backbone (C4 gated-delta matrix) loaded from `configs/recamem_selected_core.json`.
- **Engine**: orchestrator coupling anchor, plastic memory, episodic store, and backends (formerly `DroidEngine`).
- **Backend**: a frozen foundation model provider (Ollama / OpenAI-compatible / local GGUF / embedded RTU).
- **Teacher**: distills unstructured text into propositions via an OpenAI-compatible endpoint.
EOF
echo "created"