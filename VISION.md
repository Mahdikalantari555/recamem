# RecaMem

RecaMem is a local-first plastic memory engine for foundation models.

Unlike traditional memory systems that merely store and retrieve information through embeddings and RAG, RecaMem is designed to accumulate experience. At its core, a Recurrent Trace Unit (RTU) continuously updates a persistent memory state from interactions, feedback, and learned knowledge, allowing future reasoning to be influenced by past experience rather than simple text retrieval. Foundation models remain frozen and provide language capabilities, while RecaMem supplies adaptive, long-term memory, online learning, memory consolidation, forgetting mechanisms, and trace-based recall. The result is a lightweight cognitive layer that enables AI systems to learn continuously without retraining, preserve specialized knowledge, and evolve their behavior over time.

Every claim is measured, not asserted. See `plans/09_recamem_plan.md` and `reports/recamem-core-selection.md`.
