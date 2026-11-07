# Research note: one RecaMem experience, multiple specialized mechanisms

Status: recommendation for discussion, not an approved migration or measured
improvement. Reviewed 2026-11-07.

## Current repository

- `recamem/engine.py` exposes teach, recall, chat, research and memory injection;
  `recamem/runner.py` already wraps a subset in `run_once`.
- `recamem/retrieval.py` implements a two-channel result with episodic and plastic
  features; `recamem/episodic.py` and `recamem/hybrid_retrieval.py` provide
  storage, lexical/dense retrieval and indexes. `docs/recall.md` describes score
  provenance and the distinction between plastic-only and verbatim facts.
- `src/model/cores/registry.py` provides research sequence-model candidates.
  C4 remains the selected provisional core; C7 Muir is an experimental sequential
  CPU reference, not a validated replacement. The native checkpoint sampler is
  separate from companion chat.
- CLI, MCP, REST and the Svelte companion are different entry points, not all
  equivalent capabilities. Research is opt-in and already lives in
  `recamem/search.py` and `recamem/web_teacher.py`.

## Recommendation

One public `AgentSession` / `handle_turn` contract, one canonical coordinator
and one structured event/result model. CLI, REST, MCP and Svelte become thin
adapters. The coordinator classifies intent, searches memory, optionally performs
bounded research, calls an explicitly selected generator if available, and emits
an answer with evidence, requested/executed route and explicit errors. Teach,
correct and consolidate remain explicit auditable operations. An offline mode
never searches. Use existing `run_once` as the starting point rather than assume
all paths are already wired identically.

Unify the **interface and policy**, not the actual memory mechanisms: retain
SQLite records and provenance as the source of truth, sparse and dense indexes
as rebuildable retrieval views, plastic state as an associative signal, and a
Muir/C4 state as sequence context where appropriate. Combine ranked evidence at
one boundary; do not average incompatible scores blindly or claim Muir state is
an inspectable fact database. Keep the ability to disable or ablate each
mechanism. On CPU, measure exact-search cost and only add a reranker if accuracy
benefits justify latency. Native generation must validate checkpoint and
vocabulary and explicitly disclose when untrained or unavailable.

## Evidence and caveats

- A financial-document RAG benchmark found hybrid retrieval improved recall over
  either sparse or dense in its settings, and hybrid plus neural reranking did
  better there. This does not establish gains or acceptable CPU latency in
  RecaMem: https://arxiv.org/html/2604.01733v1
- MemX describes local-first keyword/vector RRF and low-confidence rejection,
  with provenance-oriented design: https://arxiv.org/html/2603.16171v1
- Another published practical benchmark found a reranker increased latency
  substantially without improving its hybrid ranking; workload matters:
  https://www.readysetcloud.io/blog/allen.helton/agent-memory-needs-more-than-vector-search/
- OWASP recommends untrusted-content separation, deterministic tool-boundary
  validation and least privilege; prompt labeling alone is not a security
  boundary: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html

## Rollout after approval

1. Inventory actual CLI/API/MCP/Svelte paths and profile/checkpoint compatibility;
   define versioned request, result and error contracts, with contract parity tests.
2. Wrap existing teach/recall/research/generation paths in a coordinator without
   changing defaults, storing provenance or silently enabling network access.
3. Add native CPU model integration only once its tokenizer, quality gates and
   checkpoint/profile lifecycle are supported. Continue returning explicit
   unavailability if no adequate model is present.
4. Compare current and unified paths on held-out recall, corrections, provenance,
   abstention, latency/RAM and restart persistence; include ablations and baseline
   parity. Stage migration, provide rollback, retain old paths until verified.

Remaining implementation decisions include exact resource limits, approved data sources and API compatibility. The user decisions below supersede the earlier scope questions.

## User decisions

The user selected unification of the experience followed by investigation of
internal memory consolidation; exploration of both native scratch and compatible
pretrained PyTorch CPU models; automatic storage of attributed research findings
when browsing is enabled; and an approximately 16 GB RAM target. Automatic
storage increases misinformation and memory-poisoning risk: retain unverified
status, provenance, controls and conflict handling. These choices do not bypass
the browser change's outstanding network isolation and safety gates.
