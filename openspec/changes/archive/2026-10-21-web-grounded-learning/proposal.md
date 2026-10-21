# Proposal

## Why

The Droid only learns from what a human hands it: `teach()`, a pasted paragraph, a
continuous stream. Nothing in the repository performs an external lookup, so the
"online learning" thesis in `VISION.md` stops at the storage layer. The memory
machinery it needs already exists — hybrid retrieval (SQLite FTS5 + dense MVM fused
with RRF), proposition distillation through an OpenAI-compatible teacher, and
surprise-gated plastic updates — but it has no *source*. The missing link is a
pluggable search provider plus the policy that decides when to search, what to keep,
and where it came from.

The capability is also what makes the system honest at inference time. Today a
retrieval that finds nothing silently returns nothing, which a user cannot tell apart
from "the model knows nothing". A search-backed path turns "I don't know" into a
verifiable, attributed, and reproducible action.

## What Changes

- Add a pluggable search provider and a stdlib-only fetcher (`src/model/web_teacher.py`):
  issue a web query, fetch the result pages, and extract readable text without a
  third-party HTML, scraping, or search-SDK dependency.
- Add a `SearchLearner` (`src/model/search_learn.py`) that closes the loop: judge
  whether the question is answerable from memory, search when it is not, distill the
  retrieved pages into standalone propositions, teach them into the knowledge store
  and the plastic memory, and record provenance for every fact learned this way.
- Record provenance on web-sourced facts: source URL, retrieval timestamp, and the
  query that found them. Recall output distinguishes web-sourced facts from
  user-taught ones.
- Add a distinct answer path for researched answers, so an answer that came from the
  web is visibly different from one that came out of memory.
- Expose the loop headlessly: a `research` CLI subcommand and a `research_topic` MCP
  tool, so the capability is usable without the Streamlit UI.

## Capabilities

### New Capabilities

- `adaptation/web-grounded-learning`: acquiring knowledge from an external web source
  at inference time — the search decision, the bounded search execution, provenance of
  what was learned, and the honest failure path.

### Modified Capabilities

- `cli`: the documented subcommand list gains `research`, with the same `key=value`
  output contract the `eval` subcommand already uses.
- `mcp-server`: the documented tool list gains `research_topic`.
- `model/dual-mode-generation`: the observable answer-path vocabulary gains
  `researched`, so an answer grounded in externally acquired facts is distinguishable
  from one that came out of memory alone.
- `droid/memory-console`: the answer-path badge gains `researched`, and the console
  lists the sources a researched answer was grounded in.

## Impact

- **Code areas:** new `src/model/web_teacher.py` (provider + fetcher); new
  `src/model/search_learn.py` (decision policy + loop); `src/model/droid.py` (answer
  path, provenance plumbing, profile configuration); `src/model/teacher.py` (reuse the
  existing distillation prompt for web text); `src/model/knowledge_store.py` (source
  provenance is already a column); `src/cli.py`; `src/mcp_server.py`.
- **Tests:** new `tests/test_web_teacher.py` and `tests/test_search_learn.py`. Every
  test is hermetic — the provider is injected, and no test touches the network.
- **Dependencies:** none. Stdlib `urllib` and `html.parser` only, matching the
  stdlib-only constraint already stated in the `mcp-server` capability and the style
  of `src/model/teacher.py`.
- **Compatibility:** the existing `recall` / `fused` / `llm` answer paths are
  unchanged. Research is opt-in — a profile without a configured provider behaves
  exactly as it does today.
- **Non-goals:** no web UI, no agentic multi-step browsing, no autonomous background
  crawling, no paid search API dependency, no change to the plastic update maths.

## Risks

- A key-free public search endpoint is unofficial and can change shape or rate-limit.
  The provider is an interface with one key-free implementation and one
  key-configurable implementation, and the failure path is a structured error rather
  than an exception escaping into chat.
- Fetching arbitrary URLs is a server-side request forgery surface. The fetcher
  restricts the scheme to `http`/`https`, rejects loopback, link-local, and private
  address ranges, caps the response size, and applies a per-request timeout.
- Web text is untrusted input. Retrieved content is treated as data to be distilled,
  never as instructions, and propositions are filtered for length and completeness
  before they are taught.
- Searching on every low-confidence query turns a local system into a network client
  with a latency and a quota cost. Search is rate-limited, deduplicated per topic
  within a suppression window, and configurable to off.
