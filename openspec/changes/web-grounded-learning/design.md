# Design

## Context

See `proposal.md` for motivation. The constraints that shape this design:

- The memory engine (`src/model/droid.py`, 1758 lines) already owns retrieval,
  distillation-assisted teaching, plastic updates, and the answer-path vocabulary.
  It has no network code and no external dependency beyond `torch` and
  `onnxruntime`.
- The project deliberately runs on a CPU-only machine with no GPU and, for many
  users, no paid API key. `src/model/teacher.py` already speaks the
  OpenAI-compatible protocol with a *local* endpoint (Ollama, LM Studio) and
  degrades to a local fallback when the endpoint is unreachable.
- `src/model/knowledge_store.py` stores facts in SQLite with FTS5 + dense MVM + RRF.
  The `facts` table already has `source`, `timestamp`, `step`, `novelty` columns and
  a `_migrate_schema` path.
- The MCP server capability is documented as stdlib-only, and the project style is
  stdlib-only HTTP (`urllib` in `teacher.py`, stdlib `sqlite3` in
  `knowledge_store.py`).
- `VISION.md` frames the whole project around online learning: the gap is not
  storage, it is *acquisition*.

## Goals / Non-Goals

**Goals:**

- A droid can acquire a fact it was never told, from an external source, at
  inference time, and keep it.
- Every acquired fact is attributable to a URL, a timestamp, and a query.
- The system can say "I looked it up" and "I did not know" without ambiguity.
- Zero new third-party dependencies; every network call is bounded.

**Non-Goals:**

- Agentic browsing (following links, multi-hop planning). One query, one bounded
  fetch set.
- Any change to the plastic update mathematics, the recall scorer, or the fused
  byte-RTU path.
- A hosted service. The feature is a local client of public endpoints.
- Replacing the teacher endpoint. The web path reuses it when present.

## Decisions

### D1. A provider interface, not a hardcoded client

`SearchProvider` is a protocol with one method, `search(query, limit) -> list[SearchResult]`,
where `SearchResult` carries `title`, `url`, `snippet`. Three implementations ship:
a key-free public provider, a key-configurable API provider, and a null provider
that reports "not configured".

**Alternatives considered.** A single hardcoded HTTP call was rejected: the
key-free endpoint is unofficial and can change shape, and a hardcoded call makes
that breakage a runtime surprise rather than a substitutable implementation.
Requiring an API key was rejected for the same reason the project exists: the
target user has no GPU and often no key.

### D2. The provider returns URLs; a separate fetcher retrieves text

The provider never fetches pages. `WebFetcher` does, and owns every safety limit:
scheme allowlist, host resolution check, redirect handling, byte cap, timeout.
Each provider therefore stays a thin, trivially testable function, and the safety
limits are implemented once.

**Alternatives considered.** Letting each provider fetch (some APIs return page
content directly) was rejected: the SSRF guard, size cap, and timeout would then
be reimplemented per provider and could diverge.

### D3. HTML to text with the stdlib `html.parser`

A small `HTMLParser` subclass drops `script`, `style`, `nav`, `header`, `footer`,
`aside`, and `form`; keeps text from `p`, `li`, `h1`-`h6`, `td`, `pre`; unescapes
entities; collapses whitespace; and stops at a per-page character budget.

**Alternatives considered.** `beautifulsoup4` was rejected: it adds a dependency to
a stdlib-only project, and the extraction needed here is shallow. A
readability-style main-content extractor was rejected as disproportionate for
input that is about to be reduced to a handful of propositions.

### D4. Distillation reuses the existing teacher, with a deterministic local fallback

When a teacher endpoint is configured, `OpenAITeacher.distill_propositions` is used
on the extracted text. When it is not, the web path falls back to
`src/model/propositions.extract_propositions` (sentence split plus length and
conversational-marker filtering) rather than to the teacher's own fallback, which
returns the whole page as a single proposition.

**Alternatives considered.** Making an LLM endpoint mandatory was rejected: it
would put the feature behind a GPU or a key, exactly the dependency the project
exists to avoid. Inventing a third summarizer was rejected: `propositions.py` and
`teacher.py` already cover both ends of this tradeoff.

### D5. Provenance gets its own columns, and `source` stays a closed vocabulary

The `facts` table gains `source_url TEXT` and `source_query TEXT` through the
existing `_migrate_schema` path. `source` keeps its meaning — the *origin class* —
and becomes a closed set: `user_text`, `chat`, `continuous_stream`, `web`.

**Alternatives considered.** Packing provenance into `source` as JSON was rejected:
it destroys the closed origin vocabulary that filtering and the fact browser rely
on. Adding a separate `provenance` table was rejected as one join too many for a
four-field payload.

### D6. The answer-path vocabulary gains `researched`

`ANSWER_PATH_RESEARCHED = "researched"` joins `recall`, `fused`, and `llm`.
`legacy_answer_path` maps it to `extractive`, because a researched answer is
still extractive with respect to the fact store.

**Alternatives considered.** A boolean "was researched" flag beside the badge was
rejected: the badge is the UI's contract, and a flag next to a badge is invisible.
Collapsing "researched" into `recall` was rejected for the same reason the feature
exists: a user must be able to tell "I knew this" from "I just looked it up".

### D7. The loop lives in its own module; the engine stays the orchestrator

`SearchLearner` is constructed with the engine it serves and holds the decision
policy, the provider, the fetcher, and the per-topic suppression state. The engine
holds an optional `search_learner` and calls into it.

**Alternatives considered.** Implementing `research()` directly on `DroidEngine`
was rejected: it would mix network policy into a module that already owns
retrieval, plastic updates, versioning, merging, and profile persistence, and it
would make the decision policy testable only through the whole engine.
`DroidEngine` is already 1758 lines; this change must not grow it further.

### D8. The search decision reuses the existing retrieval confidence

The learner asks the engine to recall the topic and compares the top hit's
confidence against the profile's `search_trigger_threshold`. Above it, no search
happens.

**Alternatives considered.** A separate "do I know this" model was rejected: the
engine already computes a documented, weighted retrieval confidence
(`droid.py:245`, `RECALL_SCORE_WEIGHTS`), and a second notion of confidence would
drift from the one the console shows.

### D9. Every search is bounded, and the bounds are configuration, not constants

Per profile: `max_pages_per_search` (default 3), `max_queries_per_search`
(default 2), `timeout_seconds` (default 10), `max_bytes_per_page` (default
256 KiB), `max_searches_per_minute` (default 4), `suppression_seconds` (default
300), `max_searches_per_session` (default 20). A topic is normalized (lowercased,
whitespace collapsed) before it is used as a suppression key.

**Alternatives considered.** Bounding only by the model quota was rejected: a
per-minute and per-session bound is what stops an accidental crawler even when the
model is curious.

### D10. Fetching is treated as a server-side request forgery surface

The fetcher allows only `http` and `https`; resolves the host and refuses
loopback, private, link-local, reserved, and multicast addresses; follows at most
two redirects and re-checks every hop; sends a descriptive user agent; reads at
most `max_bytes_per_page`; and applies the timeout to connect and read.

**Alternatives considered.** A denylist of known-bad hosts was rejected: it is
trivially bypassed by DNS rebinding, and an allowlist of trusted sources would
defeat the purpose of open-web research.

### D11. Retrieved text is untrusted data, never instructions

Retrieved text is passed to the teacher inside a quoted data block with an
explicit instruction that it is source material to summarize, not instructions to
follow. Acquired facts are stored with origin `web`, are shown with their source
in the fact browser, and can be reverted like any other fact.

**Alternatives considered.** Stripping the text of anything that looks like an
instruction was rejected: it is an unwinnable classification problem, and the
provenance plus review path is the honest mitigation. Treating web content as
trusted was rejected outright.

### D12. The feature is opt-in and off by default

A profile enables it with a `web_research` block in `config.json`. Without that
block, the engine behaves exactly as it does today: no network call, no new
import-time requirement, no changed answer.

**Alternatives considered.** Enabling it by default was rejected: a profile that
starts making network calls on its own changes the trust and cost model of every
existing install.

### D13. Tests are hermetic; the network is only touched by an explicit script

`SearchProvider` and the fetch boundary are injected, so the whole loop is
exercised with fake providers. The real fetcher is tested against a
`http.server` bound to loopback in a thread — which means the SSRF guard needs an
explicit test-only allowance for loopback, scoped to the test fixture, not a
production bypass. A live smoke script is a separate entry point gated behind an
environment variable.

**Alternatives considered.** Mocking at the `urllib` level was rejected: it
tests the mock, not the fetcher. Allowing loopback in the fetcher
unconditionally was rejected: it would ship the SSRF hole to make a test
convenient.

## Risks / Trade-offs

- **[The key-free provider breaks or rate-limits]** → The provider is an interface;
  a second provider with a key is one class. Failures are structured, never
  exceptions escaping into chat, and the failure path is a no-information answer
  rather than a wrong answer.
- **[A prompt injection in a retrieved page produces a false fact]** → The text is
  framed as data, the origin is recorded as `web`, the fact is reviewable and
  revertible in the console, and the confidence of a web-sourced answer is
  reported lower than a taught one. Residual risk accepted and documented.
- **[Search makes the chat path slow or flaky]** → The whole flow is bounded by
  timeouts and a page cap, and the no-information fallback is returned when the
  budget is exhausted.
- **[Local fallback produces low-quality propositions]** → Accepted. The
  deterministic fallback is sentence-level, so the facts are verbose rather than
  wrong, and the endpoint path is better when it is available.
- **[Provenance columns change the knowledge-store schema]** → The migration path
  already exists and is additive; existing databases keep working, and a fact
  inserted before this change simply has no URL.
- **[The `researched` answer path is a breaking change for external callers of
  `legacy_answer_path`** → `legacy_answer_path` still returns only
  `extractive`/`generative`, so legacy callers are unaffected. Only the new value
  appears on the modern path, and the CLI and MCP surfaces are extended rather than
  changed.
- **[The MCP tool/spec parity guard reads this change's delta before the code
  lands]** → The guard now enforces a delta only once every task in its change is
  checked, which is when the code has landed; before that, the delta is a plan and
  does not fail the build. The guard is strict again the moment this change's
  tasks are complete, so the `research_topic` tool must exist by then.

## Migration Plan

1. Ship the modules with the feature disabled by default. Existing profiles are
   unaffected: no config key, no behavior change, no schema requirement.
2. The knowledge store migration adds two nullable columns. Existing databases are
   migrated in place on open; facts without a URL stay valid.
3. Rollback is configuration: remove or disable the `web_research` block. The
   learned facts stay in the store and remain recallable, marked with origin `web`.
4. No weights, checkpoints, or profile directories change.

## Open Questions

- Should web-sourced facts carry a different retrieval weight than user-taught
  facts? Deferred: this change records origin and shows provenance; re-weighting
  touches the recall scorer, which is a separate capability with its own
  requirements.
- Should the key-free provider be DuckDuckGo's HTML endpoint or a self-hosted
  SearXNG instance? Deferred: the provider is pluggable and the configuration
  selects one, so the default can change without a spec change.
