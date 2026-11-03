# Design: Complete Companion Pet Architecture & Clean Web-Grounded Pet Profile

## Context

The companion is an interactive local agent with real measured state (energy, mood, active facts) and fast-weight associative memory. Previously, the default profile name `droid-alpha` shared test fixtures that occasionally inherited remote sensing facts from e2e test scripts, and web search capabilities lacked dedicated UI affordances in the Svelte frontend. This design defines a clean isolation architecture for the companion pet and a streamlined web search tool integration.

## Goals / Non-Goals

**Goals:**
- Provide a dedicated, factory-fresh `companion-pet` profile initialized with zero facts and no domain contamination.
- Integrate DuckDuckGo web search natively into the companion API and Svelte UI with explicit user triggers and evidence citations.
- Preserve fast associative chat absorption so casual conversations naturally shape the pet's memory.
- Maintain full compatibility with all offline verification gates and `llama.cpp` GGUF local model execution.

**Non-Goals:**
- Removing remote sensing datasets from existing domain-specific benchmark test files (`test_remote_sensing_e2e.py`).
- Introducing third-party external search APIs requiring paid API keys (DuckDuckGo HTML remains keyless and local).
- Modifying the underlying Gated Delta Core or RTU architecture.

## Decisions

### 1. Dedicated `companion-pet` Default Profile
- **Decision**: In `recamem/cli.py` and `recamem/api.py`, change default profile from `droid-alpha` to `companion-pet`.
- **Rationale**: Isolates user pet interactions from generic test profiles. A fresh directory `~/.recamem/agents/companion-pet` is created without any pre-loaded facts.
- **Alternative considered**: Purging `droid-alpha` on startup. Rejected because users or tests might expect persistent state in `droid-alpha`.

### 2. Auto-Configured Web Research on Pet Profile
- **Decision**: When `companion-pet` is created, automatically configure:
  ```json
  "research": {
    "enabled": true,
    "provider": "duckduckgo",
    "max_queries_per_session": 8,
    "max_pages_per_search": 2,
    "result_limit": 3
  }
  ```
- **Rationale**: Gives the pet immediate ability to search the web without manual profile editing or API keys, while strictly enforcing query budgets.
- **Alternative considered**: Requiring users to issue `/configure_research` manually. Rejected because the pet should feel capable out of the box.

### 3. Companion Action Dispatch: Native "search" and "research"
- **Decision**: Add a dedicated "Search Web" button in `companion/src/components/ChatPanel.svelte` that dispatches `POST /api/v1/companion/act` with `{"action": "search", "text": query}`.
- **Rationale**: Directly maps the user's intent to the engine's `SearchLearner`, displaying learned propositions and source URLs in the "What changed" and evidence log.
- **Alternative considered**: Forcing users to type `/search` in the regular chat input. Adding a dedicated button provides better discoverability and visual feedback.

### 4. Zero Domain Seeding Guarantee
- **Decision**: Validate on pet initialization that the episodic store and semantic vector store start at exactly 0 rows. Verify in automated tests that no satellite, radar, or remote sensing propositions exist.
- **Rationale**: Guarantees the pet belongs entirely to the user and their personal conversational history.

## Risks / Trade-offs

- **[Risk] DuckDuckGo HTML rate limiting or markup changes** → Mitigation: `SearchLearner` uses cached responses, exponential backoff, and a clean fallback to existing memory if search is temporarily throttled.
- **[Risk] Svelte frontend build synchronization** → Mitigation: `companion/dist` is updated and validated against `scripts/offline_gate.py` to prevent packaging drift.
