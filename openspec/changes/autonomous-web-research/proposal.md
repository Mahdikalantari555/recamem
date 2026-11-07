# Proposal: Bounded autonomous web research

## Why
The companion needs to research public web sources autonomously, including sites that need JavaScript rendering, while remaining local-first and usable on CPU. The existing opt-in `recamem.search.SearchLearner` and `recamem.web_teacher` already provide search, static fetching, budgets and source provenance; this change extends them rather than creating a second learning engine. The current fetcher's hostname check precedes a separate HTTP connection, so it is not by itself proof of DNS-rebinding-safe network isolation. A small smoke-trained model cannot be assumed capable of autonomous planning.

## What Changes
- Add a bounded, inspectable research session with search, public read-only navigation, extraction, citations, stopping and cancellation.
- Reuse the existing static HTTP fetch path after hardening; evaluate a pinned, patched optional local Crawl4AI/Playwright adapter for JS pages, not a required dependency or external service.
- Route every network operation through enforced public-address egress policy, including browser subresources and redirects; fail closed when this cannot be guaranteed.
- Present sourced evidence and support automatic durable teaching when the user explicitly enables the new research mode, with provenance and unverified status. A review-before-saving option remains available. Existing legacy profiles remain unchanged unless explicitly migrated.
- Surface budgets, progress, sources, errors, save policy and optional review in the Svelte companion and API.

## Capabilities
- `adaptation/autonomous-web-research`: bounded, auditable and safe public web research.

## Non-goals
No CAPTCHA bypass, stealth, paywall evasion, authentication, form submission, arbitrary code execution, unrestricted browsing, or autonomous actions that change external websites; opt-in memory writes are permitted. No guarantee that a CPU-only scratch model can plan accurately. No change to unrelated retrieval/channel architecture before user decisions.

## Impact
`recamem/search.py`, `recamem/web_teacher.py`, engine research integration, `recamem/api.py`, Svelte companion and tests; optional Chromium and crawling dependencies only when configured. This is a proposal, not a claim of implementation or a security guarantee.

## Continuous learning alignment

Approved web findings may eventually enter the incremental learning intake with attribution and unverified status; background learning alone never enables web access. See `autonomous-incremental-learning`.
