# Tasks: Complete Companion Pet Design & Clean Web-Grounded Pet Profile

## 1. Clean Pet Profile & Zero Domain Contamination

- [x] 1.1 Configure `recamem/cli.py` and `recamem/api.py` default profile to `companion-pet` with fresh initial memory state and verify via `recamem companion --dry-run`.
- [x] 1.2 Implement automatic `research` block initialization for `companion-pet` (DuckDuckGo provider, bounded rate limits) in `recamem/manager.py` and verify in unit test.
- [x] 1.3 Add test `tests/test_companion_clean_pet.py` asserting that a fresh companion profile contains 0 stored facts, 0 remote sensing propositions, and an active web search learner.

## 2. Interactive Web Search Tool & Svelte UI Integration

- [x] 2.1 Add "Search Web" action button in `companion/src/components/ChatPanel.svelte` that submits `{"action": "search", "text": prompt}` to `/api/v1/companion/act`.
- [x] 2.2 Add research evidence rendering in `ChatPanel.svelte` displaying learned facts, source URLs, and search query status.
- [x] 2.3 Rebuild and verify companion frontend assets (`companion/dist`) using `pnpm --prefix companion run build`.

## 3. End-to-End Pet Verification & Gates

- [x] 3.1 Verify conversational memory in `companion-pet` absorbs regular user chat and answers questions without pre-loaded domain data in `tests/test_companion_clean_pet.py`.
- [x] 3.2 Run `PYTHONPATH=. python scripts/offline_gate.py` and ensure 100% pass across all 7 checks.
- [x] 3.3 Validate OpenSpec proposal with `PYTHONPATH=. python scripts/validate_openspec.py`.
