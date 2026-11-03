# Proposal: Complete Companion Pet Design & Clean Web-Grounded Pet Profile

## Why

The RecaMem companion is designed to serve as a personal, interactive AI pet with persistent associative memory. Currently, default test fixtures and benchmark examples pollute the companion's context with remote sensing facts, while web search capabilities lack native companion UI triggers. This change establishes a clean, dedicated `companion-pet` profile free of remote sensing bias, configures autonomous web search tooling directly for the pet, and completes the companion UI for seamless conversation and web growth.

## What Changes

- **Dedicated Clean Pet Profile**: Introduce `companion-pet` profile as the default companion target, initialized with zero domain facts, clean associative memory, and custom pet identity metadata.
- **Remove Domain Bias from Companion**: Ensure companion initialization never pre-populates or assumes remote sensing facts; all knowledge is acquired dynamically from user chat or web search.
- **Interactive Web Search Tool for Pet**: Expose native web search (`/search` and `/research` via DuckDuckGo HTML) directly in the companion API and Svelte UI with dedicated search trigger and source evidence rendering.
- **Companion UI Completion**: Add a "Search Web" action button and research evidence card in `companion/src/components/ChatPanel.svelte`, linking real-time research results to pet state and conversational memory.
- **CLI Default Alignment**: Update `recamem companion` CLI default to use `--name companion-pet` with automatic web research enablement.

## Capabilities

### New Capabilities
- `surfaces/companion-app`: Complete companion pet surface with one-command launch, interactive pet feedback actions, clean conversational memory, and direct web search tooling.

### Modified Capabilities
- `droid/profile-management`: Clean pet companion profile initialization without domain fact pollution, and default web research configuration.

## Impact

- `recamem/cli.py`: default companion profile updated from `droid-alpha` to `companion-pet`.
- `recamem/companion.py`: companion action dispatch supports `search` and `research` with structured query payloads and state changes.
- `recamem/api.py`: companion runtime initializes with clean pet profile and web research provider enabled.
- `companion/src/components/ChatPanel.svelte`: adds Web Search button, search query feedback, and web source citation cards.
- Test suites: adds verification for clean companion profile, web search action dispatch, and zero default domain facts.
