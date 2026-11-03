# Spec Delta: surfaces/companion-app

## Purpose

Defines the complete companion pet interactive surface providing honest runtime state disclosure, pet interactions, clean conversational memory, and integrated web research tools.

## ADDED Requirements

### Requirement: Pet-centric interactive surface and honest state disclosure
The companion surface SHALL present the user with an interactive AI pet experience whose state attributes (energy, mood, active facts count, memory norm) derive directly from real measured engine metrics. It SHALL disclose anchor mode, generation backend, and model status honestly without fabricating artificial intelligence metrics or synthetic tokens.

#### Scenario: Real measured pet attributes
- **WHEN** the user inspects the pet dashboard at `/api/v1/companion/state`
- **THEN** every pet signal (energy, mood, memory norm) SHALL be computed deterministically from real engine counters
- **THEN** no metric SHALL be fabricated or mocked when real signals are available

#### Scenario: Clean initial pet state
- **WHEN** the companion is launched with the default `companion-pet` profile
- **THEN** it SHALL start with zero initial domain facts and zero remote sensing pre-seeding
- **THEN** its memory store SHALL be clean and ready to absorb personal conversation from the user

### Requirement: Integrated web research tool for pet growth
The companion surface SHALL provide an integrated web research tool allowing the pet to search the web, learn facts from reputable online sources, and cite its findings back to the user. The tool SHALL be accessible both via chat slash commands (`/search <query>`, `/research <topic>`) and through a dedicated UI action button in the chat interface.

#### Scenario: Web research triggered from companion UI
- **WHEN** the user submits a topic with the "Search Web" action in the companion interface
- **THEN** the system SHALL invoke the configured search learner with query rate-limiting
- **THEN** it SHALL extract clean declarative propositions from retrieved web content
- **THEN** it SHALL absorb new facts into the pet's associative memory and report the number of learned facts and source citations

#### Scenario: Automatic fallback search for unanswerable pet queries
- **WHEN** the user asks the pet a factual question about a topic absent from local memory
- **THEN** if web research is enabled, the pet SHALL trigger an autonomous search query
- **THEN** it SHALL incorporate the newly learned facts into its generative response

### Requirement: Conversational chat memory without domain pollution
The companion SHALL automatically absorb user statements from regular conversation into its plastic memory without requiring explicit teaching prefixes. It SHALL NOT contain hardcoded pre-trained remote sensing facts in its active memory space.

#### Scenario: Chat turn natural absorption
- **WHEN** the user tells the pet personal details during casual chat (e.g. "My name is Alex and I have two cats.")
- **THEN** the companion engine SHALL extract declarative clauses and update its associative memory
- **THEN** in subsequent turns, the pet SHALL accurately recall those personal facts when queried
