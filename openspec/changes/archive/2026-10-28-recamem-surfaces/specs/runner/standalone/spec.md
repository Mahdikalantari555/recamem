# Spec Delta — Standalone runner

## ADDED Requirements

### Requirement: Headless runner
The system SHALL provide a headless runner that operates without Streamlit and
depends only on core dependencies. It SHALL accept a single prompt via `--prompt`
and print one JSON object per invocation to stdout.

#### Scenario: Single prompt
- **WHEN** user runs `recamem run --prompt "hello" --name demo`
- **THEN** the runner SHALL load the named profile and selected core
- **THEN** stdout SHALL contain a single JSON object with keys `prompt`, `response`, and `memory`
- **THEN** the process SHALL exit 0 on success

#### Scenario: Stdin loop
- **WHEN** user runs `recamem run --name demo` with lines on stdin
- **THEN** the runner SHALL process each line and emit one JSON object per line

#### Scenario: No Streamlit dependency
- **WHEN** Streamlit is not installed
- **THEN** `recamem run` SHALL still function
