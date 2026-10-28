# Spec Delta — CLI accessibility

## ADDED Requirements

### Requirement: run, ui, doctor subcommands
The `recamem` CLI SHALL provide `run`, `ui`, and `doctor` subcommands.

#### Scenario: doctor
- **WHEN** user runs `recamem doctor`
- **THEN** the CLI SHALL print anchor mode, model path, load time, and memory stats
- **THEN** exit 0 only if all required items are ready

#### Scenario: ui
- **WHEN** user runs `recamem ui`
- **THEN** the CLI SHALL launch the WebUI (requires `[ui]` extra)

#### Scenario: run
- **WHEN** user runs `recamem run --prompt "..." --name <profile>`
- **THEN** the CLI SHALL execute the runner and print JSON
