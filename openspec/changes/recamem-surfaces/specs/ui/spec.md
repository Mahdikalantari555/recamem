# Spec Delta — WebUI

## MODIFIED Requirements

### Requirement: Rebranded WebUI
The WebUI SHALL use RecaMem branding (not "Droid"). It SHALL provide tabs:
Chat, Teach, Recall, Memory health, Bench. The sidebar SHALL display anchor mode
(real/mock), selected core id, and memory stats.

#### Scenario: Reset Memory works
- **WHEN** user clicks "Reset Memory"
- **THEN** the app SHALL call `engine.clear_memory()`
- **THEN** no `AttributeError` SHALL be raised

#### Scenario: Anchor mode visible
- **WHEN** the WebUI loads
- **THEN** the sidebar SHALL show the anchor mode (real or mock)

## ADDED Requirements

### Requirement: Optional UI dependency
The WebUI SHALL be installable via the `ui` extra. Core functionality SHALL work
without it.

#### Scenario: Core-only install
- **WHEN** `recamem` is installed without `[ui]`
- **THEN** `recamem run` and `recamem doctor` SHALL work
- **THEN** `recamem ui` SHALL report that Streamlit is required
