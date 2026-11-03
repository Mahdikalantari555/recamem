# Spec Delta: droid/profile-management

## ADDED Requirements

### Requirement: Clean companion pet profile template
The profile manager SHALL provide a pre-configured, clean profile template for the companion (`companion-pet`). This profile SHALL be configured with the dynamic `delta_rtu` plastic adapter, an active web search configuration with rate bounds (DuckDuckGo HTML provider, maximum 8 queries per session), and an empty knowledge database containing zero remote sensing or synthetic aperture radar facts.

#### Scenario: Creating a default companion-pet profile
- **WHEN** the companion launcher or API requests profile `companion-pet` and no saved directory exists
- **THEN** the manager SHALL initialize a fresh `companion-pet` profile with zero initial facts
- **THEN** it SHALL configure `research.enabled = True` with provider `duckduckgo`
- **THEN** it SHALL NOT import any remote sensing fixtures or satellite benchmark datasets

#### Scenario: Isolation from domain benchmark profiles
- **WHEN** benchmark runs populate remote sensing or domain-specific profiles
- **THEN** the `companion-pet` profile artifacts, database, and plastic weights SHALL remain completely isolated and unpolluted
