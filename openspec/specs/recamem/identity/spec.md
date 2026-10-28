# recamem/identity Specification

## Purpose
Gives the project one name and one home for user data: the `recamem` package and command, profiles under a recamem home directory, and a migration path from the earlier `droids/` layout that copies rather than destroys.

## Requirements

### Requirement: Single command identity with a deprecation shim
The system SHALL provide the `recamem` command as the primary entry point. The earlier `python -m src.cli` entry point and the `tmt-droid` command SHALL keep working for one release and SHALL emit a deprecation notice on standard error that names the replacement.

#### Scenario: Old entry point still works
- **WHEN** a user runs `python -m src.cli chat --name droid-alpha`
- **THEN** the command SHALL behave as `recamem chat --name droid-alpha`
- **THEN** a deprecation notice naming `recamem` SHALL be printed to standard error

### Requirement: Profiles live under a recamem home
The system SHALL store agent profiles under `~/.recamem/agents/<name>/`. The location SHALL be overridable with the `RECAMEM_HOME` environment variable. A legacy profile under `droids/<name>/` SHALL remain readable in place.

#### Scenario: Default location
- **WHEN** a profile is created without a location override
- **THEN** it SHALL be created under `~/.recamem/agents/<name>/`

#### Scenario: Legacy profile is read in place
- **WHEN** a profile exists only under `droids/<name>/` and is loaded
- **THEN** it SHALL load with its recorded state unchanged

### Requirement: Migration copies and never deletes
The system SHALL provide `recamem migrate droids/` that copies each legacy profile into the recamem home, records a checksum of every copied file in a migration record, and leaves the source unchanged. Re-running the migration SHALL NOT duplicate or alter a profile that already migrated with matching checksums.

#### Scenario: Migration preserves the source
- **WHEN** `recamem migrate droids/` runs
- **THEN** every legacy profile SHALL be copied into the recamem home
- **THEN** the source directory SHALL be byte-identical after the run

#### Scenario: Migration is repeatable
- **WHEN** the migration is run a second time with no source changes
- **THEN** the report SHALL state that each profile was already migrated
- **THEN** no destination file SHALL change

### Requirement: Offline by default
The system SHALL NOT make network requests during `recamem chat`, `teach`, `ingest`, `recall`, `decide`, `doctor`, or `migrate`. Network use SHALL occur only in commands that were explicitly asked to fetch data, and `doctor` SHALL report which backends need the network.

#### Scenario: Doctor reports readiness without network
- **WHEN** a user runs `recamem doctor` with no network
- **THEN** the report SHALL list the anchor mode, the generation backend, and the storage path
- **THEN** each item that is unavailable SHALL state the missing piece and SHALL NOT be reported as ready
