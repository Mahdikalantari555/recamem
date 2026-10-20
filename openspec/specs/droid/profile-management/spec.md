# droid/profile-management Specification

## Purpose
Enables creating, selecting, saving, and switching between named Droid instances, each maintaining independent persistent memory states, configs, and audit logs.

## Requirements

### Requirement: Named Droid Isolation
The system SHALL support creating and switching between multiple Droid instances stored under `droids/<droid-name>/`. Each profile SHALL record the plastic adapter type it was constructed with in `config.json`, and loading a profile SHALL read that recorded value rather than inferring the adapter from whatever class happens to be live. A profile without the field SHALL load with the current default adapter, preserving older profiles.

#### Scenario: Switching active Droid
- **WHEN** user selects a different Droid (e.g. `droid-remote-sensing`) from the profile selector
- **THEN** the active RTU memory state, logs, and metadata SHALL switch cleanly to the target profile without cross-contamination

#### Scenario: Memory round-trip persistence
- **WHEN** a Droid absorbs knowledge and is saved
- **THEN** re-loading that Droid SHALL restore its exact internal memory states and audit log history

#### Scenario: Adapter type survives a round trip
- **WHEN** a Droid constructed with a non-default plastic adapter is saved and loaded into a fresh engine
- **THEN** the loaded engine's adapter class SHALL match the saved `plastic_adapter` value
- **THEN** the loader SHALL read the field that the saver writes, and SHALL NOT read a field that no saver writes

#### Scenario: Legacy profile without the adapter field
- **WHEN** a profile's `config.json` has no `plastic_adapter` field
- **THEN** loading SHALL succeed with the default adapter and SHALL NOT raise
