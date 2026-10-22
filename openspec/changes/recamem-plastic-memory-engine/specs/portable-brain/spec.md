# Spec Delta

## ADDED Requirements

### Requirement: Format version in the manifest
The `.droid` archive manifest SHALL record a format version. Import SHALL accept archives of every format version the running code supports, and SHALL reject an archive declaring a newer version with a named error and without extracting any member.

#### Scenario: Older archive imports
- **WHEN** an archive with an older supported format version is imported
- **THEN** the import SHALL succeed and the restored profile SHALL keep every stored value

#### Scenario: Newer archive is refused before extraction
- **WHEN** an archive declares a format version newer than the running code supports
- **THEN** the import SHALL fail with a named error naming both versions
- **THEN** no member SHALL be extracted to the target profile directory
