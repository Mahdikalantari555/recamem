# cloud-huggingface Specification (Delta)

## ADDED Requirements

### Requirement: Checkpoint provenance is machine-readable

Every checkpoint directory exposed to the companion SHALL contain a
`manifest.json` with sha256 per file plus config, seed, dataset identity, commit,
command and measured gate results. A checkpoint without these fields SHALL be
reported as ineligible and SHALL NOT be presented as the best model.

#### Scenario: Tampered manifest

- **WHEN** a file's sha256 does not match the manifest
- **THEN** selection SHALL abort and SHALL leave the loaded profile unchanged

### Requirement: Notebook template is valid JSON

The Colab notebook template SHALL parse as valid `nbformat` JSON and SHALL
contain the install, configure, train and push cells.

#### Scenario: Invalid notebook

- **WHEN** the notebook contains trailing text after the JSON object
- **THEN** the notebook validity test SHALL fail naming the file

### Requirement: Companion keeps tokens server-side

The companion SHALL NOT accept an `HF_TOKEN` in any browser request, SHALL NOT
store it in job parameters, logs or evidence, and SHALL perform push/pull only
in the Python process using the environment or a prior CLI login.

#### Scenario: Token sent from the browser

- **WHEN** a request payload contains a token field
- **THEN** the API SHALL reject the request
- **THEN** no token SHALL appear in job params, logs or evidence records
