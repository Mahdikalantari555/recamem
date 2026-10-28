# Spec Delta

## Purpose

Benchmarks for recamem against published long-conversation memory tasks (LoCoMo, LongMemEval-S, DMR), with a download-once cache, deterministic subsets, an offline fixture fallback, and reports that state whether each number came from real data.

## ADDED Requirements

### Requirement: Three benchmark runners with named metrics
The system SHALL provide runners for LoCoMo, LongMemEval-S and DMR. Each runner SHALL report the metric names defined by its benchmark and SHALL report the number of questions evaluated.

#### Scenario: Runner reports its metrics
- **WHEN** a runner completes on its dataset
- **THEN** the report SHALL contain the benchmark name, the metric names, and the number of questions evaluated

### Requirement: Download once, verify, then reuse
The system SHALL fetch each benchmark dataset once into a cache outside the repository, record its checksum, and reuse the cached copy on later runs only when the checksum still matches.

#### Scenario: Cached dataset is reused
- **WHEN** a benchmark is run again and the cached dataset matches its recorded checksum
- **THEN** the system SHALL NOT fetch the dataset again

#### Scenario: Checksum mismatch forces a refetch
- **WHEN** the cached dataset does not match its recorded checksum
- **THEN** the system SHALL treat the cache as invalid and SHALL refetch or report the dataset as unavailable

### Requirement: Offline fixture is labelled
When a dataset cannot be fetched, the system MAY run a committed offline fixture. Every report SHALL state `dataset_source=real` or `dataset_source=fixture`, and a fixture result SHALL NOT be presented as a benchmark score.

#### Scenario: Fixture run is labelled
- **WHEN** a benchmark runs on the offline fixture
- **THEN** the report SHALL state `dataset_source=fixture`
- **THEN** the report SHALL NOT be cited as a LoCoMo, LongMemEval-S or DMR score

### Requirement: Unreachable datasets are reported as blocked
When a dataset is unavailable and no fixture is selected, the system SHALL report that benchmark as blocked with the reason, and SHALL NOT print a metric for it.

#### Scenario: Blocked benchmark prints no metric
- **WHEN** the dataset cannot be fetched and no fixture is selected
- **THEN** the report SHALL state the benchmark is blocked and name the failed source
- **THEN** no metric value SHALL be printed for that benchmark

### Requirement: Deterministic subsets and provenance
The system SHALL support evaluating a deterministic subset selected with a recorded seed and size. Each report SHALL include the commit hash, the command, the seed, the subset size, the dependency versions, and the hardware description.

#### Scenario: Subset is reproducible
- **WHEN** the same benchmark is run twice with the same seed and size
- **THEN** both runs SHALL evaluate the same questions in the same order

### Requirement: Real-model benchmark gates are not passed without real models
A benchmark that requires a generation model or a real embedding anchor SHALL report the gate as blocked when that model is absent, and SHALL NOT substitute a mock result.

#### Scenario: Absent anchor blocks the gate
- **WHEN** a benchmark requiring the real anchor is run without the anchor model
- **THEN** the gate SHALL be reported as blocked with the missing path
- **THEN** the run SHALL NOT use the mock anchor silently
