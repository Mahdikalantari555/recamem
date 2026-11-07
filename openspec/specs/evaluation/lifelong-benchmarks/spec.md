# evaluation/lifelong-benchmarks Specification

## Purpose
Provides reproducible continual learning evaluation protocols measuring retention, backward transfer, and catastrophic forgetting resistance.

## Requirements

### Requirement: Continual Retention Metric
The system SHALL evaluate model perplexity/loss on baseline domain $A$ after fine-tuning sequentially on novel domain $B$.

#### Scenario: Domain A retention measurement
- **WHEN** the model trains on novel domain text $B$
- **THEN** the retention ratio $\text{Loss}_A(\text{post}) / \text{Loss}_A(\text{pre})$ SHALL remain bounded below 1.25 to prevent catastrophic collapse

### Requirement: Backward transfer after sequential teaching
The lifelong benchmark SHALL teach a sequence of domains and, after each step, re-evaluate every earlier domain. It SHALL report the backward-transfer change for each earlier domain and the mean memory loss across domains, using the same question set at every step.

#### Scenario: Earlier domain forgetting is measured
- **WHEN** a third domain has been taught after two earlier domains
- **THEN** the report SHALL state the score for each earlier domain before and after the third teaching step
- **THEN** the report SHALL state the mean memory loss across earlier domains

#### Scenario: Retired facts are not counted as lost
- **WHEN** an earlier domain's facts were moved to the retired tier and remain answerable from the audit interface
- **THEN** the report SHALL state the loss separately for active-tier recall and audit-tier recall

### Requirement: Six-arm memory and online-adaptation benchmark protocol
The lifelong benchmark suite SHALL evaluate all six candidate memory/adaptation arms (`M0` persistent core state, `M1` hashed n-gram Engram, `M2` surprise-gated residual episodic KV, `M3` hybrid SQLite RRF store, `M4` sparse gated-delta fast weights, and `M5` online weight gradient plasticity) over an identical sequence of at least five domain shifts and report pre-adaptation accuracy, post-adaptation accuracy, full domain retention matrix, forward transfer, backward transfer (forgetting), adaptation wall-clock latency, and memory delta in bytes.

#### Scenario: Frozen-weight memory arms vs weight-plasticity arms compared side by side
- **WHEN** the 5-domain adaptation benchmark runs across `M0` through `M5` with identical domain order and seed
- **THEN** every arm SHALL report its full $5 \times 5$ retention matrix, mean forgetting, p50/p95 adaptation latency per taught paragraph, and peak memory overhead
- **THEN** any arm whose post-teaching retention ratio on Domain 1 violates the collapse ceiling (`> 1.25` loss ratio or `> 20` point accuracy collapse) SHALL be flagged as `failed_retention_gate`

### Requirement: Non-monotonic contradiction belief-revision benchmark
The lifelong benchmark suite SHALL include paired factual updates containing explicit entity-slot contradictions (such as replacing an outdated capital, project lead, or status with a newer fact) and measure whether each memory/adaptation candidate answers with the superseding fact without surfacing the superseded fact as current truth.

#### Scenario: Contradictory update supersedes prior fact cleanly
- **WHEN** Fact A ("The project status of Helios is paused.") is ingested followed later by Fact B ("The project status of Helios is active.")
- **THEN** querying the current status of Helios SHALL score `1.0` only when the response/selected option reflects "active" and does not assert "paused" as current
- **THEN** the benchmark report SHALL record contradiction-update accuracy separately from additive fact recall
