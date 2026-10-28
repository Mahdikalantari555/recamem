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
