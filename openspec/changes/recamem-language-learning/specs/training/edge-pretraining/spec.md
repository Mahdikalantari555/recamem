# Spec Delta

## MODIFIED Requirements

### Requirement: Scale run acceptance criteria

A scale run SHALL be reported as passed only when every one of the following
holds: the consumed byte count reached the documented target, the parameter count
matches the documented configuration, the final logged training loss is below the
initial logged training loss, the held-out perplexity is finite and was computed on
the excluded split, and the run completed on the documented hardware class. For
language-learning runs, the following additional criteria SHALL hold: the BPB
SHALL be below the trigram baseline (G2), at least 15 of 20 generated stories
SHALL be readable (G3), and MQAR-16 accuracy SHALL remain above 0.75 (G4). When
any criterion does not hold, the outcome SHALL be failed or blocked and the report
SHALL name the criterion that was not met.

#### Scenario: Passing run

- **WHEN** a run meets every acceptance criterion
- **THEN** the outcome SHALL be passed
- **THEN** every criterion SHALL be individually recorded as met

#### Scenario: Failing run names the criterion

- **WHEN** a run does not meet one acceptance criterion
- **THEN** the outcome SHALL be failed or blocked
- **THEN** the report SHALL name the criterion that was not met
- **THEN** the report SHALL NOT present the run as passed

#### Scenario: Language-learning run checks gates

- **WHEN** a language-learning run completes
- **THEN** the report SHALL check G2 (BPB below trigram)
- **THEN** the report SHALL check G3 (15/20 readable stories)
- **THEN** the report SHALL check G4 (MQAR-16 above 0.75)
- **THEN** the report SHALL name any gate that was not met