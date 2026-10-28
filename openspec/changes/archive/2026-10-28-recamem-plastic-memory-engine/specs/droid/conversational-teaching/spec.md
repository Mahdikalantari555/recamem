# Spec Delta

## ADDED Requirements

### Requirement: Teaching uses the shared abbreviation-safe segmenter
The system SHALL segment text for `teach()` and for the chat teaching path with the same abbreviation-safe segmentation used by ingestion. It SHALL NOT use a bare sentence-boundary regex for that purpose.

#### Scenario: Teaching keeps a title intact
- **WHEN** the user teaches "The project lead is Dr. Mina Rostami."
- **THEN** the stored fact SHALL read "The project lead is Dr. Mina Rostami."
- **THEN** the stored fact SHALL NOT end at "Dr."

#### Scenario: Learning rate recorded for each write
- **WHEN** a teaching call stores facts
- **THEN** the train log SHALL record the learning rate applied to each stored write, as required by the schedule requirement above
