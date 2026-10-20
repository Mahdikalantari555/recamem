# Spec Delta

## MODIFIED Requirements

### Requirement: Structured decision output
The system SHALL return a structured decision object containing the selected action, probability distribution, confidence score, and execution metadata. The memory console SHALL expose a decision panel that displays that object for an operator-supplied query without executing the selected action.

#### Scenario: Structured decision response
- **WHEN** a decision is made
- **THEN** the response SHALL include `action` (string), `probabilities` (list of floats), `confidence` (float), and `latency_ms` (float)
- **THEN** the response SHALL be JSON-serializable

#### Scenario: Console decision panel
- **WHEN** user enters a query in the decision panel
- **THEN** the console SHALL display the selected action, the probability distribution, the confidence, and the measured latency
- **THEN** displaying a decision SHALL NOT execute the action, mutate memory, or call a tool

#### Scenario: Rejected decision is displayed as rejected
- **WHEN** the query is out of domain and the decision head rejects it
- **THEN** the console SHALL show the rejection status and the observed similarity
- **THEN** the panel SHALL NOT show an action as if one had been selected
