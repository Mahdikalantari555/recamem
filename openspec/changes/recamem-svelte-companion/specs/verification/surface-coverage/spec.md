# surface-coverage Specification (Delta)

## ADDED Requirements

### Requirement: Companion surface has contract tests

The companion surface SHALL have tests that start the launcher headlessly,
exercise every additive endpoint, assert the built `companion/dist` is served,
and verify the CLI parity check, MCP tool list and existing API payloads remain
unchanged. These tests SHALL run on core Python dependencies plus Node for the
frontend job, without model weights or network access.

#### Scenario: Headless smoke

- **WHEN** `python scripts/run_companion.py` runs with an explicit mock anchor and a temporary home
- **THEN** it SHALL pass with no Node, Rust or weights present
- **THEN** any failure SHALL exit non-zero with a JSON error object

#### Scenario: Existing surface regression

- **WHEN** a companion change alters an existing `/api/v1/*` response shape
- **THEN** the existing API test suite SHALL fail

### Requirement: Frontend tests do not fabricate results

The frontend test suite SHALL use a fixture API. A test SHALL fail if a view
renders an invented metric, a fake token stream, or a success state for a
`blocked` response.

#### Scenario: Blocked benchmark response

- **WHEN** the fixture returns `{"status": "blocked", "reason": "..."}` for a benchmark
- **THEN** the view SHALL render blocked with the reason
- **THEN** it SHALL NOT render a chart or a passing value
