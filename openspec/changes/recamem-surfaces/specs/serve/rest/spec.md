# Spec Delta — Versioned loopback REST API

## ADDED Requirements

### Requirement: Versioned loopback API
The system SHALL expose a versioned REST API under `/api/v1/` when `recamem serve
--rest` runs. The API SHALL bind to loopback only by default, SHALL answer
`GET /api/v1/health` with the anchor mode and profile name, and SHALL reject
unknown paths with HTTP 404 and a JSON body. Errors SHALL return a JSON object
with an `error` key and SHALL never leak tracebacks.

#### Scenario: Health probe
- **WHEN** the client requests `GET /api/v1/health`
- **THEN** the response SHALL be 200 with keys `status`, `profile`, and `anchor`
- **THEN** `anchor` SHALL be the explicit anchor mode, never a silent default

#### Scenario: Profile operations
- **WHEN** the client posts `{"text": "..."}` to `/api/v1/teach`
- **THEN** the response SHALL report how many facts were stored
- **WHEN** the client posts `{"query": "..."}` to `/api/v1/recall`
- **THEN** the response SHALL include each hit's `text`, `source`, and channel scores

#### Scenario: Failure surface
- **WHEN** a request targets an unknown path
- **THEN** the server SHALL return 404 with a JSON `error` body
- **WHEN** the profile cannot be loaded
- **THEN** the server SHALL return 503 with a JSON `error` body naming the cause
