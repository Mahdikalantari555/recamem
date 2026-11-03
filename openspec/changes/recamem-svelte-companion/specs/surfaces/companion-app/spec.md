# companion-app Specification

## Purpose

Defines a one-command Svelte companion surface that runs on core Python
dependencies, uses the same-origin loopback API, and discloses exactly what is
real: anchor mode, backend, checkpoint eligibility and streaming capability.

## ADDED Requirements

### Requirement: One-command launch with no runtime Node

The system SHALL provide `recamem companion` that starts the REST API and serves
the committed `companion/dist` on the same origin, prints the resolved URL, and
works with only the declared core Python dependencies. Browser-facing code
SHALL use relative URLs and SHALL NOT hard-code a loopback address or a second
origin.

#### Scenario: Cold start

- **WHEN** `recamem companion` runs on a machine with core dependencies and no Node toolchain
- **THEN** it SHALL serve the built app and answer `/api/v1/health`
- **THEN** every API request from the page SHALL be a relative URL resolved against the same origin

#### Scenario: Dry run

- **WHEN** `recamem companion --dry-run` runs
- **THEN** it SHALL print the URL, host policy, dist path and anchor mode without starting a server
- **THEN** it SHALL exit 0

### Requirement: Loopback-only default

The companion SHALL bind loopback by default. A non-loopback bind SHALL require
both an explicit `--host` and the existing remote opt-in, and SHALL be reported
in the launch output.

#### Scenario: Remote bind without opt-in

- **WHEN** `--host 0.0.0.0` is supplied without the remote opt-in
- **THEN** the launcher SHALL refuse to bind and SHALL explain the required flag

### Requirement: Honest runtime disclosure

The UI SHALL display anchor mode, generation backend, selected checkpoint and
model status from the API. When no eligible trained checkpoint exists, it SHALL
show an explicit `demo / recall-only` state and SHALL NOT present an untrained
or random-init model as trained. A `blocked` endpoint response SHALL be shown
as blocked, not as success.

#### Scenario: No trained checkpoint

- **WHEN** `GET /api/v1/checkpoints/best` returns `none`
- **THEN** the UI SHALL label chat as recall-only and SHALL name the missing artifact
- **THEN** it SHALL NOT display a "model ready" badge

#### Scenario: Explicit mock anchor

- **WHEN** the runtime anchor is mock
- **THEN** the status bar SHALL report `anchor=mock`
- **THEN** no surface SHALL report real-anchor quality

### Requirement: Streaming is capability-dependent

The UI SHALL stream chat tokens only when the selected backend advertises a
streaming callback. Otherwise it SHALL show a generating state and then the full
turn tagged `full_turn`, with a polling fallback for progress. It SHALL NOT
animate synthetic tokens.

#### Scenario: Backend without streaming

- **WHEN** `engine.chat()` returns a string and no streaming callback is available
- **THEN** the UI SHALL show one generating state followed by the complete response
- **THEN** the turn record SHALL state `delivery=full_turn`

### Requirement: Long connections do not block other requests

The companion server SHALL serve concurrent requests so that a job event stream
does not block health, inference or feedback calls.

#### Scenario: SSE open while inference runs

- **WHEN** a job event stream is open and an inference request arrives
- **THEN** the inference request SHALL complete without waiting for the stream to close
