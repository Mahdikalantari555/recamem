# Spec Delta

## MODIFIED Requirements

### Requirement: Exposed MCP tools

The system SHALL expose exactly the following tools via the MCP protocol, and the
documented list SHALL match the `tools/list` response:

- `recall_memory(query, top_k)`: Sub-5ms factual lookup from plastic memory.
- `teach_fact(text, domain)`: Incremental on-device learning with automatic contradiction detection.
- `decide_action(query, options)`: Non-autoregressive System-1 option scoring.
- `list_droids()`: List available Droid profiles.
- `inspect_state(droid_name)`: Audit trace buffers, plastic weight norms, and fact counts.
- `generate_with_memory(query, strategy, droid_name)`: Grounded generation through the profile's configured backend.
- `research_topic(query, droid_name)`: Acquire knowledge about a topic from an external web source and learn it into the named Droid.

#### Scenario: Tool invocation

- **WHEN** an agent sends a `tools/call` request for `recall_memory`
- **THEN** the server SHALL execute the recall against the specified Droid
- **THEN** the response SHALL include the matched facts with similarity scores
- **THEN** the response SHALL be a valid JSON-RPC result object

#### Scenario: Documented list matches the server

- **WHEN** the server advertises its tools
- **THEN** the advertised names SHALL equal the documented list, including `generate_with_memory` and `research_topic`
- **THEN** a test SHALL compare the advertised names with the documented names and SHALL fail when either side gains or loses a tool

#### Scenario: Grounded generation tool is discoverable

- **WHEN** an agent lists the tools
- **THEN** `generate_with_memory` SHALL appear with its `query`, `strategy`, and `droid_name` arguments
- **WHEN** the tool is called for a profile with recalled facts
- **THEN** the result SHALL report the generated text and the strategy used

#### Scenario: Research tool is discoverable

- **WHEN** an agent lists the tools
- **THEN** `research_topic` SHALL appear with its `query` and `droid_name` arguments
- **WHEN** the tool is called for a profile with a configured search provider
- **THEN** the result SHALL report whether a search was performed, the facts learned, and the sources consulted
- **WHEN** the tool is called for a profile with no configured search provider
- **THEN** the result SHALL report that external acquisition is unavailable
- **THEN** the result SHALL be a valid JSON-RPC result object rather than an error

## ADDED Requirements

### Requirement: Research tool is bounded

The `research_topic` tool SHALL bound the work it performs on behalf of a caller: it
SHALL NOT issue more than the configured maximum number of queries and page fetches
for one call, and it SHALL apply the profile's timeout to every request. A failure
inside the tool SHALL be reported inside the JSON-RPC result, with the error named,
and SHALL NOT terminate the server.

#### Scenario: Bounded work per call

- **WHEN** an agent calls `research_topic` on a topic with many results
- **THEN** the server SHALL NOT exceed the configured fetch limit for that call
- **WHEN** a provider call fails inside the tool
- **THEN** the result SHALL name the failure
- **THEN** the server SHALL remain available for the next request
