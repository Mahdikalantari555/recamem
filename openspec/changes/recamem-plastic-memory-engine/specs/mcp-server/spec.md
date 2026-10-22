# Spec Delta

## MODIFIED Requirements

### Requirement: Exposed MCP tools

The system SHALL expose the following tools via the MCP protocol, each named `recamem_<name>`, and the documented list SHALL match the `tools/list` response. During the deprecation window, the earlier unprefixed names SHALL remain callable as aliases and SHALL be listed with `deprecated: true`.

- `recamem_recall_memory(query, top_k)`: Sub-5ms factual lookup from plastic and episodic memory, returning verbatim facts with sources.
- `recamem_teach_fact(text, domain)`: Incremental on-device learning with automatic contradiction detection.
- `recamem_decide_action(query, options)`: Non-autoregressive System-1 option scoring.
- `recamem_list_droids()`: List available agent profiles.
- `recamem_inspect_state(droid_name)`: Audit trace buffers, plastic weight norms, and fact counts.
- `recamem_generate_with_memory(query, strategy, droid_name)`: Grounded generation through the profile's configured backend.
- `recamem_research_topic(query, droid_name)`: Acquire knowledge about a topic from an external web source and learn it into the named profile.

#### Scenario: Tool invocation

- **WHEN** an agent sends a `tools/call` request for `recamem_recall_memory`
- **THEN** the server SHALL execute the recall against the specified profile
- **THEN** the response SHALL include the matched facts with their sources and similarity scores
- **THEN** the response SHALL be a valid JSON-RPC result object

#### Scenario: Alias call during the deprecation window

- **WHEN** an agent sends a `tools/call` request for `recall_memory`
- **THEN** the server SHALL execute the same handler as `recamem_recall_memory`
- **THEN** `tools/list` SHALL mark `recall_memory` as `deprecated: true`

#### Scenario: Documented list matches the server

- **WHEN** the server advertises its tools
- **THEN** the advertised names SHALL equal the documented list plus the documented aliases
- **THEN** a test SHALL compare the advertised names with the documented names and SHALL fail when either side gains or loses a tool

#### Scenario: Grounded generation tool is discoverable

- **WHEN** an agent lists the tools
- **THEN** `recamem_generate_with_memory` SHALL appear with its `query`, `strategy`, and `droid_name` arguments
- **WHEN** the tool is called for a profile with recalled facts
- **THEN** the result SHALL report the generated text and the strategy used

#### Scenario: Research tool is discoverable

- **WHEN** an agent lists the tools
- **THEN** `recamem_research_topic` SHALL appear with its `query` and `droid_name` arguments
- **WHEN** the tool is called for a profile with a configured search provider
- **THEN** the result SHALL report whether a search was performed, the facts learned, and the sources consulted
- **WHEN** the tool is called for a profile with no configured search provider
- **THEN** the result SHALL report that external acquisition is unavailable
- **THEN** the result SHALL be a valid JSON-RPC result object rather than an error

### Requirement: Stdio JSON-RPC 2.0 transport
The system SHALL implement an MCP server that communicates exclusively over stdin/stdout using JSON-RPC 2.0 framing, with all diagnostic logging redirected to stderr.

#### Scenario: JSON-RPC framing integrity
- **WHEN** the server receives a JSON-RPC request on stdin
- **THEN** the server SHALL parse the request and dispatch to the appropriate tool handler
- **THEN** the response SHALL be written to stdout as a single JSON-RPC response object
- **THEN** no diagnostic output SHALL appear on stdout

#### Scenario: Stdio server startup
- **WHEN** the MCP server is launched via `recamem serve --mcp`
- **THEN** the server SHALL initialize without errors
- **THEN** the server SHALL advertise its available tools via the `tools/list` method

## ADDED Requirements

### Requirement: Bounded research tool
The `recamem_research_topic` tool SHALL bound the work it performs on behalf of a caller: it SHALL NOT issue more than the configured maximum number of queries and page fetches for one call, and it SHALL apply the profile's timeout to every request. A failure inside the tool SHALL be reported inside the JSON-RPC result, with the error named, and SHALL NOT terminate the server.

#### Scenario: Bounded work per call
- **WHEN** an agent calls `recamem_research_topic` on a topic with many results
- **THEN** the server SHALL NOT exceed the configured fetch limit for that call
- **WHEN** a provider call fails inside the tool
- **THEN** the result SHALL name the failure
- **THEN** the server SHALL remain available for the next request
