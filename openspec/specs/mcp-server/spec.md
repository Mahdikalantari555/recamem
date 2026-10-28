# mcp-server Specification

## Purpose

Zero-dependency stdio Model Context Protocol (MCP) server exposing Droid memory recall, teaching, and decision primitives to external AI agents (Claude Code, Cursor, Windsurf) via JSON-RPC 2.0 over standard input/output.

## Requirements

### Requirement: Stdio JSON-RPC 2.0 transport
The system SHALL implement an MCP server that communicates exclusively over stdin/stdout using JSON-RPC 2.0 framing, with all diagnostic logging redirected to stderr.

#### Scenario: JSON-RPC framing integrity
- **WHEN** the server receives a JSON-RPC request on stdin
- **THEN** the server SHALL parse the request and dispatch to the appropriate tool handler
- **THEN** the response SHALL be written to stdout as a single JSON-RPC response object
- **THEN** no diagnostic output SHALL appear on stdout

#### Scenario: Stdio server startup
- **WHEN** the MCP server is launched via `python -m src.cli serve --mcp`
- **THEN** the server SHALL initialize without errors
- **THEN** the server SHALL advertise its available tools via the `tools/list` method

### Requirement: Exposed MCP tools

The system SHALL expose exactly the following tools via the MCP protocol, each
named `recamem_<name>`, and the documented list SHALL match the `tools/list`
response. During the deprecation window the earlier unprefixed names SHALL
remain callable as aliases and SHALL be listed with `deprecated: true`:

- `recamem_recall_memory(query, top_k)`: Sub-5ms factual lookup from plastic and episodic memory.
- `recamem_teach_fact(text, domain)`: Incremental on-device learning with automatic contradiction detection.
- `recamem_decide_action(query, options)`: Non-autoregressive System-1 option scoring.
- `recamem_list_droids()`: List available agent profiles.
- `recamem_inspect_state(droid_name)`: Audit trace buffers, plastic weight norms, and fact counts.
- `recamem_generate_with_memory(query, strategy, droid_name)`: Grounded generation through the profile's configured backend.
- `recamem_research_topic(query, droid_name)`: Acquire knowledge about a topic from an external web source and learn it into the named agent.

Deprecated aliases (callable for one release, listed with `deprecated: true`):

- `recall_memory(query, top_k)`
- `teach_fact(text, domain)`
- `decide_action(query, options)`
- `list_droids()`
- `inspect_state(droid_name)`
- `generate_with_memory(query, strategy, droid_name)`
- `research_topic(query, droid_name)`

#### Scenario: Tool invocation

- **WHEN** an agent sends a `tools/call` request for `recamem_recall_memory`
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

### Requirement: Zero-dependency implementation
The system SHALL implement the MCP server using only Python standard library modules (`json`, `sys`, `socketserver`), without any external web framework or MCP SDK dependencies.

#### Scenario: Stdlib-only implementation
- **WHEN** the MCP server is imported
- **THEN** no third-party packages SHALL be required beyond the existing project dependencies
- **THEN** the server SHALL run in the standard `ai` conda environment without additional installs

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
