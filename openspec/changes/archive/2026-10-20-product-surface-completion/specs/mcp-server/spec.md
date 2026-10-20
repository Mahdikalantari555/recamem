# Spec Delta

## MODIFIED Requirements

### Requirement: Exposed MCP tools
The system SHALL expose exactly the following tools via the MCP protocol, and the documented list SHALL match the `tools/list` response:
- `recall_memory(query, top_k)`: Sub-5ms factual lookup from plastic memory.
- `teach_fact(text, domain)`: Incremental on-device learning with automatic contradiction detection.
- `decide_action(query, options)`: Non-autoregressive System-1 option scoring.
- `list_droids()`: List available Droid profiles.
- `inspect_state(droid_name)`: Audit trace buffers, plastic weight norms, and fact counts.
- `generate_with_memory(query, strategy, droid_name)`: Grounded generation through the profile's configured backend.

#### Scenario: Tool invocation
- **WHEN** an agent sends a `tools/call` request for `recall_memory`
- **THEN** the server SHALL execute the recall against the specified Droid
- **THEN** the response SHALL include the matched facts with similarity scores
- **THEN** the response SHALL be a valid JSON-RPC result object

#### Scenario: Documented list matches the server
- **WHEN** the server advertises its tools
- **THEN** the advertised names SHALL equal the documented list, including `generate_with_memory`
- **THEN** a test SHALL compare the advertised names with the documented names and SHALL fail when either side gains or loses a tool

#### Scenario: Grounded generation tool is discoverable
- **WHEN** an agent lists the tools
- **THEN** `generate_with_memory` SHALL appear with its `query`, `strategy`, and `droid_name` arguments
- **WHEN** the tool is called for a profile with recalled facts
- **THEN** the result SHALL report the generated text and the strategy used
