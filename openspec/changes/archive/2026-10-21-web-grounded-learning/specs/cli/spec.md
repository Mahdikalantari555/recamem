# Spec Delta

## MODIFIED Requirements

### Requirement: Subcommand routing

The system SHALL support a unified CLI entrypoint with subcommands: `chat`, `teach`,
`decide`, `eval`, `research`, `serve`, and `package`. The `eval` subcommand SHALL run
a real evaluation and SHALL NOT return success for an unevaluated profile. Invalid
arguments SHALL return exit code 2, an evaluation failure SHALL return 1, and a
completed evaluation SHALL return 0.

#### Scenario: Chat subcommand

- **WHEN** user runs `python -m src.cli chat --name droid-alpha`
- **THEN** the CLI SHALL start an interactive terminal chat loop with the specified Droid
- **THEN** user input SHALL be sent to the Droid's `chat()` method and the response printed to stdout

#### Scenario: Teach subcommand

- **WHEN** user runs `cat report.md | python -m src.cli teach --name droid-remote-sensing`
- **THEN** the CLI SHALL read markdown from stdin
- **THEN** the CLI SHALL ingest the markdown into the specified Droid's memory
- **THEN** the CLI SHALL print the number of new facts absorbed and the memory state norm

#### Scenario: Decide subcommand

- **WHEN** user runs `python -m src.cli decide "Check soil moisture" --options "irrigate,inspect,wait"`
- **THEN** the CLI SHALL score the options using the System-1 decision head
- **THEN** the CLI SHALL print the selected action, probability distribution, and confidence score

#### Scenario: Eval subcommand

- **WHEN** user runs `python -m src.cli eval --name droid-rs --queries queries.jsonl --top-k 5`
- **THEN** the CLI SHALL compute precision@k, recall@k, and MRR against the labeled fact ids
- **WHEN** no query file is supplied
- **THEN** the CLI SHALL run the documented self-recall sanity check over a sample of stored facts
- **THEN** the output SHALL name the evaluation mode it actually ran

#### Scenario: Research subcommand

- **WHEN** user runs `python -m src.cli research "what is a lidar bathymetry system" --name droid-alpha`
- **THEN** the CLI SHALL ask the named Droid to acquire knowledge about that topic
- **THEN** the CLI SHALL report whether a search was performed, how many facts were learned, and which sources were used
- **WHEN** the named Droid has no search provider configured
- **THEN** the CLI SHALL report that external acquisition is unavailable
- **THEN** the CLI SHALL exit non-zero

#### Scenario: Serve subcommand

- **WHEN** user runs `python -m src.cli serve --mcp`
- **THEN** the CLI SHALL start the MCP stdio server
- **WHEN** user runs `python -m src.cli serve --rest`
- **THEN** the CLI SHALL start a lightweight REST microdaemon on localhost

#### Scenario: Package subcommand

- **WHEN** user runs `python -m src.cli package export droid-remote-sensing --out ./rs-v1.droid`
- **THEN** the CLI SHALL create a portable `.droid` archive containing the Droid's full state
- **WHEN** user runs `python -m src.cli package import ./rs-v1.droid --name droid-restored`
- **THEN** the CLI SHALL restore the Droid state from the archive

## ADDED Requirements

### Requirement: Research output is parseable

The `research` subcommand SHALL print one `key=value` pair per line in a stable
order, including the topic, the profile, whether a search was performed, the number
of facts learned, and the sources consulted. It SHALL print no metric for a run that
learned nothing.

#### Scenario: Scriptable research output

- **WHEN** the `research` subcommand completes
- **THEN** stdout SHALL contain lines such as `topic=`, `profile=`, `searched=`, `facts_learned=`, and `sources=`
- **THEN** a shell pipeline SHALL be able to split on `=` without stripping decoration
- **THEN** no source list SHALL be printed for a run that performed no search
