# cli Specification

## Purpose

Unified headless command-line interface (`tmt-droid`) for terminal chat, markdown ingestion, System-1 decisions, and daemon serving without the Streamlit web UI.

## Requirements

### Requirement: Subcommand routing
The system SHALL support a unified CLI entrypoint with subcommands: `chat`, `teach`, `decide`, `eval`, `serve`, and `package`. The `eval` subcommand SHALL run a real evaluation and SHALL NOT return success for an unevaluated profile. Invalid arguments SHALL return exit code 2, an evaluation failure SHALL return 1, and a completed evaluation SHALL return 0.

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

### Requirement: Unix-friendly output
The system SHALL print clean, parseable text to stdout suitable for shell scripting and piping. The `eval` subcommand SHALL print one `key=value` pair per line in a stable order, including the evaluation mode and the sample size.

#### Scenario: Scriptable output
- **WHEN** the CLI is run in non-interactive mode
- **THEN** all response text SHALL be printed to stdout without ANSI escape codes or progress bars
- **THEN** error messages SHALL be printed to stderr with a clear error prefix

#### Scenario: Eval output is parseable
- **WHEN** `eval` completes successfully
- **THEN** stdout SHALL contain lines such as `mode=`, `profile=`, `samples=`, and `precision_at_k=`
- **THEN** a shell pipeline SHALL be able to split on `=` without stripping decoration
- **THEN** no metric SHALL be printed for a run that measured nothing
