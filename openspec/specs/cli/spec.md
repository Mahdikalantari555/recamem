# cli Specification

## Purpose

Unified headless command-line interface (`tmt-droid`) for terminal chat, markdown ingestion, System-1 decisions, and daemon serving without the Streamlit web UI.

## Requirements

### Requirement: Subcommand routing
The system SHALL provide a unified CLI entrypoint named `recamem` with subcommands: `chat`, `teach`, `ingest`, `recall`, `decide`, `eval`, `bench`, `research`, `serve`, `package`, `doctor`, `migrate`, and `rtu`. The `eval` subcommand SHALL run a real evaluation and SHALL NOT return success for an unevaluated profile. Invalid arguments SHALL return exit code 2, an evaluation failure SHALL return 1, and a completed evaluation SHALL return 0.

#### Scenario: Chat subcommand
- **WHEN** user runs `recamem chat --name droid-alpha`
- **THEN** the CLI SHALL start an interactive terminal chat loop with the specified profile
- **THEN** user input SHALL be sent to the profile's `chat()` method and the response printed to stdout

#### Scenario: Teach subcommand
- **WHEN** user runs `cat report.md | recamem teach --name droid-remote-sensing`
- **THEN** the CLI SHALL read markdown from stdin
- **THEN** the CLI SHALL ingest the markdown into the specified profile's memory
- **THEN** the CLI SHALL print the number of new facts absorbed and the memory state norm

#### Scenario: Decide subcommand
- **WHEN** user runs `recamem decide "Check soil moisture" --options "irrigate,inspect,wait"`
- **THEN** the CLI SHALL score the options using the System-1 decision head
- **THEN** the CLI SHALL print the selected action, probability distribution, and confidence score

#### Scenario: Eval subcommand
- **WHEN** user runs `recamem eval --name droid-rs --queries queries.jsonl --top-k 5`
- **THEN** the CLI SHALL compute precision@k, recall@k, and MRR against the labeled fact ids
- **WHEN** no query file is supplied
- **THEN** the CLI SHALL run the documented self-recall sanity check over a sample of stored facts
- **THEN** the output SHALL name the evaluation mode it actually ran

#### Scenario: Research subcommand
- **WHEN** user runs `recamem research "what is a lidar bathymetry system" --name droid-alpha`
- **THEN** the CLI SHALL ask the named profile to acquire knowledge about that topic
- **THEN** the CLI SHALL report whether a search was performed, how many facts were learned, and which sources were used
- **WHEN** the named profile has no search provider configured
- **THEN** the CLI SHALL report that external acquisition is unavailable
- **THEN** the CLI SHALL exit non-zero

#### Scenario: Serve subcommand
- **WHEN** user runs `recamem serve --mcp`
- **THEN** the CLI SHALL start the MCP stdio server
- **WHEN** user runs `recamem serve --rest`
- **THEN** the CLI SHALL start a lightweight REST microdaemon on localhost

#### Scenario: Package subcommand
- **WHEN** user runs `recamem package export droid-remote-sensing --out ./rs-v1.droid`
- **THEN** the CLI SHALL create a portable `.droid` archive containing the profile's full state
- **WHEN** user runs `recamem package import ./rs-v1.droid --name droid-restored`
- **THEN** the CLI SHALL restore the profile state from the archive

#### Scenario: Migrate and doctor subcommands
- **WHEN** user runs `recamem doctor`
- **THEN** the CLI SHALL print the readiness items required by the identity capability and exit 0 only if all required items are ready
- **WHEN** user runs `recamem migrate droids/`
- **THEN** the CLI SHALL run the non-destructive migration and print the migration record path

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

### Requirement: Benchmark subcommand

The system SHALL provide a `bench` subcommand that measures the documented
performance and evaluation metrics for a profile, a checkpoint, or a model
configuration, and prints them in the parseable output format. A measurement that
could not be taken SHALL be reported as skipped with a reason, and no value SHALL
be printed for it.

#### Scenario: Benchmark subcommand measures the documented metrics

- **WHEN** user runs `recamem bench --profile droids/droid-rs --tokens 200000`
- **THEN** the CLI SHALL measure the documented throughput, parameter count, and memory footprint
- **THEN** stdout SHALL contain lines such as `tokens_per_second=`, `params=`, and `rss_mb=`
- **WHEN** a requested measurement cannot be taken
- **THEN** the CLI SHALL report that measurement as skipped with a reason
- **THEN** the CLI SHALL NOT print a value for it

#### Scenario: Benchmark writes a report

- **WHEN** the `bench` subcommand completes
- **THEN** the CLI SHALL write a machine-readable report containing the commit hash, the command, the dependency versions, the configuration including seeds, the hardware, the measurements, and the outcome
- **THEN** the CLI SHALL print the report path to stdout

#### Scenario: Benchmark can score the grammar probes

- **WHEN** user runs `recamem bench --probes evals/grammar_probes.jsonl --checkpoint droids/droid-rs/rtu.safetensors`
- **THEN** the CLI SHALL score the probe set against that checkpoint
- **THEN** stdout SHALL contain the per-phenomenon accuracies
- **THEN** the aggregate accuracy SHALL NOT be printed without the per-phenomenon breakdown

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
