# memory/ingestion Specification

## Purpose
Turns a folder of Markdown, text and optional PDF documents into stored, attributed facts with abbreviation-safe sentence boundaries, so that a user can build a specialist memory from a search result, a PDF, or about one hundred Markdown files in one command.

## Requirements

### Requirement: Abbreviation-safe sentence segmentation
The system SHALL split prose into propositions without treating a period after a known abbreviation, an initial, or a decimal number as a sentence boundary. This SHALL apply to every path that stores propositions from text, including teaching and ingestion.

#### Scenario: Title abbreviation is not a boundary
- **WHEN** the text is "The project lead is Dr. Mina Rostami. The budget is 5 million."
- **THEN** the system SHALL produce two propositions
- **THEN** the first proposition SHALL be "The project lead is Dr. Mina Rostami."

#### Scenario: Initials and decimals are not boundaries
- **WHEN** the text contains "J. R. R. Tolkien" or "the value is 3.5 metres"
- **THEN** the system SHALL NOT split inside those expressions

#### Scenario: Real sentence ends still split
- **WHEN** the text contains two sentences ending in a period followed by a space and a capital letter
- **THEN** the system SHALL produce two propositions

### Requirement: Folder ingest with provenance
The system SHALL ingest every supported file under a directory through `recamem ingest <dir>`. Supported files are Markdown and plain text, and PDF when the optional PDF reader is installed. Each stored fact SHALL record the file path, the heading path it appeared under, and a content hash of the source file.

#### Scenario: Markdown folder is ingested with headings
- **WHEN** a directory of Markdown files is ingested
- **THEN** each fact SHALL record its file path and the heading path it was written under
- **THEN** code fences SHALL NOT be segmented as prose

#### Scenario: Unsupported files are skipped with a reason
- **WHEN** the directory contains a file of an unsupported type
- **THEN** the report SHALL list the file as skipped with the reason
- **THEN** the run SHALL continue with the remaining files

### Requirement: PDF ingestion is optional and honest
The system SHALL extract text from PDF files only when the optional PDF reader is installed. Without it, PDF files SHALL be reported as skipped with the reason that the reader is missing. A PDF with no extractable text SHALL be reported as skipped with that reason.

#### Scenario: PDF without the reader
- **WHEN** a PDF is given to ingest and the reader is not installed
- **THEN** the file SHALL be reported as skipped with the missing-reader reason
- **THEN** no fact SHALL be stored from that file

#### Scenario: PDF with no text layer
- **WHEN** a PDF yields no extractable text
- **THEN** the file SHALL be reported as skipped with the no-text reason

### Requirement: Re-ingest is idempotent and supersedes changed content
The system SHALL skip a file whose content hash is unchanged since its last ingest. When a file changed, the system SHALL supersede the facts previously stored from that file rather than duplicating them, and SHALL NOT remove those facts from the store.

#### Scenario: Unchanged file is a no-op
- **WHEN** an ingested directory is ingested again without changes
- **THEN** no new facts SHALL be stored
- **THEN** the report SHALL state that the files were unchanged

#### Scenario: Changed file supersedes its old facts
- **WHEN** a file's content changes and it is ingested again
- **THEN** the facts from the previous version SHALL be marked superseded
- **THEN** the superseded facts SHALL remain queryable for audit

### Requirement: Ingest report contains only measured values
The ingest command SHALL report the number of files read, files skipped with reasons, and facts stored. It SHALL NOT report a recall or quality figure for an ingest run.

#### Scenario: Run that stores nothing
- **WHEN** every file is skipped
- **THEN** the report SHALL state that zero facts were stored and SHALL list each skip reason
