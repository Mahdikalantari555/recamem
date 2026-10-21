# adaptation/web-grounded-learning Specification

## Purpose

Lets the Droid acquire knowledge it was never given by searching an external web
source at inference time: deciding when to search, bounding the search, keeping only
what is usable, and attributing every fact it learned to its source.

## Requirements

### Requirement: External knowledge acquisition

The system SHALL be able to acquire knowledge it was not given. It SHALL issue a web
query, fetch the returned pages, convert the retrieved content into standalone
factual propositions, and store those propositions as facts through the target's
existing knowledge store and plastic memory.

#### Scenario: Question outside the memory is researched

- **WHEN** the user asks a question the memory cannot answer
- **THEN** the system SHALL retrieve external content about that question
- **THEN** the system SHALL store the resulting propositions as facts
- **THEN** a later question about the same topic SHALL be answerable from memory without another search

#### Scenario: Searched content becomes recallable

- **WHEN** facts were learned from a web source
- **THEN** the ordinary recall path SHALL return them
- **THEN** the recalled answer SHALL contain the searched content

### Requirement: Search decision policy

The system SHALL decide whether to search from the retrieval confidence of the
existing memory. It SHALL NOT search when memory answers the question confidently,
and it SHALL NOT repeat a search for a topic it already searched within a
configurable suppression window.

#### Scenario: Confident memory suppresses the search

- **WHEN** memory answers the question above the configured confidence threshold
- **THEN** the system SHALL NOT perform an external search
- **THEN** the answer SHALL be produced from memory alone

#### Scenario: Repeated question is not re-searched

- **WHEN** the same topic was searched within the suppression window
- **THEN** the system SHALL NOT perform a second external search
- **THEN** the system SHALL answer from what it already learned

#### Scenario: Search can be disabled

- **WHEN** the profile disables external acquisition
- **THEN** the system SHALL never perform an external search
- **THEN** the system SHALL answer exactly as it does with no provider configured

### Requirement: Automatic research on an unanswerable question

When a profile enables automatic acquisition, the system SHALL research a question
that memory cannot answer confidently, and SHALL answer the question from the facts
it acquired. The answer SHALL be reported as produced by the researched path, and
the question SHALL NOT be answered by free generation without acquired facts.

#### Scenario: Unanswerable question triggers a search

- **WHEN** the chat receives a question whose retrieval confidence is below the configured threshold
- **THEN** the system SHALL research the question
- **THEN** the system SHALL answer from the facts it acquired
- **THEN** the answer path SHALL be reported as `researched`

#### Scenario: Research failure falls back to the no-information answer

- **WHEN** the research yields no usable facts
- **THEN** the system SHALL return the no-information fallback
- **THEN** the system SHALL NOT generate an ungrounded answer

### Requirement: Bounded search execution

The system SHALL bound every search: a maximum number of queries, a maximum number of
pages fetched per search, a per-request timeout, and a maximum response size. It
SHALL reject non-HTTP schemes and hosts that resolve to loopback, link-local, or
private address ranges. A provider or transport failure SHALL return a structured
failure and SHALL NOT write to the memory.

#### Scenario: Timeout is reported, not raised

- **WHEN** a search or fetch exceeds its timeout
- **THEN** the system SHALL report a structured failure naming the timeout
- **THEN** the memory SHALL be unchanged

#### Scenario: Unsafe target is refused

- **WHEN** a result points at a loopback or private address
- **THEN** the system SHALL refuse to fetch it
- **THEN** the refusal SHALL be reported and SHALL NOT abort the rest of the search

#### Scenario: Response size is capped

- **WHEN** a page exceeds the maximum response size
- **THEN** the system SHALL stop reading the response at the cap
- **THEN** the truncated text SHALL still be usable as distillation input

### Requirement: Provenance of learned facts

Facts acquired through search SHALL record the source URL, the retrieval timestamp,
and the query that found them. The system SHALL be able to report where a
web-sourced fact came from, and SHALL distinguish web-sourced facts from
user-taught facts in recall output.

#### Scenario: Provenance survives a restart

- **WHEN** a profile holding web-sourced facts is saved and loaded again
- **THEN** the source URL and retrieval timestamp SHALL still be attached to the fact
- **THEN** the fact SHALL still be reported as web-sourced

#### Scenario: Recall distinguishes web facts from taught facts

- **WHEN** recall returns a mix of web-sourced and user-taught facts
- **THEN** each result SHALL identify its origin
- **THEN** the origin SHALL be one of the documented origins, not free text

### Requirement: Researched answers are declared

The system SHALL distinguish an answer produced from web-sourced facts from an answer
produced from stored memory alone. It SHALL report the researched answer as
researched, SHALL report the sources it used, and SHALL NOT report a researched
answer as a purely recalled one.

#### Scenario: Researched answer cites its sources

- **WHEN** the answer is grounded in facts learned from the web
- **THEN** the answer path SHALL be reported as researched
- **THEN** the sources used SHALL be listed with the answer

### Requirement: Honest failure when nothing is found

When a search yields no usable content, the system SHALL report that it does not
know. It SHALL NOT fabricate an answer and SHALL NOT store an empty or placeholder
fact.

#### Scenario: Empty search result stores nothing

- **WHEN** the search returns no usable content
- **THEN** the system SHALL report that it has no information
- **THEN** the number of stored facts SHALL be unchanged

### Requirement: Provider portability

The system SHALL support at least one key-free public search provider and at least
one key-configurable provider, selected by configuration. With no provider
configured, the system SHALL report that external acquisition is unavailable rather
than failing.

#### Scenario: Provider selection by configuration

- **WHEN** the profile names a key-configurable provider and its credentials are present
- **THEN** that provider SHALL be used
- **WHEN** the profile names no provider
- **THEN** the system SHALL report that external acquisition is unavailable
- **THEN** the system SHALL NOT raise an error
