# Tasks

## 1. Search provider interface

- [ ] 1.1 Define the `SearchProvider` interface, the `SearchResult` value type, and a
  null provider in `src/model/web_teacher.py`. The interface exposes one method that
  takes a query and a result limit and returns results carrying a title, a URL, and a
  snippet. Verify in `tests/test_web_teacher.py` that the null provider returns no
  results and raises nothing.
- [ ] 1.2 Implement the key-free public provider against a key-free HTML endpoint.
  Parse the result listing into titles, URLs, and snippets. Verify in
  `tests/test_web_teacher.py` against a checked-in HTML fixture that three listings
  parse into three results with the expected URLs. The test SHALL NOT open a socket.
- [ ] 1.3 Implement the key-configurable provider against a JSON search API, reading
  the credential from the environment. Verify in `tests/test_web_teacher.py` with an
  injected fake transport that the request carries the credential and that the
  response is mapped to results.
- [ ] 1.4 Select the provider by configuration name. Verify in
  `tests/test_web_teacher.py` that an unknown provider name yields a structured
  failure naming the unknown name and that no provider yields the unavailable
  result.

## 2. Fetcher and page extraction

- [ ] 2.1 Implement the page fetcher with a scheme allowlist of `http` and `https`
  and a host check that refuses loopback, private, link-local, reserved, and
  multicast addresses after resolution. Verify in `tests/test_web_teacher.py` that a
  loopback URL and a private-range URL are both refused and that the refusal names
  the reason.
- [ ] 2.2 Bound every fetch: at most two redirects with the host check re-applied at
  each hop, a response byte cap, a timeout on connect and read, and a descriptive
  user agent. Verify in `tests/test_web_teacher.py` against a `http.server` bound to
  loopback in a thread that the size cap truncates, that the timeout returns a
  structured timeout failure, and that the loopback allowance is scoped to the test
  fixture rather than a production bypass.
- [ ] 2.3 Implement HTML to text extraction. Drop `script`, `style`, `nav`,
  `header`, `footer`, `aside`, and `form` content, keep text from `p`, `li`,
  `h1`-`h6`, `td`, and `pre`, unescape entities, collapse whitespace, and stop at
  the per-page character budget. Verify in `tests/test_web_teacher.py` that script
  content is absent, that paragraph text survives, and that entities are decoded.
- [ ] 2.4 Report every fetch failure as a structured failure carrying the reason,
  and never as an exception escaping the fetcher. Verify in `tests/test_web_teacher.py`
  that an unreachable host yields a failure naming the host and that no exception
  propagates.

## 3. Provenance in the knowledge store

- [ ] 3.1 Add `source_url` and `source_query` columns to the `facts` table through
  the existing schema-migration path. Verify in `tests/test_knowledge_store.py` that
  a database created before the change opens, gains the columns, and keeps its
  existing rows.
- [ ] 3.2 Define the closed origin vocabulary `user_text`, `chat`,
  `continuous_stream`, `web` and validate it on insert. Verify in
  `tests/test_knowledge_store.py` that an insert with an unknown origin is rejected
  with an error naming the value.
- [ ] 3.3 Attach the URL, the query, and the retrieval timestamp to a fact inserted
  from a web source. Verify in `tests/test_knowledge_store.py` that the stored fact
  reports all three and that a fact inserted without a source keeps its null URL.
- [ ] 3.4 Return the origin, the URL, and the query in recall results. Verify in
  `tests/test_knowledge_store.py` and `tests/test_droid_engine.py` that a web-sourced
  fact is reported as web-sourced with its URL, and that a taught fact is reported
  as taught with no URL.

## 4. Search learner loop

- [ ] 4.1 Implement the search decision in `src/model/search_learn.py` from the
  engine's retrieval confidence, with a per-topic suppression window and a
  normalization of the topic used as the suppression key. Verify in
  `tests/test_search_learn.py` that a confident recall performs no search, that a
  low-confidence recall does, and that a repeated topic inside the window does not
  search again.
- [ ] 4.2 Implement the acquisition loop: search, fetch up to the page cap,
  extract text, distill into propositions through the teacher endpoint when one is
  configured, otherwise through the local sentence splitter, then teach each
  proposition with its provenance. Verify in `tests/test_search_learn.py` with an
  injected fake provider and fake teacher that the facts are taught, that each
  carries its source URL and query, and that the local fallback path also produces
  taught facts.
- [ ] 4.3 Enforce the per-session and per-minute search bounds and the fetch and
  query caps. Verify in `tests/test_search_learn.py` that the loop stops at the
  bound, that the result reports that the bound was reached, and that no further
  provider call is made.
- [ ] 4.4 Return a structured result from every path, including the empty-result
  path, and store nothing when nothing usable was found. Verify in
  `tests/test_search_learn.py` that the fact count is unchanged after an empty
  result and that the result reports no information rather than an answer.

## 5. Engine integration

- [ ] 5.1 Add the `researched` answer path alongside `recall`, `fused`, and `llm`,
  and map it in the legacy path helper to the extractive legacy name. Verify in
  `tests/test_droid_engine.py` that the modern path reports `researched` and that
  the legacy path still returns only the two legacy names.
- [ ] 5.2 Add the engine research entry point that returns the topic, whether a
  search happened, the queries issued, the sources consulted, the facts learned,
  the answer, and the answer path. Verify in `tests/test_droid_engine.py` that the
  returned contract carries every field and that a profile with no provider
  configured reports the unavailable result without raising.
- [ ] 5.3 Add the optional chat hook that researches a question the memory cannot
  answer confidently and answers from the acquired facts, gated on the profile
  configuration. Verify in `tests/test_droid_engine.py` that the hooked chat sets
  the answer path to `researched`, that a failed research returns the no-information
  fallback instead of an ungrounded answer, and that a chat on a profile with the
  feature disabled performs no network call.
- [ ] 5.4 Read the research configuration from the profile manifest, defaulting to
  disabled when the block is absent. Verify in `tests/test_droid_profiles.py` that
  a profile without the block loads with the feature disabled and that a profile
  with the block loads with the configured provider and bounds.

## 6. Surfaces

- [ ] 6.1 Add the `research` subcommand to `src/cli.py`, printing one `key=value`
  pair per line for the topic, the profile, whether a search happened, the facts
  learned, and the sources consulted. Verify in `tests/test_cli.py` that the
  command runs against an injected fake provider and that the output parses on `=`.
- [ ] 6.2 Add the `research_topic` tool to `src/mcp_server.py` and advertise it.
  Update the exact tool-list assertion in `tests/test_mcp_server_additional.py`,
  which currently names six tools, to include the new one. Verify in
  `tests/test_mcp_server.py` that the advertised tool list matches the
  documented list including the new tool, that a call for a profile with a provider
  reports the search outcome, and that a call for a profile without a provider
  returns a result object rather than an error.
- [ ] 6.3 Add the live smoke script that performs one real search, guarded by an
  environment variable, and prints the sources it consulted. Verify by running it
  with the variable unset that it exits with a message naming the variable and
  performs no request.

## 7. Documentation

- [ ] 7.1 Add `docs/research.md` describing the provider configuration, the search
  bounds, the provenance model, the origin vocabulary, and the safety limits. Verify
  that every configuration key named in the document exists in the manifest parser
  and that the documented commands run as written.
- [ ] 7.2 Update `README.md` with the research feature, the configuration block, and
  a pointer to `docs/research.md`. Verify that the documented example command runs
  and reports the unavailable result on a profile with no provider.
