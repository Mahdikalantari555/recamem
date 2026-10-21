# Web-grounded learning

A Droid can acquire knowledge it was never given: it searches, fetches, distills
the pages into standalone propositions, and teaches them into the same knowledge
store and plastic memory it uses for anything else. Everything it learned this
way is attributed to its source, and an answer built from those facts is
reported as `researched`, not as a memory answer.

Acquisition is **opt-in**. A profile with no `research` block behaves exactly as
it did before the capability existed: no provider, no network, no change.

## Configuration

The `research` block lives in the profile manifest (`droids/<name>/config.json`):

```json
{
  "research": {
    "enabled": true,
    "provider": "duckduckgo",
    "confidence_threshold": 0.55,
    "suppression_window_seconds": 300,
    "max_queries_per_session": 8,
    "max_queries_per_minute": 3,
    "max_pages_per_search": 3,
    "result_limit": 5,
    "page_char_budget": 6000,
    "min_proposition_chars": 12,
    "max_propositions_per_page": 5,
    "request_timeout": 5.0,
    "max_response_bytes": 524288
  }
}
```

| key | meaning |
| --- | --- |
| `enabled` | the master switch; `false` (or an absent block) never searches |
| `provider` | `duckduckgo` (key-free HTML endpoint), `json_api` (key-configurable), or absent |
| `base_url` | required by `json_api`: the JSON search endpoint |
| `api_key` / `api_key_env` | the `json_api` credential, read from the environment (`TMT_SEARCH_API_KEY` by default) |
| `confidence_threshold` | a recall at or above this confidence suppresses the search |
| `suppression_window_seconds` | a topic searched inside this window is not searched again |
| `max_queries_per_session` / `max_queries_per_minute` | the search budget |
| `max_pages_per_search` / `result_limit` | how many results are fetched / requested |
| `page_char_budget` / `max_propositions_per_page` | distillation input and output caps |
| `request_timeout` / `max_response_bytes` | per-request limits |

Programmatic configuration uses the same block:

```python
engine.configure_research({"enabled": True, "provider": "duckduckgo"})
```

## Commands

```sh
# Ask a profile to learn about a topic (search + fetch + distill + teach)
PYTHONPATH=. python -m src.cli research "what is a lidar bathymetry system" --name droid-alpha

# The same through the MCP server: the research_topic tool
#   research_topic(query="what is a lidar bathymetry system", droid_name="droid-alpha")

# One live search, guarded by an environment variable
TMT_ALLOW_NETWORK=1 PYTHONPATH=. python scripts/smoke_web_research.py "lidar bathymetry"
```

`research` prints one `key=value` pair per line: `topic`, `profile`, `searched`,
`facts_learned`, and `sources`. A run that performed no search prints no source
list, and a profile without a provider exits non-zero with the reason on stderr.
The MCP tool returns the unavailable result inside a normal JSON-RPC result
rather than an error, so an agent can tell "cannot look" apart from "the server
broke".

## Search decision

The policy is in `src/model/search_learn.py`, not in the provider:

1. acquisition disabled or no provider → no search;
2. retrieval confidence at or above `confidence_threshold` → no search;
3. the same normalized topic searched inside the suppression window → no search;
4. the per-minute or per-session budget exhausted → no search, reported bounded;
5. otherwise: search, fetch up to `max_pages_per_search`, distill, teach.

An explicit `research` call bypasses only rule 2 — a user who asks to research a
topic gets a search even if memory happens to be confident. Rules 1, 3 and 4
always apply.

## Provenance and origins

Every fact carries an `origin` from the closed vocabulary
`user_text`, `chat`, `continuous_stream`, `web`. Web-sourced facts additionally
carry `source_url`, `source_query`, and the retrieval timestamp (also exposed as
`retrieved_at`). Recall results report all four fields, which is how a
web-sourced fact stays distinguishable from a taught one after a restart.

The origin is validated on insert: an unknown origin raises a `ValueError` naming
the value instead of quietly creating a new bucket.

## Safety limits

* Schemes are restricted to `http` and `https`.
* Redirects are followed at most twice, and the host check is re-applied at every
  hop.
* Hosts that resolve to loopback, private, link-local, reserved, unspecified or
  multicast addresses are refused. The fetcher's own tests use an `http.server`
  on loopback inside a context-scoped exemption; production code never enters it.
* Every request has a timeout and a response byte cap; the cap truncates rather
  than failing, and truncated text is still usable distillation input.
* Retrieved text is treated as data, never as instructions. Propositions are
  filtered for length and completeness before they are taught.
* Nothing is stored when nothing usable is found, and a placeholder fact is never
  written.

## Honest failure

Provider and transport failures are returned as structured results, not raised
into chat. A failed research yields the ordinary no-information fallback rather
than an ungrounded generated answer. The fact count is unchanged after an empty
search.
