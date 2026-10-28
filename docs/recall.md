# Recall

Recall combines two channels:

* **episodic** — the SQLite store returns verbatim facts, with their source and
  provenance, ranked by FTS5 BM25 fused with dense cosine (RRF).
* **plastic** — the associative adapter's read for the query embedding gives a
  per-fact prior (`cos(read(query), fact_vector)`).

`recamem.retrieval.TwoChannelRecall.recall(query, top_k=3)` returns

```json
{"query": "...", "results": [{"id": 1, "text": "...", "source": "...",
  "scores": {"fused": 0.42, "episodic": 0.03, "plastic": 0.0, "dense": 0.61},
  "channels": ["episodic"]}], "plastic_only": false, "scorer": "plastic_prior"}
```

* Every returned item carries its source; nothing is synthesised.
* When no item reaches `min_score` (default 0.30 on the fused score) the result
  is empty — an unrelated question never returns the nearest fact.
* If only the plastic channel scores, the result is labelled `plastic_only` and
  no text is presented as a stored fact.
* A successful recall increments the fact's access count, reinforces its
  retention and writes a `recall_access` row to the feedback log.

## Learned fusion

`train_scorer(labelled)` builds candidate features
(`episodic_fused`, `dense_similarity`, `plastic_prior`, `lexical_rank`,
`recency`, `access_count`), splits them into train/held-out with a seed, and fits
a logistic scorer. The report records `train_rows`, `heldout_rows` and held-out
`precision_at_half`, `recall_at_half`, `mrr`. Below `MIN_TRAIN_ROWS +
MIN_HELDOUT_ROWS` (or with one class present) the scorer is disabled, the
plastic prior is used alone and the reason is reported.

## ANN with an exact baseline

`measure_index_recall(store, queries, k=5)` compares the configured ANN index
against the exact dense search on the same store and reports `recall_at_k`. An
index latency figure is never reported without it.
