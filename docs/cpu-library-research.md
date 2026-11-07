# CPU libraries and Rust: decision research

Reviewed 2026-11-07. These are candidates, not integrations, benchmark results,
or endorsements; verify current licenses, security advisories and versions when
actually adopting them. The requested target is approximately 16 GB RAM; this
sandbox reports about 3.85 GiB and has no `rustc` or `cargo`.

| Layer | Candidate | Fit and decision |
| --- | --- | --- |
| Agent coordination | [LangGraph](https://github.com/langchain-ai/langgraph) | Durable workflows and interrupt/resume can help long research sessions. It does not by itself provide safe tools, canonical memory or a capable model. Evaluate only when stateful orchestration outweighs dependency cost. |
| Agent tools | [smolagents](https://github.com/huggingface/smolagents) | Model-agnostic tool agents; its code-agent executor has a significant isolation boundary. Upstream explicitly says its local executor is not a security sandbox. Prefer a narrow structured tool interface for now. |
| HTML and JS | [Crawl4AI](https://github.com/unclecode/crawl4ai), [rs-trafilatura](https://github.com/Murrough-Foley/rs-trafilatura-python) | Crawl4AI handles browser-based extraction but adds a browser and security surface; a Rust extractor parses supplied HTML but is neither a browser nor network protection. Compare on representative pages after safety gates. |
| Lexical memory | [Tantivy](https://github.com/quickwit-oss/tantivy) | Rust BM25 index candidate, but this repository already has SQLite FTS5. Tokenization, scores, persistence and rebuilds differ; compare held-out recall, latency, storage and lifecycle, not vendor speed numbers. |
| Vector memory | [USearch](https://github.com/unum-cloud/usearch) | Native HNSW candidate with Python and Rust bindings. Existing exact and optional HNSW baselines already exist; compare approximate recall, add/delete, persistence and memory before changing. |
| CPU inference | [Candle](https://github.com/huggingface/candle), [Optimum ONNX](https://github.com/huggingface/optimum) | Candle supports Rust CPU inference and safetensors; Optimum offers ONNX export/runtime integrations. Architecture support, tokenizer parity, model license and export correctness must be verified for each checkpoint; neither automatically runs our Muir. Quantization is hardware-dependent. |
| Custom Rust bridge | [PyO3](https://github.com/PyO3/pyo3) and [maturin](https://github.com/PyO3/maturin) | Optional Python extension path after profiling identifies a hotspot. Account for tensor conversion, build/wheel support, backwards gradients, state parity and call overhead. No mandatory Rust or GGUF. |

Crawl4AI's [security advisories](https://github.com/unclecode/crawl4ai/security)
and [changelog](https://github.com/unclecode/crawl4ai/blob/main/CHANGELOG.md)
show why patched versions and enforced egress for **all** requests matter.
The existing static fetcher checks DNS before urllib makes its own connection;
this is not a demonstrated defense against DNS rebinding. Optional browser
integration is blocked until the separate research spec's network isolation is
implemented and tested. None of these libraries can guarantee correctness or
prevent memory poisoning from untrusted pages.

## First measured step

Run `PYTHONPATH=. python scripts/profile_cpu_cores.py --threads 1 --output runs/cpu-profile.json`.
It benchmarks seeded synthetic input, not language understanding or matched
training quality. It reports process RSS snapshots, not peak memory; Torch
profiler attributes PyTorch operator time and does not independently measure
Python dispatch. Compare operation breakdown, whole-sequence/step parity and
latency on your own hardware before choosing a native target.

### Local diagnostic baseline

On the sandbox (1 physical core, 3.85 GiB RAM, PyTorch 2.15.0+cu132
running on CPU, Python 3.11.2, one thread), seed 1337, one layer, width 32,
four heads, feed-forward width 64, sequence 32, one warmup and three samples:
C4 (20,232 parameters) had median forward 1.77 ms, streaming sequence 66.85 ms,
forward/backward 12.53 ms; C7 (20,072 parameters) had 7.08 ms, 54.18 ms and
33.42 ms respectively. Max streaming logit error was 2.15e-6 and 1.91e-6.
These tiny synthetic samples are not parameter-matched, representative, quality
or 16 GB target benchmarks; profiler overhead and run-to-run noise apply. C7's
many elementwise calls suggest a candidate for further investigation, not proof
that Rust or a particular optimization will improve end-to-end performance.
Full JSON from this run is available outside the repository at
`/home/user/cpu-profile.json`; rerun the documented command for local results.
