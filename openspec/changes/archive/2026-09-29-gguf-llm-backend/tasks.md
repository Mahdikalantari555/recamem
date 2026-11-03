# Tasks

## 1. GGUF Backend Core

- [x] 1.1 Add `llama-cpp-python` to `requirements.txt` as optional extra `[gguf]`
- [x] 1.2 Create `src/model/gguf_backend.py` with `GgufBackend` class wrapping `llama_cpp.Llama`
- [x] 1.3 Implement `__init__(model_path=None, repo_id=None, filename=None, n_ctx=4096, n_gpu_layers=-1, **params)`
- [x] 1.4 Implement `generate(prompt, max_tokens=512, temperature=0.7, top_p=0.9, stream=False, callback=None)`
- [x] 1.5 Implement `tokenize(text)`, `detokenize(tokens)`, `get_kv_cache()`, `set_kv_cache(kv)`
- [x] 1.6 Implement `download_model(repo_id, filename, local_dir)` using `huggingface_hub`
- [x] 1.7 Verify: `tests/test_gguf_backend.py` loads model, generates, streams

## 2. Memory Injection Strategies

- [x] 2.1 Create `src/model/memory_injection.py` with three strategy functions
- [x] 2.2 Implement `inject_prompt(query, facts, max_context_tokens)` → formatted prompt string
- [x] 2.3 Implement `inject_hidden(backend, query, h_trace, layer_idx)` → forward hook adding projected $h_t$
- [x] 2.4 Implement `inject_kv_cache(backend, facts, query)` → encode facts, prime KV cache
- [x] 2.5 Implement `select_strategy(fact_count, query_type, profile_config)` → strategy name
- [x] 2.6 Verify: unit tests for each strategy with mock backend

## 3. Confidence Estimation

- [x] 3.1 Create `src/model/confidence.py` with `estimate_confidence()` and components
- [x] 3.2 Implement `retrieval_confidence(similarity, surprise, health_metrics)` → float [0,1]
- [x] 3.3 Implement `generation_confidence(token_logprobs, self_consistency, entropy)` → float [0,1]
- [x] 3.4 Implement `fuse_confidence(Cr, Cg, weights)` → harmonic mean
- [x] 3.5 Implement `confidence_tier(confidence)` → "high"/"medium"/"low"
- [x] 3.6 Verify: unit tests with known inputs produce expected outputs

## 4. DroidEngine Integration — Dual-Mode Generation

- [x] 4.1 Update `DroidEngine.__init__` to accept optional `gguf_backend` and `gguf_config`
- [x] 4.2 Add `DroidEngine.generate_with_memory(query, strategy="auto")` using injection + GGUF
- [x] 4.4 Add `DroidEngine.estimate_confidence(retrieval_conf, generation_conf)` 
- [x] 4.5 Add `DroidEngine.decide_path(query)` using System-1 `DecisionHead` (fast_recall vs slow_synthesis)
- [x] 4.6 Update `chat()` to call `decide_path()`, route to `recall()` (fast) or `generate_with_memory()` (slow)
- [x] 4.7 Add `/fast` and `/slow` chat command overrides
- [x] 4.8 Verify: integration test shows fast path <50ms, slow path grounded in facts

## 5. Profile & Manager Integration

- [x] 5.1 Update `DroidManager.create_droid()` to accept optional `gguf_model` config
- [x] 5.2 Update `config.json` schema: `gguf_model_path`, `gguf_repo_id`, `gguf_filename`, `gguf_params`
- [x] 5.3 Add `DroidManager.download_model(name, repo_id, filename)` downloading to profile dir
- [x] 5.4 Update `save_profile()`/`load_profile()` to persist GGUF config
- [x] 5.5 Verify: manager test creates droid with GGUF, downloads model, generates

## 6. CLI & MCP Integration

- [x] 6.1 Add `download-model` subcommand to `src/cli.py`
- [x] 6.2 Add `--mode fast|slow|auto` to `chat` subcommand
- [x] 6.3 Add `generate` subcommand for direct LLM generation with memory
- [x] 6.4 Add MCP tool `generate_with_memory` exposing slow path
- [x] 6.5 Verify: CLI commands work; MCP tool returns structured response

## 7. Tests & Validation

- [x] 7.1 Create `tests/test_gguf_backend.py` (requires model fixture or skip if no model)
- [x] 7.2 Create `tests/test_memory_injection.py` with mock backend
- [x] 7.3 Create `tests/test_confidence.py`
- [x] 7.4 Extend `tests/test_droid_engine.py` for dual-mode, injection, confidence
- [x] 7.5 Run full test suite: `PYTHONPATH=. /home/asus/miniforge3/envs/ai/bin/pytest tests -v`
- [x] 7.6 Benchmark: fast path <50ms, slow path 500–2000ms, memory injection adds <100ms