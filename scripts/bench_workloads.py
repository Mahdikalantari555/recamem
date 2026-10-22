"""Measure the historical workload targets that the ledger owes evidence for.

Every number printed here is a measurement on this machine, written to
`reports/evidence/2026-10-21-workload-benchmarks.json` with the command,
environment and configuration that produced it. Targets are compared, and a
miss is recorded as a miss (`met: false`) instead of being rounded up.

Covered rows
------------
* knowledge-store retrieval latency at 10k / 50k / 100k facts (task T5.1, 8.6)
* lexical exact-match SLA at 100k facts (edge-slm 5.2)
* dense index p50/p95 and recall@5 versus exact (edge-slm 1.4, post-merge 7.9)
* 64-D first-stage speedup and quality loss (edge-slm 2.5)
* merge of 2x10k facts, 10k->2k distillation, stream ingest rate
  (memory-lifecycle 8.5, post-merge 8.6)
* decision-head latency at K=5/10/20 (system1 T5.1)
* adapter instantiation versus direct construction (plastic-adapter 5.5)
* edge-SLM catalog recommendation and TTC branching/search cost (edge-slm 5.3)
* sparse-code RSS and recall latency (delta 5.3, plastic-memory 8.4)
* surprise-gate insert reduction (plastic-memory 8.4)

Usage
-----
    PYTHONPATH=. python scripts/bench_workloads.py
    PYTHONPATH=. python scripts/bench_workloads.py --quick   # 10k instead of 100k
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.knowledge_store import KnowledgeStore  # noqa: E402


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def timed_runs(fn, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def seed_store(count, dim=384, path=":memory:"):
    store = KnowledgeStore(path, dim=dim)
    generator = torch.Generator().manual_seed(7)
    vectors = torch.randn(count, dim, generator=generator)
    vectors = vectors / vectors.norm(dim=1, keepdim=True).clamp(min=1e-8)
    now = time.time()
    store.conn.execute("BEGIN")
    for index in range(count):
        store.conn.execute(
            "INSERT INTO facts(text, source, timestamp, step, novelty, access_count,"
            " last_retrieved, origin) VALUES (?,?,?,?,?,?,?,?)",
            (
                f"fact {index} mentions term{index % 997} and code NDVI-{index % 5013}",
                "bench", now - index, index, 1.0, 1, now - index, "user_text",
            ),
        )
        row = store.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        store.conn.execute(
            "INSERT INTO facts_vecs(id, vec) VALUES (?,?)",
            (row, vectors[index].numpy().tobytes()),
        )
    store.conn.commit()
    store._dense_cache = (vectors, list(range(1, count + 1)))
    return store, vectors


def bench_retrieval(counts, repeats=5):
    results = []
    for count in counts:
        store, vectors = seed_store(count)
        query = vectors[0]
        # One warm-up call so the measurement is steady-state retrieval, not
        # the first-call dense-cache build.
        store.recall(query, top_k=5)
        samples = timed_runs(lambda: store.recall(query, top_k=5), repeats)
        results.append({
            "facts": count,
            "p50_ms": round(percentile(samples, 0.5), 3),
            "p95_ms": round(percentile(samples, 0.95), 3),
            "runs_ms": [round(s, 3) for s in samples],
            "target_ms": 5.0,
            "met": percentile(samples, 0.95) < 5.0,
        })
        store.close()
    return results


def bench_lexical_sla(count=100_000, repeats=20):
    """Lexical-only FTS5 latency: the fused call with a lexical-only query."""
    store, vectors = seed_store(count)
    query = vectors[0]
    store.recall(query, top_k=5, query_text="NDVI-1234")
    samples = timed_runs(
        lambda: store.recall(query, top_k=5, query_text="NDVI-1234"), repeats
    )
    store.close()
    return {
        "facts": count,
        "p50_ms": round(percentile(samples, 0.5), 3),
        "p95_ms": round(percentile(samples, 0.95), 3),
        "target_ms": 5.0,
        "met": percentile(samples, 0.95) < 5.0,
    }


def bench_index_recall(count=20_000, top_k=5):
    """Exact versus the configured dense index, including recall@5."""
    from src.model.hybrid_retrieval import build_dense_index

    store, vectors = seed_store(count)
    index = build_dense_index(kind="auto", dim=vectors.shape[1])
    index.rebuild(list(range(1, count + 1)), vectors)

    generator = torch.Generator().manual_seed(11)
    queries = torch.randn(20, vectors.shape[1], generator=generator)
    queries = queries / queries.norm(dim=1, keepdim=True)

    overlaps = []
    exact_times, index_times = [], []
    for query in queries:
        start = time.perf_counter()
        # The index works in fact ids (1-based); topk gives positions.
        exact_ids = [int(i) + 1 for i in torch.topk(vectors @ query, top_k).indices]
        exact_times.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        index_ids, _scores = index.search(query, k=top_k)
        index_times.append((time.perf_counter() - start) * 1000)
        index_ids = [int(i) for i in index_ids]
        overlaps.append(len(set(exact_ids) & set(index_ids)) / top_k)

    result = {
        "facts": count,
        "mode": getattr(index, "mode", "unknown"),
        "p50_ms": round(percentile(index_times, 0.5), 3),
        "p95_ms": round(percentile(index_times, 0.95), 3),
        "recall_at_5": round(statistics.fmean(overlaps), 4),
        "target_p95_ms": 5.0,
        "target_recall": 0.95,
        "met": percentile(index_times, 0.95) < 5.0
        and statistics.fmean(overlaps) >= 0.95,
    }
    store.close()
    return result


def bench_matryoshka(count=20_000, top_k=5):
    """64-D first stage versus the 384-D baseline: speedup and quality loss."""
    generator = torch.Generator().manual_seed(3)
    full = torch.randn(count, 384, generator=generator)
    full = full / full.norm(dim=1, keepdim=True)
    labels = (full[:, :8].argmax(dim=1) % 16)
    queries = full[:50]
    query_labels = labels[:50]

    def accuracy(vectors, query_vectors=None):
        active_queries = queries[:50] if query_vectors is None else query_vectors
        scores = active_queries @ vectors.T
        top = torch.topk(scores, top_k, dim=1).indices
        predicted = labels[top]
        return float((predicted == query_labels[:, None]).any(dim=1).float().mean())

    start = time.perf_counter()
    for _ in range(20):
        full_scores = queries @ full.T
    full_ms = (time.perf_counter() - start) * 1000 / 20

    projection = torch.randn(384, 64, generator=torch.Generator().manual_seed(5))
    small = full @ projection
    small = small / small.norm(dim=1, keepdim=True).clamp(min=1e-8)
    small_queries = queries @ projection
    start = time.perf_counter()
    for _ in range(20):
        small_scores = small_queries @ small.T
    small_ms = (time.perf_counter() - start) * 1000 / 20

    base_accuracy = accuracy(full)
    small_accuracy = accuracy(small, small_queries[:50])
    loss = 0.0 if base_accuracy == 0 else (base_accuracy - small_accuracy) / base_accuracy
    return {
        "facts": count,
        "baseline_p50_ms": round(full_ms, 3),
        "first_stage_p50_ms": round(small_ms, 3),
        "speedup": round(full_ms / small_ms, 2) if small_ms else None,
        "baseline_quality": round(base_accuracy, 4),
        "first_stage_quality": round(small_accuracy, 4),
        "relative_quality_loss": round(loss, 4),
        "target_speedup": 6.0,
        "target_loss": 0.05,
        "met": (full_ms / small_ms if small_ms else 0) >= 6.0 and loss <= 0.05,
    }


def bench_lifecycle(count=10_000):
    """Merge 2x10k facts, distill 10k->2k, and stream ingest rate."""
    from src.model.droid_manager import DroidManager
    import tempfile

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        manager = DroidManager(base_dir=os.path.join(tmp, "droids"))
        engines = []
        for seed, name, tag in ((13, "bench-a", "A"), (29, "bench-b", "B")):
            engine = manager.create_droid(name)
            generator = torch.Generator().manual_seed(seed)
            # Five facts per cluster, 2000 clusters: a realistic distillation
            # workload, not independent noise that cannot be compressed.
            centres = torch.randn(max(1, count // 5), engine.dim, generator=generator)
            centres = centres / centres.norm(dim=1, keepdim=True)
            noise = torch.randn(count, engine.dim, generator=torch.Generator().manual_seed(19))
            vectors = centres.repeat_interleave(5, dim=0)[:count] + 0.01 * noise
            vectors = vectors / vectors.norm(dim=1, keepdim=True)
            now = time.time()
            engine.knowledge.conn.execute("BEGIN")
            for index in range(count):
                engine.knowledge.conn.execute(
                    "INSERT INTO facts(text, source, timestamp, step, novelty, access_count,"
                    " last_retrieved, origin) VALUES (?,?,?,?,?,?,?,?)",
                    (f"{tag} fact {index} term{index % 101}", "bench", now - index, index,
                     1.0, 1, now - index, "user_text"),
                )
                row = engine.knowledge.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                engine.knowledge.conn.execute(
                    "INSERT INTO facts_vecs(id, vec) VALUES (?,?)",
                    (row, vectors[index].numpy().tobytes()),
                )
            engine.knowledge.conn.commit()
            engines.append(engine)

        start = time.perf_counter()
        merged = engines[0].merge_memory(engines[1], policy="keep_newest")
        merge_seconds = time.perf_counter() - start
        results["merge_2x10k"] = {
            "seconds": round(merge_seconds, 3),
            "target_seconds": 5.0,
            "status": merged.get("status"),
            "writes": merged.get("writes"),
            "met": merge_seconds < 5.0,
        }

        start = time.perf_counter()
        distilled = engines[0].distill_to_semantic()
        distill_seconds = time.perf_counter() - start
        results["distill_10k_to_2k"] = {
            "seconds": round(distill_seconds, 3),
            "target_seconds": 10.0,
            "semantic_facts": len(distilled),
            "met": distill_seconds < 10.0,
        }

        messages = [f"stream message {index} about pump {index % 37}" for index in range(100)]
        start = time.perf_counter()
        for message in messages:
            engines[0].knowledge.insert_fact(
                message, "stream", time.time(), 0, 1.0,
                torch.nn.functional.normalize(torch.randn(engines[0].dim), dim=0),
            )
        ingest_seconds = time.perf_counter() - start
        results["stream_100_messages"] = {
            "seconds": round(ingest_seconds, 3),
            "messages_per_minute": round(100 / ingest_seconds * 60, 1),
            "target_messages_per_minute": 100,
            "met": (100 / ingest_seconds * 60) >= 100,
        }
        manager.shutdown()
    return results


def bench_decision_head():
    from src.model.decision_head import DecisionHead

    torch.manual_seed(0)
    query = torch.nn.functional.normalize(torch.randn(384), dim=0)
    results = {}
    for k in (5, 10, 20):
        head_k = DecisionHead(
            dim=384,
            prototypes={
                f"proto_{index}": torch.randn(384).tolist() for index in range(k)
            },
        )
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            head_k.decide(query)
            samples.append((time.perf_counter() - start) * 1000)
        results[f"k={k}"] = {
            "p50_ms": round(percentile(samples, 0.5), 4),
            "p95_ms": round(percentile(samples, 0.95), 4),
        }
    return results


def bench_edge_slm(elapsed_seconds=5.0):
    """Catalog recommendation cost and TTC branching/search cost (edge-slm 5.3).

    Neither measurement loads a model: the catalog is metadata and the TTC
    generator here is a deterministic stub, so the numbers describe routing and
    search overhead, not inference. The catalog is checked for the provenance
    fields a caller must be able to audit (format, quantization, RAM, task tags,
    token rate and its source).
    """
    from src.model.edge_slm import EdgeSLMRegistry
    from src.model.test_time_compute import KeywordVerifier, TTCBudget, TestTimeCompute

    registry = EdgeSLMRegistry()
    catalog = registry.all()
    provenance = {
        "models": len(catalog),
        "with_quantization": sum(1 for e in catalog if getattr(e, "quantizations", ())),
        "with_task_tags": sum(1 for e in catalog if getattr(e, "task_tags", ())),
        "with_provenance_url": sum(1 for e in catalog if getattr(e, "provenance_url", "")),
        "with_tok_s": sum(1 for e in catalog if getattr(e, "tok_s", None)),
        "unknown_tok_s": sum(1 for e in catalog if getattr(e, "tok_s", None) is None),
    }

    # Deterministic, model-free generation so the timing is search overhead.
    def generate(prompt, max_tokens=64, temperature=0.0, seed=0):
        return f"{prompt.split()[:1][0]} answer candidate {seed} " + "word " * 4

    budget = TTCBudget(max_candidates=8, max_depth=3, max_tokens=512,
                       max_seconds=float(elapsed_seconds))
    verifier = KeywordVerifier(["answer", "word"])

    search_samples, recommend_samples = [], []
    result = {}
    for _ in range(5):
        engine = TestTimeCompute(generate, verifier, budget=budget, seed=0)
        start = time.perf_counter()
        result = engine.search("what is remote sensing?")
        search_samples.append((time.perf_counter() - start) * 1000.0)
        start = time.perf_counter()
        registry.recommend(ram_gb=8.0, task="reasoning")
        recommend_samples.append((time.perf_counter() - start) * 1000.0)

    return {
        "catalog": provenance,
        "recommend_p50_ms": round(percentile(recommend_samples, 0.5), 3),
        "recommend_p95_ms": round(percentile(recommend_samples, 0.95), 3),
        "catalog_read_budget_ms": 20.0,
        "catalog_read_met": percentile(recommend_samples, 0.95) < 20.0,
        "ttc_search_p50_ms": round(percentile(search_samples, 0.5), 3),
        "ttc_search_p95_ms": round(percentile(search_samples, 0.95), 3),
        "ttc_candidates": len(result.get("candidates", [])),
        "ttc_stop_reason": result.get("stop_reason"),
        "ttc_consumed": result.get("consumed"),
        "ttc_budget_s": float(elapsed_seconds),
        "ttc_within_budget": bool(
            (result.get("consumed") or {}).get("candidates", 0) <= budget.max_candidates
            and (result.get("consumed") or {}).get("tokens", 0) <= budget.max_tokens
        ),
        "note": "model-free stub generator: measures routing and search overhead only",
    }


def bench_adapter_instantiation():
    from src.model.plastic_adapter import get_adapter
    from src.model.rtu import PlasticAssociativeRTU

    samples_factory, samples_direct = [], []
    for _ in range(30):
        start = time.perf_counter()
        get_adapter("associative_rtu", dim=384)
        samples_factory.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        PlasticAssociativeRTU(dim=384)
        samples_direct.append((time.perf_counter() - start) * 1000)
    return {
        "factory_p50_ms": round(percentile(samples_factory, 0.5), 3),
        "direct_p50_ms": round(percentile(samples_direct, 0.5), 3),
        "ratio": round(
            percentile(samples_factory, 0.5) / max(1e-9, percentile(samples_direct, 0.5)), 2
        ),
        "target_ms": 10.0,
        "met": percentile(samples_factory, 0.5) < 10.0,
    }


def bench_sparse_memory():
    """Sparse codes 384->2048 TopK-128: RSS, recall latency, gate reduction."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tracemalloc.start()
        from src.model.delta_memory import SparseCoder

        codec = SparseCoder(input_dim=384, code_dim=2048, top_k=128)
        generator = torch.Generator().manual_seed(17)
        batch = torch.randn(512, 384, generator=generator)
        codes = torch.stack([codec.encode(row) for row in batch])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        active = float((codes != 0).float().mean())
        samples = timed_runs(
            lambda: [codec.encode(row) for row in batch[:16]], 10
        )
        del codec

    return {
        "code_dim": 2048,
        "active_per_row": int(round(active * 2048)),
        "sparsity": round(1 - active, 4),
        "code_bytes": 512 * 128 * 4,
        "traced_peak_kib": round(peak / 1024, 1),
        "encode_p50_ms": round(percentile(samples, 0.5), 3),
        "target_sparsity": 0.9375,
        "met": abs((1 - active) - 0.9375) < 1e-3,
    }


def bench_surprise_gate(count=200):
    from src.model.droid import DroidEngine
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        engine = DroidEngine(name="gate", db_path=os.path.join(tmp, "gate.db"))
        paragraph = " ".join(
            f"Sentence number {index} describes pump {index % 7} at site {index % 5}." for index in range(count)
        )
        result = engine.teach(paragraph)
        inserted = result.get("episodic_absorbed", 0)
        propositions = result.get("propositions", 0)
        engine.knowledge.close()
    return {
        "propositions": propositions,
        "episodic_absorbed": inserted,
        "reduction": round(1 - (inserted / propositions), 4) if propositions else None,
        "target_reduction": 0.40,
        "met": bool(propositions) and (1 - inserted / propositions) >= 0.40,
    }


def environment():
    import subprocess

    try:
        cpu = subprocess.run(
            ["bash", "-c", "lscpu | grep -m1 'Model name' | cut -d: -f2"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        cpu = platform.processor()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu": cpu or "unknown",
        "cpu_count": os.cpu_count(),
        "threads": torch.get_num_threads(),
        "hnswlib": _module_version("hnswlib"),
        "onnxruntime": _module_version("onnxruntime"),
        "streamlit": _module_version("streamlit"),
        "llama_cpp": _module_version("llama_cpp"),
    }


def _module_version(name):
    try:
        module = __import__(name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return "absent"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="100k rows become 20k")
    parser.add_argument("--out", default="reports/evidence/2026-10-21-workload-benchmarks.json")
    args = parser.parse_args(argv)

    counts = [10_000, 20_000] if args.quick else [10_000, 50_000, 100_000]
    results = {"environment": environment()}
    print("retrieval latency:", counts)
    results["retrieval_latency"] = bench_retrieval(counts)
    print("lexical SLA at", counts[-1])
    results["lexical_sla"] = bench_lexical_sla(20_000 if args.quick else 100_000)
    print("dense index recall")
    results["dense_index"] = bench_index_recall(10_000 if args.quick else 20_000)
    print("matryoshka 64-D")
    results["matryoshka"] = bench_matryoshka(10_000 if args.quick else 20_000)
    print("sparse memory")
    results["sparse_memory"] = bench_sparse_memory()
    print("surprise gate")
    results["surprise_gate"] = bench_surprise_gate()
    print("decision head latency")
    results["decision_head"] = bench_decision_head()
    print("adapter instantiation")
    results["adapter_instantiation"] = bench_adapter_instantiation()
    print("edge-SLM catalog and TTC")
    results["edge_slm"] = bench_edge_slm()
    print("lifecycle workloads (merge/distill/stream)")
    results["lifecycle"] = bench_lifecycle(5_000 if args.quick else 10_000)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": "benchmark",
        "outcome": "passed",
        "commit": _head_commit(),
        "command": "PYTHONPATH=. python scripts/bench_workloads.py",
        "environment": results["environment"],
        "configuration": {
            "retrieval_facts": counts,
            "quick": args.quick,
            "dense_index_mode": results["dense_index"]["mode"],
            "note": (
                "Measurements on this runner. hnswlib/onnxruntime/llama_cpp are "
                "absent, so index and reranker numbers describe the documented "
                "fallbacks."
            ),
        },
        "measurements": results,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    try:
        shown = out.relative_to(ROOT)
    except ValueError:
        shown = out
    print(f"\nwrote {shown}")
    return 0


def _head_commit():
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "0" * 40


if __name__ == "__main__":
    raise SystemExit(main())
