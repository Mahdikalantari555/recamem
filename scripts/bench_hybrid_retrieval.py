#!/usr/bin/env python
"""Benchmark dense candidates, reranking and the 64-D Matryoshka path.

Everything here is synthetic and reproducible: seeds, corpus size, query count
and index parameters are recorded in the JSON output.  The script reports the
measured mode (HNSW or exact fallback), so a missing optional dependency can
never be mistaken for an approximate-search result.
"""

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.model.hybrid_retrieval import (
    CrossEncoderReranker,
    DualIndex,
    ExactDenseIndex,
    HNSWIndex,
    MatryoshkaAnchor,
    build_dense_index,
)


def _corpus(count: int, dim: int, seed: int):
    """Clustered corpus; the cluster label is the retrieval relevance label."""
    generator = torch.Generator().manual_seed(seed)
    centers = F.normalize(torch.randn(1000, dim, generator=generator), dim=-1)
    assignments = torch.randint(0, 1000, (count,), generator=generator)
    # Noise scaled by 1/sqrt(dim) so each point stays near its cluster centre.
    noise = torch.randn(count, dim, generator=generator) * (0.5 / math.sqrt(dim))
    vectors = F.normalize(centers[assignments] + noise, dim=-1)
    return vectors, assignments


def _percentiles(samples):
    ordered = sorted(samples)
    def at(p):
        return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]
    return {"mean": statistics.mean(ordered), "p50": at(0.50), "p95": at(0.95)}


def dense_benchmark(count: int, dim: int, queries: int, k: int, seed: int):
    vectors, _labels = _corpus(count, dim, seed)
    generator = torch.Generator().manual_seed(seed + 1)
    query_index = torch.randint(0, count, (queries,), generator=generator)
    queries_t = vectors[query_index]

    exact = ExactDenseIndex(dim)
    exact.rebuild(list(range(count)), vectors)
    exact_samples = []
    exact_truth = []
    for query in queries_t:
        started = time.perf_counter()
        ids, _scores = exact.search(query, k=k)
        exact_samples.append((time.perf_counter() - started) * 1000)
        exact_truth.append(ids)

    hnsw = build_dense_index("auto", dim=dim)
    build_started = time.perf_counter()
    hnsw.rebuild(list(range(count)), vectors)
    build_ms = (time.perf_counter() - build_started) * 1000
    hnsw_samples = []
    recalls = []
    for query, truth in zip(queries_t, exact_truth):
        started = time.perf_counter()
        ids, _scores = hnsw.search(query, k=k)
        hnsw_samples.append((time.perf_counter() - started) * 1000)
        recalls.append(len(set(ids) & set(truth)) / max(1, len(truth)))

    return {
        "corpus_facts": count,
        "dim": dim,
        "queries": queries,
        "k": k,
        "knnlib_mode": hnsw.mode,
        "hnsw_available": getattr(hnsw, "available", False),
        "fallback_reason": getattr(hnsw, "fallback_reason", None),
        "build_ms": build_ms,
        "dense_bytes": int(vectors.numel() * 4),
        "exact_latency_ms": _percentiles(exact_samples),
        "candidate_latency_ms": _percentiles(hnsw_samples),
        "recall_at_k_vs_exact": statistics.mean(recalls),
        "targets": {
            "p95_ms": 5.0,
            "recall_at_5": 0.95,
            "p95_met": _percentiles(hnsw_samples)["p95"] < 5.0,
            "recall_met": statistics.mean(recalls) >= 0.95,
        },
        "note": (
            "Exact fallback measured because hnswlib is unavailable"
            if not getattr(hnsw, "available", False) else
            "HNSW candidates measured against exact MVM truth"
        ),
    }


def matryoshka_benchmark(count: int, dim: int, queries: int, seed: int):
    vectors, labels = _corpus(count, dim, seed + 7)
    anchor = MatryoshkaAnchor(fast_dim=64, full_dim=dim, seed=seed)
    dual = DualIndex(anchor)
    dual.rebuild(list(range(count)), vectors)

    generator = torch.Generator().manual_seed(seed + 8)
    query_positions = torch.randint(0, count, (queries,), generator=generator).tolist()

    def label_precision_at_5(ids, query_label):
        if not ids:
            return 0.0
        return sum(1 for fact_id in ids if int(labels[fact_id]) == int(query_label)) / len(ids)

    fast_samples, full_samples, fast_quality, full_quality = [], [], [], []
    for position in query_positions:
        query = vectors[position]
        started = time.perf_counter()
        fast = dual.search(query, k=5, stage="fast")
        fast_samples.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        full = dual.search(query, k=5, stage="full")
        full_samples.append((time.perf_counter() - started) * 1000)
        fast_quality.append(label_precision_at_5(fast["ids"], labels[position]))
        full_quality.append(label_precision_at_5(full["ids"], labels[position]))

    fast_p50 = _percentiles(fast_samples)["p50"]
    full_p50 = _percentiles(full_samples)["p50"]
    speedup = full_p50 / max(1e-9, fast_p50)
    fast_mean = statistics.mean(fast_quality)
    full_mean = statistics.mean(full_quality)
    quality_loss = (
        (full_mean - fast_mean) / full_mean if full_mean > 0 else None
    )
    return {
        "corpus_facts": count,
        "fast_dim": 64,
        "full_dim": dim,
        "queries": queries,
        "fast_latency_ms": _percentiles(fast_samples),
        "full_latency_ms": _percentiles(full_samples),
        "fast_path_speedup_p50": speedup,
        "fast_quality_p_at_5": fast_mean,
        "full_quality_p_at_5": full_mean,
        "relative_quality_loss": quality_loss,
        "targets": {
            "speedup": 6.0,
            "relative_quality_loss": 0.05,
            "speedup_met": speedup >= 6.0,
            "quality_met": bool(quality_loss is not None and quality_loss <= 0.05),
        },
        "metric": "cluster-label precision@5 (fast 64-D vs full 384-D)",
        "projection_checksum": anchor.projection_checksum,
    }


def rerank_benchmark(count: int, queries: int, seed: int):
    """Labeled synthetic queries: five relevant facts per query, P@5 measured."""
    generator = torch.Generator().manual_seed(seed + 3)
    facts = []
    for index in range(count):
        facts.append({
            "id": index,
            "text": f"topic{index} records sensor reading {index} for station {index % 37}",
            "fused_score": 1.0 / (1 + index),
        })
    reranker = CrossEncoderReranker(model_name=None, limit=20)

    def precision_at_5(items, relevant_ids):
        return sum(1 for item in items[:5] if item["id"] in relevant_ids) / 5.0

    before, after = [], []
    for _ in range(queries):
        topic = int(torch.randint(0, count, (1,), generator=generator))
        relevant = {topic}
        while len(relevant) < 5:
            relevant.add(int(torch.randint(0, count, (1,), generator=generator)))
        distractors = [i for i in torch.randint(0, count, (40,), generator=generator).tolist()
                       if i not in relevant][:15]
        order = torch.randperm(20, generator=generator).tolist()
        pool = [dict(facts[i]) for i in list(relevant) + distractors]
        candidates = [pool[i] for i in order]
        before.append(precision_at_5(candidates, relevant))
        trace = reranker.rerank(f"topic{topic}", candidates)
        after.append(precision_at_5(trace["results"], relevant))
    before_mean = statistics.mean(before)
    after_mean = statistics.mean(after)
    gain = (after_mean - before_mean) / max(1e-9, before_mean) if before_mean else None
    return {
        "labeled_queries": queries,
        "relevant_per_query": 5,
        "scorer": reranker.scorer,
        "precision_at_5_before": before_mean,
        "precision_at_5_after": after_mean,
        "relative_gain": gain,
        "targets": {"relative_gain": 0.20,
                    "met": bool(gain is not None and gain >= 0.20)},
        "note": "lexical fallback scorer; no cross-encoder model supplied",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=int, default=100000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--queries", type=int, default=25)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="reports/evidence/hybrid-retrieval-100k.json")
    args = parser.parse_args()

    results = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "seed": args.seed,
        "dense": dense_benchmark(args.facts, args.dim, args.queries, 5, args.seed),
        "matryoshka": matryoshka_benchmark(
            args.facts, args.dim, max(10, args.queries // 2), args.seed
        ),
        "reranking": rerank_benchmark(min(args.facts, 2000),
                                      max(20, args.queries * 2), args.seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps({key: results[key] for key in ("dense", "matryoshka", "reranking")},
                     indent=2))
    print(f"Written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
