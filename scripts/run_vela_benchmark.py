#!/usr/bin/env python3
"""Proof-of-concept run of the Vela text-only learning benchmark on RecaMem."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = os.environ.get(
    "RECAMEM_GGUF",
    "/mnt/d/AI/.lmstudio/models/OpenBMB/MiniCPM5-2B-Q4_K_M/MiniCPM5-2B-Q4_K_M.gguf",
)

TRAINING_FACTS = [
    ("T1", "In the fictional Vela greenhouse, every bed has a soil code and a growth phase. Decisions are governed only by the rules recorded for Vela. Do not apply real-world agricultural knowledge to Vela."),
    ("T2", "For soil code Amber in phase 1, irrigation is required when the moisture index is below 30. At 30 or above, irrigation is not required."),
    ("T3", "For soil code Amber in phase 2, irrigation is required when the moisture index is below 36. At 36 or above, irrigation is not required."),
    ("T4", "For soil code Blue in phase 1, irrigation is required when the moisture index is below 22. At 22 or above, irrigation is not required."),
    ("T5", "For soil code Blue in phase 2, irrigation is required when the moisture index is below 26. At 26 or above, irrigation is not required."),
    ("T6", "A moisture index exactly equal to the threshold means irrigation is not required."),
    ("T7", "A record is valid only when it names a soil code, growth phase, and moisture index. If one of these fields is missing, do not guess the missing value. State that the case is underspecified."),
    ("T8", "A later correction explicitly marked as a Vela policy update replaces the earlier rule for the same soil code and phase. Keep a record of the superseded rule; do not silently merge the two rules."),
    ("T9", "Vela policy update, version 2: for soil code Amber in phase 2, the threshold is now 40, not 36. This update applies to cases received after version 2. The Amber phase 2 rule in version 1 is superseded."),
    ("T10", "For soil code Blue in phase 1, the threshold remains 22 in policy version 2. No update has been made to this rule."),
]

FACT_BY_ID = {fid: text for fid, text in TRAINING_FACTS}

VALIDATION = [
    ("V1", "Vela case: soil Amber, phase 1, moisture index 29. Should irrigation occur?", "IRRIGATE", ["T2"]),
    ("V2", "Vela case: soil Amber, phase 1, moisture index 30. Should irrigation occur?", "DO_NOT_IRRIGATE", ["T2", "T6"]),
    ("V3", "Vela case: soil Blue, phase 2, moisture index 25. Should irrigation occur?", "IRRIGATE", ["T5"]),
    ("V4", "Vela case: soil Amber, phase 2, moisture index 38. Use the latest policy version. Should irrigation occur?", "IRRIGATE", ["T9"]),
    ("V5", "Vela case: soil Amber, phase 2, moisture index 40. Use the latest policy version. Should irrigation occur?", "DO_NOT_IRRIGATE", ["T6", "T9"]),
    ("V6", "Vela case: soil Blue, phase 1, moisture index 21. Use policy version 2. Should irrigation occur?", "IRRIGATE", ["T10"]),
    ("V7", "Vela case: soil Amber, moisture index 28. The growth phase is not provided. Should irrigation occur?", "INSUFFICIENT_INFORMATION", ["T7"]),
    ("V8", "A note says: \"Amber, phase 2, threshold 36.\" A later policy update says: \"Amber, phase 2, threshold 40.\" For a new case with moisture index 38, which threshold applies?", "IRRIGATE", ["T8", "T9"]),
    ("V9", "Vela case: soil Blue, phase 2, moisture index 26. Return the decision and the rule IDs used.", "DO_NOT_IRRIGATE", ["T5", "T6"]),
    ("V10", "A user asks: \"Does the Amber phase 2 threshold of 40 also apply to Blue phase 2?\" Answer using only the Vela rules.", "INSUFFICIENT_INFORMATION", ["T5"]),
    ("V11", "Two records arrive for the same new Vela case: Record A: Amber, phase 2, moisture index 38, policy version 1. Record B: Amber, phase 2, moisture index 38, policy version 2. What decision applies to a case received after version 2?", "IRRIGATE", ["T9"]),
    ("V12", "Vela case: soil Amber, phase 2, moisture index 41. Use policy version 2. Should irrigation occur?", "DO_NOT_IRRIGATE", ["T9"]),
]

DECISIONS = ["IRRIGATE", "DO_NOT_IRRIGATE", "INSUFFICIENT_INFORMATION", "CONFLICT_REQUIRES_CLARIFICATION"]

CATEGORIES = {
    "Recall and boundaries": ["V1", "V2", "V9"],
    "Compositional application": ["V3", "V4", "V5", "V12"],
    "Online update": ["V4", "V5", "V8", "V11"],
    "Retention / interference": ["V6"],
    "Safe abstention": ["V7", "V10"],
    "Traceability": ["V9", "V8"],
}

PROMPT_TEMPLATE = (
    "Reply with one token only: IRRIGATE, DO_NOT_IRRIGATE, INSUFFICIENT_INFORMATION, or CONFLICT_REQUIRES_CLARIFICATION.\n"
    "Rules:\n{facts}\n"
    "Query: {query}\n"
    "Answer:"
)


def extract_decision(text: str):
    if not text:
        return None
    up = text.upper()
    if "CONFLICT" in up:
        return "CONFLICT_REQUIRES_CLARIFICATION"
    if "INSUFFICIENT" in up:
        return "INSUFFICIENT_INFORMATION"
    if "DO_NOT" in up or "DO NOT" in up or "NOT_REQUIRED" in up or "NOT REQUIRED" in up or "SHOULD NOT" in up or "NO IRRI" in up:
        return "DO_NOT_IRRIGATE"
    if "IRRIGATE" in up or "SHOULD OCCUR" in up or "YES" in up or "REQUIRED" in up:
        return "IRRIGATE"
    return None


def ordinary_retrieve(facts, query, k=4):
    qtoks = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = []
    for f in facts:
        ftoks = set(re.findall(r"[a-z0-9]+", f.lower()))
        inter = len(qtoks & ftoks)
        union = len(qtoks | ftoks) or 1
        scored.append((inter / union, f))
    scored.sort(reverse=True)
    return [f for _, f in scored[:k]]


def run_condition(cond, facts_list, engine=None, backend=None, oracle_map=None):
    results = []
    for vid, query, expected, ev in VALIDATION:
        print(f"  {cond} {vid}", flush=True)
        if cond == "A":
            facts = []
        elif cond == "B":
            facts = ordinary_retrieve(facts_list, query, k=4)
        elif cond in ("C", "D"):
            hits = engine.recall(query, top_k=4, threshold=0.3)
            facts = [h["text"] for h in hits]
        elif cond == "E":
            facts = [FACT_BY_ID[eid] for eid in ev if eid in FACT_BY_ID]
        else:
            facts = []
        prompt = PROMPT_TEMPLATE.format(facts="\n".join(f"- {f}" for f in facts) if facts else "(none)", query=query)
        t0 = time.perf_counter()
        try:
            out = backend.generate(prompt, max_tokens=4, temperature=0.0, top_p=0.9)
        except Exception as e:
            out = f"[ERROR] {e}"
        latency = time.perf_counter() - t0
        decision = extract_decision(out)
        correct = decision == expected
        results.append({
            "id": vid,
            "query": query,
            "expected": expected,
            "decision": decision,
            "correct": correct,
            "latency_s": round(latency, 3),
            "retrieved_facts": facts,
            "raw_output": out[:500],
        })
    return results


def score_categories(results):
    by_id = {r["id"]: r for r in results}
    cats = {}
    for cat, ids in CATEGORIES.items():
        correct = sum(1 for vid in ids if by_id[vid]["correct"])
        cats[cat] = {"correct": correct, "total": len(ids), "accuracy": correct / len(ids)}
    total_correct = sum(1 for r in results if r["correct"])
    cats["TOTAL"] = {"correct": total_correct, "total": len(results), "accuracy": total_correct / len(results)}
    return cats


def main():
    from recamem.gguf import GgufBackend

    if not Path(MODEL_PATH).exists():
        print(f"BLOCKED: model not found at {MODEL_PATH}")
        sys.exit(1)

    backend = GgufBackend(model_path=MODEL_PATH, n_ctx=1024, n_gpu_layers=0, verbose=False, n_threads=16, n_batch=512)
    all_facts_text = [t for _, t in TRAINING_FACTS]

    report = {
        "model_path": MODEL_PATH,
        "commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "conditions": {},
    }

    # Condition A
    print("Running condition A...")
    res_a = run_condition("A", all_facts_text, backend=backend)
    report["conditions"]["A"] = {"results": res_a, "categories": score_categories(res_a)}

    # Condition B
    print("Running condition B...")
    res_b = run_condition("B", all_facts_text, backend=backend)
    report["conditions"]["B"] = {"results": res_b, "categories": score_categories(res_b)}

    # Condition C
    print("Running condition C...")
    from recamem.engine import DroidEngine
    engine_c = DroidEngine(name="vela-c", gguf_backend=backend)
    for _, text in TRAINING_FACTS:
        engine_c.teach(text, source="vela")
    res_c = run_condition("C", all_facts_text, engine=engine_c, backend=backend)
    report["conditions"]["C"] = {"results": res_c, "categories": score_categories(res_c), "rtu_enabled": True}

    # Condition D (RTU frozen: skip plastic updates)
    print("Running condition D...")
    import recamem.engine as _eng
    class _NoOpUpdate:
        applied_lr = 0.0
        scheduling_supported = True
    _orig_apply = _eng.apply_plastic_update
    _eng.apply_plastic_update = lambda *a, **k: _NoOpUpdate()
    engine_d = DroidEngine(name="vela-d", gguf_backend=backend)
    for _, text in TRAINING_FACTS:
        engine_d.teach(text, source="vela")
    _eng.apply_plastic_update = _orig_apply
    res_d = run_condition("D", all_facts_text, engine=engine_d, backend=backend)
    report["conditions"]["D"] = {"results": res_d, "categories": score_categories(res_d), "rtu_enabled": False}

    # Condition E
    print("Running condition E...")
    res_e = run_condition("E", all_facts_text, backend=backend)
    report["conditions"]["E"] = {"results": res_e, "categories": score_categories(res_e)}

    out_path = ROOT / "reports" / f"vela-poc-{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Report written: {out_path}")
    for cid in ["A", "B", "C", "D", "E"]:
        tot = report["conditions"][cid]["categories"]["TOTAL"]
        print(f"Condition {cid}: {tot['correct']}/{tot['total']} = {tot['accuracy']:.3f}")


def _git_commit():
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()