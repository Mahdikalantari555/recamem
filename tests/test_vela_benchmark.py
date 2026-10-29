import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_vela_benchmark import extract_decision, ordinary_retrieve, VALIDATION, CATEGORIES, score_categories


def test_extract_decision_basic():
    assert extract_decision("The answer is IRRIGATE.") == "IRRIGATE"
    assert extract_decision("DO_NOT_IRRIGATE now") == "DO_NOT_IRRIGATE"
    assert extract_decision("no token here") is None


def test_ordinary_retrieve_returns_k():
    facts = ["For soil code Amber in phase 1, irrigation below 30.",
             "For soil code Blue in phase 2, irrigation below 26."]
    out = ordinary_retrieve(facts, "soil Blue phase 2 moisture 25", k=1)
    assert len(out) == 1
    assert "Blue" in out[0]


def test_score_categories_shape():
    results = [{"id": vid, "correct": True} for vid, *_ in VALIDATION]
    cats = score_categories(results)
    assert "TOTAL" in cats
    assert cats["TOTAL"]["correct"] == len(VALIDATION)


def test_policy_update_supersedes_old_rule():
    from recamem.engine import DroidEngine
    engine = DroidEngine(name="vela-test-supersede")
    engine.teach("For soil code Amber in phase 2, irrigation is required when the moisture index is below 36. At 36 or above, irrigation is not required.", source="vela")
    engine.teach("Vela policy update, version 2: for soil code Amber in phase 2, the threshold is now 40, not 36. This update applies to cases received after version 2. The Amber phase 2 rule in version 1 is superseded.", source="vela")
    trail = engine.knowledge.get_audit_trail()
    old = [t for t in trail if "below 36" in t["text"]]
    assert old and old[0]["superseded"] is True
    hits = engine.recall("Amber phase 2 threshold", top_k=10, threshold=0.0)
    assert all("below 36" not in h["text"] for h in hits)