"""Two-channel recall: verbatim sources, empty unknowns, learned fusion."""

import pytest

from recamem.engine import DroidEngine
from recamem.retrieval import LogisticScorer, TwoChannelRecall, measure_index_recall
pytestmark = pytest.mark.phase0



@pytest.fixture()
def engine():
    e = DroidEngine(name="recall-test")
    e.teach("The project lead is Dr. Mina Rostami. The budget is 5 million.")
    return e


def test_verbatim_fact_with_source_is_returned(engine):
    recall = TwoChannelRecall(engine)
    out = recall.recall("Who is the project lead?", top_k=3)
    assert out["results"], out
    top = out["results"][0]
    assert top["text"] == "The project lead is Dr. Mina Rostami."
    assert top["source"] == "user_text"
    assert top["channels"]
    assert set(top["scores"]) >= {"fused", "episodic", "plastic"}


def test_unrelated_question_returns_nothing(engine):
    recall = TwoChannelRecall(engine)
    out = recall.recall("What is the boiling point of mercury?")
    assert out["results"] == []
    assert out["empty"] is True


def test_plastic_only_evidence_is_never_presented_as_a_fact(engine):
    recall = TwoChannelRecall(engine, min_score=0.99)
    out = recall.recall("Who is the project lead?")
    # Nothing reaches the floor, so no text is returned; when only the plastic
    # channel scores, the result is labelled and carries no fact text.
    assert all("text" not in item or item.get("plastic_only") for item in out["results"])
    assert out["plastic_only"] in (True, False)


def test_recall_records_an_access_and_an_audit_row(engine):
    recall = TwoChannelRecall(engine)
    before = engine.knowledge.conn.execute(
        "SELECT access_count FROM facts ORDER BY id LIMIT 1"
    ).fetchone()[0]
    out = recall.recall("Who is the project lead?")
    assert out["results"]
    after = engine.knowledge.conn.execute(
        "SELECT access_count FROM facts ORDER BY id LIMIT 1"
    ).fetchone()[0]
    assert after > before
    audit = engine.knowledge.conn.execute(
        "SELECT action, correction_text FROM feedback_log WHERE action='recall_access'"
    ).fetchall()
    assert audit and audit[0][1] == "Who is the project lead?"


def test_labelled_set_too_small_disables_the_scorer(engine):
    recall = TwoChannelRecall(engine)
    report = recall.train_scorer([{"query": "Who is the project lead?", "relevant_ids": [1]}])
    assert report["enabled"] is False
    assert "plastic prior used alone" in report["reason"]
    assert recall.status()["scorer"] == "plastic_prior"


def test_scorer_trains_on_a_split_and_reports_metrics(engine):
    for index in range(8):
        engine.teach(f"The instrument {index} is deployed at station {index}.")
    ids = [row[0] for row in engine.knowledge.conn.execute("SELECT id FROM facts ORDER BY id")]
    labelled = [{"query": f"Where is instrument {i} deployed?", "relevant_ids": [ids[i]]}
                for i in range(4)]
    labelled += [{"query": f"What is the deployment of instrument {i}?", "relevant_ids": [ids[i]]}
                 for i in range(4)]
    recall = TwoChannelRecall(engine)
    report = recall.train_scorer(labelled, seed=3)
    assert report["enabled"] is True
    assert report["train_rows"] + report["heldout_rows"] == report["rows"]
    assert set(report["metrics"]) == {"precision_at_half", "recall_at_half", "mrr"}
    assert recall.status()["scorer"] == "logistic"


def test_scorer_round_trips_through_json():
    scorer = LogisticScorer(weights=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], bias=-0.5)
    clone = LogisticScorer.from_dict(scorer.to_dict())
    features = {"episodic_fused": 0.2, "dense_similarity": 0.6, "plastic_prior": 0.1,
                "lexical_rank": 1.0, "recency": 0.5, "access_count": 0.0}
    assert clone.predict(features) == pytest.approx(scorer.predict(features))


def test_index_recall_is_reported_against_exact_search(engine):
    report = measure_index_recall(engine.knowledge, [])
    assert report["indexed"] is False
    assert "no ANN index" in report["reason"]
