"""Hermetic tests for the search-and-learn policy loop.

Every provider, fetcher and teacher is injected; nothing here touches a socket
or an endpoint.
"""

import pytest

from src.model.search_learn import SearchLearner, normalize_research_config, normalize_topic
from src.model.web_teacher import FetchResult, SearchResult


class FakeKnowledge:
    def __init__(self):
        self.count = 0

    def active_count(self):
        return self.count


class FakeEngine:
    def __init__(self):
        self.knowledge = FakeKnowledge()
        self.taught = []
        self.recall_hits = []

    def teach(self, text, source="user_text", **kwargs):
        self.taught.append((text, source, kwargs))
        self.knowledge.count += 1
        return {"propositions": 1, "episodic_absorbed": 1}

    def recall(self, query, top_k=3, threshold=0.0):
        return self.recall_hits[:top_k]


class FakeProvider:
    name = "fake"

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.calls = []

    def search(self, query, limit=5):
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.results[:limit]


class FakeTeacher:
    def __init__(self, propositions=None, error=None):
        self.propositions = propositions or []
        self.error = error
        self.calls = 0

    def distill_propositions(self, text, timeout=15.0):
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.propositions)


PAGE_TEXT = (
    "Radar is a detection system that uses radio waves. "
    "Lidar measures distance using laser light. "
    "Both systems are used for remote sensing."
)


def _fetcher(text=PAGE_TEXT):
    def fetch(url, **kwargs):
        return FetchResult(ok=True, url=url, text=text, status=200, bytes_read=len(text))
    return fetch


def _learner(provider=None, fetcher=None, teacher=None, clock=None, **config):
    base = {"enabled": True, "provider": "duckduckgo"}
    base.update(config)
    return SearchLearner(
        FakeEngine(), config=base,
        provider=provider if provider is not None else FakeProvider(
            [SearchResult("Radar", "https://example.test/radar", "snippet")]
        ),
        fetcher=fetcher or _fetcher(),
        teacher=teacher,
        clock=clock,
    )


# --------------------------------------------------------------------------- #
# Search decision policy
# --------------------------------------------------------------------------- #
def test_confident_memory_suppresses_the_search():
    learner = _learner()
    decision = learner.should_search("what is radar", hits=[{"similarity": 0.92}])
    assert decision["search"] is False
    assert "confidently" in decision["reason"]
    assert learner.provider.calls == []


def test_low_confidence_triggers_a_search():
    learner = _learner()
    decision = learner.should_search("what is radar", hits=[{"similarity": 0.1}])
    assert decision["search"] is True


def test_a_repeated_topic_inside_the_window_is_not_searched_again():
    now = {"value": 1000.0}
    learner = _learner(clock=lambda: now["value"], suppression_window_seconds=60)
    first = learner.research("What is radar?", force=True)
    assert first["searched"] is True
    assert len(learner.provider.calls) == 1

    now["value"] += 10
    second = learner.should_search("what is radar", hits=[{"similarity": 0.0}])
    assert second["search"] is False
    assert "suppression window" in second["reason"]

    now["value"] += 120
    third = learner.should_search("what is radar", hits=[{"similarity": 0.0}])
    assert third["search"] is True


def test_search_can_be_disabled_and_makes_no_provider_call():
    learner = _learner(enabled=False)
    result = learner.research("radar")
    assert result["searched"] is False
    assert result["available"] is False
    assert "disabled" in result["reason"]
    assert learner.provider.calls == []


def test_provider_selection_failure_is_reported_without_raising():
    learner = SearchLearner(FakeEngine(), config={"enabled": True, "provider": "nope"})
    assert learner.available is False
    result = learner.research("radar")
    assert result["searched"] is False
    assert "nope" in result["reason"]


# --------------------------------------------------------------------------- #
# Acquisition loop
# --------------------------------------------------------------------------- #
def test_acquisition_teaches_facts_with_their_provenance():
    engine = FakeEngine()
    learner = SearchLearner(
        engine,
        config={"enabled": True, "provider": "duckduckgo"},
        provider=FakeProvider([SearchResult("Radar", "https://example.test/radar", "")]),
        fetcher=_fetcher(),
    )
    result = learner.research("radar")
    assert result["searched"] is True
    assert result["facts_learned"] == engine.knowledge.count > 0
    assert result["sources"] == ["https://example.test/radar"]
    for text, source, kwargs in engine.taught:
        assert source == "web"
        assert kwargs["origin"] == "web"
        assert kwargs["source_url"] == "https://example.test/radar"
        assert kwargs["source_query"] == "radar"
        assert kwargs["source_query"] == normalize_topic("radar")


def test_teacher_distillation_is_used_when_configured():
    teacher = FakeTeacher(["Radar uses radio waves.", "Lidar uses laser light."])
    learner = _learner(teacher=teacher)
    result = learner.research("radar")
    assert teacher.calls == 1
    assert result["facts_learned"] == 2
    assert [text for text, _, _ in learner.engine.taught] == teacher.propositions


def test_local_sentence_splitter_is_the_fallback():
    learner = _learner(teacher=None)
    result = learner.research("radar")
    assert result["facts_learned"] >= 2
    assert all(len(text) >= learner.config["min_proposition_chars"]
               for text, _, _ in learner.engine.taught)


def test_a_teacher_outage_falls_back_to_the_local_splitter():
    teacher = FakeTeacher(error=RuntimeError("endpoint down"))
    learner = _learner(teacher=teacher)
    result = learner.research("radar")
    assert teacher.calls == 1
    assert result["facts_learned"] >= 1


def test_a_refused_fetch_does_not_abort_the_rest_of_the_search():
    urls = []

    def fetcher(url, **kwargs):
        urls.append(url)
        if len(urls) == 1:
            return FetchResult(ok=False, url=url, reason="refused: loopback")
        return FetchResult(ok=True, url=url, text=PAGE_TEXT, status=200)

    learner = SearchLearner(
        FakeEngine(),
        config={"enabled": True, "provider": "duckduckgo", "max_pages_per_search": 2},
        provider=FakeProvider([
            SearchResult("bad", "http://127.0.0.1/"),
            SearchResult("good", "https://example.test/good", ""),
        ]),
        fetcher=fetcher,
    )
    result = learner.research("radar")
    assert len(urls) == 2
    assert result["facts_learned"] >= 1
    assert result["results"][0]["ok"] is False
    assert result["sources"] == ["https://example.test/good"]


# --------------------------------------------------------------------------- #
# Bounds
# --------------------------------------------------------------------------- #
def test_the_session_bound_stops_further_provider_calls():
    learner = _learner(max_queries_per_session=1, suppression_window_seconds=0)
    learner.research("radar")
    calls_after_first = len(learner.provider.calls)
    second = learner.research("lidar")
    assert second["searched"] is False
    assert second["bounded"] is True
    assert "per-session" in second["reason"]
    assert len(learner.provider.calls) == calls_after_first


def test_the_per_page_cap_bounds_fetches():
    fetched = []

    def fetcher(url, **kwargs):
        fetched.append(url)
        return FetchResult(ok=True, url=url, text=PAGE_TEXT, status=200)

    learner = SearchLearner(
        FakeEngine(),
        config={"enabled": True, "provider": "duckduckgo", "max_pages_per_search": 2},
        provider=FakeProvider([
            SearchResult("a", "https://example.test/a"),
            SearchResult("b", "https://example.test/b"),
            SearchResult("c", "https://example.test/c"),
        ]),
        fetcher=fetcher,
    )
    learner.research("radar")
    assert len(fetched) == 2


# --------------------------------------------------------------------------- #
# Honest failure
# --------------------------------------------------------------------------- #
def test_an_empty_result_stores_nothing_and_reports_no_information():
    engine = FakeEngine()
    learner = SearchLearner(
        engine,
        config={"enabled": True, "provider": "duckduckgo"},
        provider=FakeProvider([]),
        fetcher=_fetcher(),
    )
    result = learner.research("radar")
    assert result["searched"] is True
    assert result["facts_learned"] == 0
    assert result["answer"] is None
    assert engine.knowledge.count == 0
    assert engine.taught == []
    assert "no results" in result["reason"]


def test_a_provider_failure_leaves_the_memory_untouched():
    engine = FakeEngine()
    learner = SearchLearner(
        engine,
        config={"enabled": True, "provider": "duckduckgo"},
        provider=FakeProvider(error=RuntimeError("rate limited")),
        fetcher=_fetcher(),
    )
    result = learner.research("radar")
    assert result["facts_learned"] == 0
    assert "rate limited" in result["reason"]
    assert engine.knowledge.count == 0


def test_a_page_with_no_usable_text_stores_nothing():
    engine = FakeEngine()
    learner = SearchLearner(
        engine,
        config={"enabled": True, "provider": "duckduckgo"},
        provider=FakeProvider([SearchResult("empty", "https://example.test/empty")]),
        fetcher=_fetcher(text="  "),
    )
    result = learner.research("radar")
    assert result["facts_learned"] == 0
    assert engine.knowledge.count == 0
    assert "no usable content" in result["reason"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_configuration_defaults_to_disabled_and_merges_the_profile_block():
    defaults = normalize_research_config()
    assert defaults["enabled"] is False
    assert defaults["provider"] is None
    merged = normalize_research_config(
        {"enabled": True, "provider": "JSON_API", "max_pages_per_search": "2"}
    )
    assert merged["enabled"] is True
    assert merged["provider"] == "json_api"
    assert merged["max_pages_per_search"] == 2


def test_topic_normalization_is_stable():
    assert normalize_topic("  What IS Radar? ") == "what is radar"
    assert normalize_topic("What is radar?") == normalize_topic("what is radar")
