"""The search-and-learn loop: decide, search, fetch, distill, teach.

The policy lives here, not in the provider and not in the engine:

* **decide** from the retrieval confidence of existing memory — a confident
  recall never searches, and a topic searched within the suppression window is
  not searched again;
* **bound** every search by a per-session and per-minute query budget, a page
  cap, and the fetcher's own size and timeout limits;
* **distill** fetched text into standalone propositions through the configured
  teacher, or through a local sentence splitter when no teacher is configured;
* **teach** every proposition with its provenance (source URL, retrieving
  query, retrieval timestamp) so web-sourced facts stay distinguishable;
* **fail honestly** — a provider or transport failure, or an empty result,
  leaves the memory unchanged and returns a structured result. Nothing is
  fabricated and no placeholder fact is stored.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from src.model import web_teacher

__all__ = [
    "DEFAULT_RESEARCH_CONFIG",
    "normalize_research_config",
    "normalize_topic",
    "SearchLearner",
]

DEFAULT_RESEARCH_CONFIG: Dict[str, Any] = {
    # Acquisition is opt-in: an absent block means the feature is disabled and a
    # profile behaves exactly as it did before this capability existed.
    "enabled": False,
    "provider": None,
    "confidence_threshold": 0.55,
    "suppression_window_seconds": 300.0,
    "max_queries_per_session": 8,
    "max_queries_per_minute": 3,
    "max_pages_per_search": 3,
    "result_limit": 5,
    "page_char_budget": 6000,
    "min_proposition_chars": 12,
    "max_propositions_per_page": 5,
    "request_timeout": web_teacher.DEFAULT_TIMEOUT_SECONDS,
    "max_response_bytes": web_teacher.DEFAULT_MAX_BYTES,
    "base_url": None,
    "api_key_env": "TMT_SEARCH_API_KEY",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CONVERSATIONAL_MARKERS = (
    "based on", "here is", "according to", "in summary", "to summarize",
)


def normalize_research_config(config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Merge a profile's research block onto the documented defaults."""
    merged = dict(DEFAULT_RESEARCH_CONFIG)
    for key, value in dict(config or {}).items():
        merged[key] = value
    merged["enabled"] = bool(merged.get("enabled"))
    merged["provider"] = (str(merged["provider"]).strip().lower()
                          if merged.get("provider") else None)
    for key in ("max_queries_per_session", "max_queries_per_minute",
                "max_pages_per_search", "result_limit", "page_char_budget",
                "max_propositions_per_page", "min_proposition_chars",
                "max_response_bytes"):
        merged[key] = int(merged[key])
    for key in ("confidence_threshold", "suppression_window_seconds", "request_timeout"):
        merged[key] = float(merged[key])
    return merged


def normalize_topic(query: str) -> str:
    """The suppression key: lowercase content words, order preserved."""
    words = re.findall(r"[a-z0-9]+", str(query).lower())
    return " ".join(words) or str(query).strip().lower()


class SearchLearner:
    """Closes the loop from an unanswerable question to attributed facts."""

    def __init__(self, engine: Any, config: Optional[Mapping[str, Any]] = None,
                 provider: Optional[web_teacher.SearchProvider] = None,
                 fetcher: Optional[Callable[..., Any]] = None,
                 teacher: Optional[Any] = None,
                 clock: Optional[Callable[[], float]] = None):
        self.engine = engine
        self.config = normalize_research_config(config)
        self.clock = clock or time.time
        self.fetcher = fetcher or web_teacher.fetch_page
        self.teacher = teacher
        self.provider_failure: Optional[Dict[str, Any]] = None
        if provider is not None:
            self.provider = provider
        else:
            self.provider, self.provider_failure = web_teacher.select_provider(self.config)
        self._session_queries = 0
        self._query_times: List[float] = []
        self._suppressed: Dict[str, float] = {}
        self.last_research: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled"))

    @property
    def available(self) -> bool:
        return self.enabled and self.provider is not None

    def unavailable_reason(self) -> str:
        if not self.enabled:
            return "external acquisition is disabled for this profile"
        if self.provider is None:
            self.provider_failure = self.provider_failure or web_teacher.unavailable()
            return str(self.provider_failure.get("reason") or "external acquisition is unavailable")
        return ""

    def retrieval_confidence(self, hits: Optional[Sequence[Mapping[str, Any]]]) -> float:
        """The engine's retrieval confidence: the best hit's fused similarity."""
        if not hits:
            return 0.0
        return max(float(hit.get("similarity") or 0.0) for hit in hits)

    def _check_bounds(self) -> Optional[str]:
        now = self.clock()
        self._query_times = [stamp for stamp in self._query_times if now - stamp < 60.0]
        if len(self._query_times) >= self.config["max_queries_per_minute"]:
            return (
                f"per-minute search bound reached "
                f"({self.config['max_queries_per_minute']} queries)"
            )
        if self._session_queries >= self.config["max_queries_per_session"]:
            return (
                f"per-session search bound reached "
                f"({self.config['max_queries_per_session']} queries)"
            )
        return None

    def should_search(self, query: str, hits: Optional[Sequence[Mapping[str, Any]]] = None,
                      force: bool = False) -> Dict[str, Any]:
        """The documented decision policy, returned as a structured decision."""
        topic = normalize_topic(query)
        decision: Dict[str, Any] = {"topic": topic, "search": False, "reason": None}
        reason = self.unavailable_reason()
        if reason:
            decision["reason"] = reason
            return decision

        confidence = self.retrieval_confidence(hits)
        decision["confidence"] = confidence
        if not force and confidence >= self.config["confidence_threshold"]:
            decision["reason"] = (
                f"memory answers confidently (confidence {confidence:.2f} >= "
                f"{self.config['confidence_threshold']})"
            )
            return decision

        suppressed_until = self._suppressed.get(topic)
        if suppressed_until is not None and self.clock() < suppressed_until:
            decision["reason"] = (
                f"topic was searched within the {self.config['suppression_window_seconds']}s "
                "suppression window"
            )
            return decision

        bounded = self._check_bounds()
        if bounded:
            decision["bound_reached"] = True
            decision["reason"] = bounded
            return decision

        decision["search"] = True
        return decision

    # ------------------------------------------------------------------ #
    # Acquisition
    # ------------------------------------------------------------------ #
    def _distill(self, text: str, query: str) -> List[str]:
        """Propositions from a page: the teacher when configured, else locally."""
        if self.teacher is not None:
            try:
                distilled = self.teacher.distill_propositions(text, timeout=self.config["request_timeout"])
                propositions = self._filter_propositions(distilled)
                if propositions:
                    return propositions
            except Exception:
                # A teacher outage is not a reason to discard usable text.
                pass
        return self._local_propositions(text)

    def _filter_propositions(self, items: Sequence[Any]) -> List[str]:
        kept: List[str] = []
        for item in items or []:
            text = " ".join(str(item).split())
            if len(text) < self.config["min_proposition_chars"]:
                continue
            if text.lower().startswith(_CONVERSATIONAL_MARKERS):
                continue
            if not re.search(r"[a-zA-Z]", text):
                continue
            if text not in kept:
                kept.append(text)
        return kept[: self.config["max_propositions_per_page"]]

    def _local_propositions(self, text: str) -> List[str]:
        sentences = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
        return self._filter_propositions(sentences)

    def _active_fact_count(self) -> Optional[int]:
        try:
            return int(self.engine.knowledge.active_count())
        except Exception:
            return None

    def _teach(self, proposition: str, url: str, query: str) -> bool:
        """Teach one proposition with provenance; return whether it was accepted."""
        kwargs = {
            "source": "web",
            "origin": "web",
            "source_url": url,
            "source_query": query,
        }
        try:
            self.engine.teach(proposition, **kwargs)
            return True
        except TypeError:
            # An engine that predates provenance still learns the fact.
            try:
                self.engine.teach(proposition, source="web")
                return True
            except Exception:
                return False
        except Exception:
            return False

    def _record_search(self, topic: str) -> None:
        now = self.clock()
        self._query_times.append(now)
        self._session_queries += 1
        self._suppressed[topic] = now + self.config["suppression_window_seconds"]

    def research(self, query: str, hits: Optional[Sequence[Mapping[str, Any]]] = None,
                 force: bool = True) -> Dict[str, Any]:
        """Run the loop once and return the structured result.

        ``force`` bypasses only the confidence gate (an explicit research
        request). Availability, the suppression window and the query bounds
        always apply.
        """
        decision = self.should_search(query, hits=hits, force=force)
        topic = decision["topic"]
        result: Dict[str, Any] = {
            "topic": topic,
            "query": query,
            "searched": False,
            "queries": [],
            "sources": [],
            "facts_learned": 0,
            "facts": [],
            "results": [],
            "answer": None,
            "available": self.available,
            "bounded": bool(decision.get("bound_reached")),
            "reason": decision.get("reason"),
        }
        if not decision["search"]:
            self.last_research = result
            return result

        before = self._active_fact_count()
        result["searched"] = True
        result["reason"] = None

        search_result = self._run_search(query, topic, result)
        if not search_result:
            self.last_research = result
            return result

        after = self._active_fact_count()
        if before is not None and after is not None:
            result["facts_learned"] = max(0, after - before)
        else:
            result["facts_learned"] = len(result["facts"])

        if result["facts_learned"] > 0:
            result["answer"] = self._answer_from_memory(topic, result["facts"])
        elif not result["reason"]:
            result["reason"] = "no usable content was found"
        self.last_research = result
        return result

    def _run_search(self, query: str, topic: str,
                    result: Dict[str, Any]) -> bool:
        """Execute one bounded search and teach what it yields."""
        self._record_search(topic)
        result["queries"].append(query)
        try:
            search_results = self.provider.search(query, limit=self.config["result_limit"])
        except Exception as exc:  # provider failure is structured, never raised
            result["reason"] = f"{type(exc).__name__}: {exc}"
            return False

        if not search_results:
            result["reason"] = "the search provider returned no results"
            return False

        for entry in search_results[: self.config["max_pages_per_search"]]:
            url = getattr(entry, "url", "")
            title = getattr(entry, "title", "")
            fetched = self.fetcher(
                url,
                timeout=self.config["request_timeout"],
                max_bytes=self.config["max_response_bytes"],
                max_chars=self.config["page_char_budget"] * 2,
            )
            ok = bool(getattr(fetched, "ok", False))
            record = {
                "url": url,
                "title": title,
                "ok": ok,
                "reason": getattr(fetched, "reason", None),
                "truncated": bool(getattr(fetched, "truncated", False)),
                "chars": len(getattr(fetched, "text", "") or ""),
                "propositions": 0,
                "facts_learned": 0,
            }
            result["results"].append(record)
            if not ok:
                # A refused or failed fetch does not abort the rest of the search.
                continue
            text = (getattr(fetched, "text", "") or "")[: self.config["page_char_budget"]]
            if len(text.strip()) < self.config["min_proposition_chars"]:
                record["reason"] = "the page yielded no usable text"
                continue
            propositions = self._distill(text, query)
            record["propositions"] = len(propositions)
            if not propositions:
                record["reason"] = "distillation produced no propositions"
                continue
            result["sources"].append(url)
            taught: List[str] = []
            for proposition in propositions:
                if self._teach(proposition, url, query):
                    taught.append(proposition)
            record["facts_learned"] = len(taught)
            result["facts"].extend(taught)

        return True

    def _answer_from_memory(self, topic: str, taught: Sequence[str]) -> str:
        """Answer from what was learned, preferring the ordinary recall path."""
        try:
            hits = self.engine.recall(topic, top_k=3, threshold=0.0) or []
        except Exception:
            hits = []
        web_hits = [hit for hit in hits if hit.get("origin") == "web"]
        chosen = web_hits or hits
        if chosen:
            return " ".join(str(hit.get("text", "")).strip() for hit in chosen if hit.get("text"))
        return " ".join(taught[:3])
