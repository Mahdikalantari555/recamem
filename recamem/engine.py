import contextlib
import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

from recamem.locking import ProfileBusyError, ProfileWriterLock
from recamem.segmenter import extract_propositions
from recamem.plastic_adapter import (
    apply_plastic_update, clone_state, identify_adapter,
)
from recamem.anchor import OnnxMiniLM
from recamem.episodic import KnowledgeStore
from recamem.byte_rtu import ByteRtuBackend
from src.model.rtu import InvalidProfileError, MissingCheckpointError
from recamem.consolidation import (
    DEFAULT_SLEEP_CONFIG,
    AutoSleepWorker,
    DecreasingBackprop,
    MemoryAwareSynapses,
    SleepConsolidation,
)

# ponytail: DroidEngine binds 22MB ONNX MiniLM semantic anchor to plastic RTU memory.

# ---------------------------------------------------------------------------
# Zero-dependency discourse / TMS helpers (Decision 3 & 4 in design.md)
# ---------------------------------------------------------------------------
_ANAPHORA_RE = re.compile(r"^(It|They|This|These|Those)\b")

# Function-predicate verb set for SVO slot extraction (kept deliberately coarse).
_VERB_SET = {
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "do", "does", "did", "will", "would", "can", "could", "shall", "should",
    "may", "might", "must", "use", "uses", "using", "refer", "refers",
    "mean", "means", "involve", "involves", "include", "includes",
    "provide", "provides", "enable", "enables", "allow", "allows",
    "help", "helps", "measure", "measures", "detect", "detects",
    "capture", "captures", "record", "records", "observe", "observes",
    "monitor", "monitors", "support", "supports", "contain", "contains",
    "consist", "consists", "apply", "applies", "work", "works",
    "operate", "operates", "give", "gives", "make", "makes", "take",
    "takes", "need", "needs", "require", "requires", "cover", "covers",
    "produce", "produces", "create", "creates", "build", "builds",
    "develop", "develops", "design", "designs", "test", "tests",
    "check", "checks", "compare", "compares", "analyze", "analyzes",
    "evaluate", "evaluates", "assess", "assesses", "estimate",
    "estimates", "calculate", "calculates", "compute", "computes",
    "derive", "derives", "extract", "extracts", "process", "processes",
    "transform", "transforms", "convert", "converts", "transmit",
    "transmits", "receive", "receives", "send", "sends", "store",
    "stores", "load", "loads", "save", "saves", "scan", "scans", "map",
    "maps", "model", "models", "simulate", "simulates", "predict",
    "predicts", "forecast", "forecasts", "classify", "classifies",
    "cluster", "clusters", "segment", "segments", "recognize",
    "recognizes", "identify", "identifies", "locate", "locates", "track",
    "tracks", "quantify", "quantifies", "validate", "validates",
    "verify", "verifies", "calibrate", "calibrates", "correct",
    "corrects", "adjust", "adjusts", "optimize", "optimizes", "improve",
    "improves", "enhance", "enhances", "reduce", "reduces", "increase",
    "increases", "decrease", "decreases", "minimize", "minimizes",
    "maximize", "maximizes", "balance", "balances", "control",
    "controls", "manage", "manages", "handle", "handles", "deal",
    "deals", "cope", "copes", "execute", "executes", "implement",
    "implements", "utilize", "utilizes", "employ", "employs", "leverage",
    "leverages", "adopt", "adopts", "adapt", "adapts",
}

_AUXILIARIES = {
    "do", "does", "did", "is", "are", "was", "were", "be", "been",
    "being", "has", "have", "had", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
}

_NEGATIONS = {
    "not", "no", "never", "none", "nor", "neither", "without", "n't",
    "hardly", "barely",
}


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9°'’-]+", text.lower())


def _subject_and_predicate(text: str) -> tuple[str, Optional[str]]:
    """Rough SVO slot extraction: subject words before the head verb."""
    words = re.findall(r"[A-Za-z0-9°.\u2019'-]+", text)
    if not words:
        return "", None
    lowers = [w.lower() for w in words]
    i = next((k for k, w in enumerate(lowers) if w in _VERB_SET), None)
    if i is None:
        return " ".join(words[:2]), None
    subj = " ".join(words[:i]).strip() if i > 0 else " ".join(words[:2])
    j = i
    # walk past auxiliaries + negations to reach the head verb
    while j + 1 < len(lowers) and (lowers[j] in _AUXILIARIES or lowers[j] in _NEGATIONS):
        j += 1
    head = lowers[j]
    if head not in _VERB_SET:
        head = lowers[i]  # no real head verb ahead: stay on the copula
    elif head not in _AUXILIARIES and head.endswith("s"):
        head = head.rstrip("s")  # crude lemma: "requires" -> "require"
    return subj, head


def _extract_subject(sentence: str) -> Optional[str]:
    subj, _ = _subject_and_predicate(sentence)
    return subj or None


def _has_negation(text: str) -> bool:
    return any(t in _NEGATIONS for t in _tokens(text))


def _content_overlap(a: str, b: str) -> float:
    """Shared content-token fraction over the shorter sentence."""
    ta, tb = set(_tokens(a)), set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# "The <noun> of <subject> is <object>" -> functional predicate, one output per
# input. Covers the spec's capital-of examples without a full NLP parse.
_FUNC_PRED_RE = re.compile(
    r"^\s*the\s+(\w+)\s+of\s+(.+?)\s+(?:is|are|was|were)\s+(.+)$",
    re.IGNORECASE,
)

_COPIULA = _AUXILIARIES


def _svo_slots(text: str) -> tuple[str, Optional[str]]:
    """(subject, functional predicate) for contradiction gating."""
    m = _FUNC_PRED_RE.match(text)
    if m:
        return m.group(2).strip().lower(), m.group(1).lower() + " of"
    subj, pred = _subject_and_predicate(text)
    return subj.lower(), pred


def _is_contradiction(new_text: str, old_text: str) -> bool:
    """Deterministic tier-2 check: functional-predicate slot clash or
    polarity inversion. Caller pre-gates on dense sim >= 0.65."""
    s_new, p_new = _svo_slots(new_text)
    s_old, p_old = _svo_slots(old_text)
    if not p_new or not p_old:
        # No functional predicate to compare: fall back to pure polarity
        # inversion on overlapping content.
        overlap = set(_tokens(new_text)) & set(_tokens(old_text))
        shared = {t for t in overlap if t not in _NEGATIONS}
        return _has_negation(new_text) != _has_negation(old_text) and len(shared) >= 3
    if s_new != s_old or p_new != p_old:
        return False
    if p_new in _COPIULA:
        # "X is A" vs "X is B" is not a contradiction (non-functional copula);
        # only a negation flip on the *same statement* is.
        return (
            _has_negation(new_text) != _has_negation(old_text)
            and _content_overlap(new_text, old_text) >= 0.5
        )
    obj_new = set(_tokens(new_text)) - set(_tokens(s_new)) - {p_new}
    obj_old = set(_tokens(old_text)) - set(_tokens(s_old)) - {p_old}
    slot_clash = bool(obj_new != obj_old and (obj_new | obj_old))
    polarity = (
        _has_negation(new_text) != _has_negation(old_text)
        and _content_overlap(new_text, old_text) >= 0.5
    )
    return slot_clash or polarity


# Try imports for new modules with fallbacks
try:
    from src.model.rtu import PlasticAssociativeRTU, RTUMemoryBlock
except ImportError:
    try:
        from .rtu import PlasticAssociativeRTU, RTUMemoryBlock
    except Exception:  # noqa: BLE001
        RTUMemoryBlock = None
        PlasticAssociativeRTU = None

try:
    from recamem.plastic_adapter import get_adapter, list_adapters, PlasticAdapter
except Exception:  # noqa: BLE001
    try:
        from .plastic_adapter import get_adapter, list_adapters, PlasticAdapter
    except Exception:  # noqa: BLE001
        get_adapter = None
        list_adapters = lambda: []
        PlasticAdapter = object

try:
    from recamem.decision_head import DecisionHead
except Exception:  # noqa: BLE001
    try:
        from .decision_head import DecisionHead
    except Exception:  # noqa: BLE001
        DecisionHead = None

try:
    from recamem.lifecycle import RetrievalPolicy, ContinuousStream, SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver
except Exception:  # noqa: BLE001
    try:
        from .lifecycle import RetrievalPolicy, ContinuousStream, SnapshotManager, MergeEngine, SemanticStore, ContradictionResolver
    except Exception:  # noqa: BLE001
        RetrievalPolicy = None
        ContinuousStream = None
        SnapshotManager = None
        MergeEngine = None
        SemanticStore = None
        ContradictionResolver = None

try:
    from recamem.gguf import GgufBackend
except Exception:  # noqa: BLE001
    try:
        from .gguf import GgufBackend
    except Exception:  # noqa: BLE001
        GgufBackend = None

try:
    from recamem.injection import (
        inject_prompt, inject_hidden, inject_kv_cache, select_strategy,
        resolve_strategy, plan_prompt,
    )
    from recamem.backend_contracts import (
        HiddenInjection, KvPrefill, capabilities_of,
        UnsupportedCapabilityError, QueryBudgetError,
    )
except Exception:  # noqa: BLE001
    try:
        from .injection import (
            inject_prompt, inject_hidden, inject_kv_cache, select_strategy,
            resolve_strategy, plan_prompt,
        )
        from .backend_contracts import (
            HiddenInjection, KvPrefill, capabilities_of,
            UnsupportedCapabilityError, QueryBudgetError,
        )
    except Exception:  # noqa: BLE001
        def inject_prompt(q, facts, max_context_tokens=3000):
            if not facts:
                return q
            ctx = "\n".join([f"- {f.get('text','')}" for f in facts[:5]])
            return f"Context:\n{ctx}\n\nQuery: {q}"
        def inject_hidden(backend, query, h_trace, layer_idx=12):
            return {"prompt": query, "strategy": "hidden"}
        def inject_kv_cache(backend, facts, query):
            return {"prompt": inject_prompt(query, facts), "strategy": "kv_cache"}
        def select_strategy(fact_count, query_type, profile_config=None, capabilities=None):
            if fact_count <5:
                return "prompt"
            elif fact_count <=20:
                return "hidden"
            else:
                return "kv_cache"
        def resolve_strategy(fact_count, query_type="specific_fact", profile_config=None,
                             capabilities=None, requested=None, strict=False):
            class _S:
                executed = select_strategy(fact_count, query_type, profile_config)
                requested = executed
                reason = "fallback"
                def to_dict(self):
                    return {"strategy": self.executed, "requested": self.requested,
                            "executed": self.executed, "reason": self.reason,
                            "fallback": False, "fact_count": fact_count}
            return _S()
        def plan_prompt(query, facts, **kwargs):
            raise RuntimeError("prompt planning unavailable")
        def capabilities_of(backend):
            class _C:
                name = "none"
                hidden = False
                kv_prime = False
                prompt = True
                prompt_only = True
                budget = 3000
                context_window = 4096
                output_reserve = 512
                def supports(self, strategy):
                    return strategy == "prompt"
                def to_dict(self):
                    return {"name": "none", "hidden": False, "kv_prime": False}
            return _C()
        class HiddenInjection:  # pragma: no cover - import fallback only
            def __init__(self, *args, **kwargs):
                raise RuntimeError("backend contracts unavailable")
        class KvPrefill:  # pragma: no cover - import fallback only
            def __init__(self, *args, **kwargs):
                raise RuntimeError("backend contracts unavailable")
        class UnsupportedCapabilityError(RuntimeError):
            pass
        class QueryBudgetError(ValueError):
            pass

try:
    from recamem.confidence import retrieval_confidence, generation_confidence, fuse_confidence, confidence_tier
except Exception:  # noqa: BLE001
    try:
        from .confidence import retrieval_confidence, generation_confidence, fuse_confidence, confidence_tier
    except Exception:  # noqa: BLE001
        def retrieval_confidence(sim, surprise=None, health=None):
            return sim
        def generation_confidence(logprobs=None, self_consistency=None, entropy=None):
            return 0.7
        def fuse_confidence(Cr, Cg, weights=None):
            return 2*Cr*Cg/(Cr+Cg+1e-8) if (Cr+Cg)>0 else 0.0
        def confidence_tier(c):
            return "high" if c>0.8 else "medium" if c>=0.5 else "low"

NO_INFORMATION_MESSAGE = (
    "I do not have information about this topic in my memory yet. "
    "Teach me by saying: 'learn: <facts>'."
)

# ---------------------------------------------------------------------------
# Recall scoring (change 1.12)
# ---------------------------------------------------------------------------
# One named table, one composition. The dense cosine dominates so the
# ``dense >= DENSE_GATE`` rule stays authoritative (plan 10 decision 6): the
# associative bonus is zeroed below the gate and can never resurrect an
# out-of-domain hit. Lexical/recency/confidence are weaker re-ranking signals;
# the lexical term keeps the pre-change ability to admit a near-gate candidate
# that clears the caller's threshold. The associative/recency/confidence
# weights are the slots the correctness change reserved; this change
# activates them.
RECALL_SCORE_WEIGHTS: Dict[str, float] = {
    "dense": 1.0,
    "lexical": 0.70,
    "associative": 0.15,
    "recency": 0.02,
    "confidence": 0.05,
    "contextual": 0.20,
}

def score_candidate(signals: dict, weights: dict = None) -> float:
    """Fusion score; the caller gates the associative signal by dense similarity."""
    weights = RECALL_SCORE_WEIGHTS if weights is None else weights
    additive = sum(weights[name] * float(signals.get(name, 0.0))
                   for name in ("dense", "lexical", "associative", "recency", "confidence"))
    return additive * (1.0 + weights["contextual"] * max(0.0, float(signals.get("contextual", 0.0))))


DENSE_GATE = 0.48

# Recency decay half-life for the recall score.
RECENCY_HALF_LIFE_SECONDS = 30.0 * 24.0 * 60.0 * 60.0

# Canonical answer paths (change 1.19). The old names are kept as a
# read-only mapping for external callers.
ANSWER_PATH_RECALL = "recall"
ANSWER_PATH_FUSED = "fused"
ANSWER_PATH_LLM = "llm"
ANSWER_PATH_RESEARCHED = "researched"
_LEGACY_ANSWER_PATH = {
    ANSWER_PATH_RECALL: "extractive",
    ANSWER_PATH_FUSED: "generative",
    ANSWER_PATH_LLM: "generative",
    # A researched answer is extractive in the legacy vocabulary: it is built
    # from facts, not free generation.
    ANSWER_PATH_RESEARCHED: "extractive",
}

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "by",
    "for", "with", "is", "are", "was", "were", "be", "been", "being", "do",
    "does", "did", "have", "has", "had", "it", "its", "this", "that", "these",
    "those", "which", "who", "whom", "whose", "what", "when", "where", "why",
    "how", "can", "could", "should", "would", "will", "shall", "may", "about",
    "into", "over", "under", "between", "during", "through", "out", "up",
    "down", "you", "your",
}


def _lexical_overlap(query: str, fact: str) -> float:
    """Content-word overlap in [0, 1]; the lexical signal of the recall score."""
    q_words = set(re.findall(r"[a-z0-9]+", query.lower())) - _STOP_WORDS
    f_words = set(re.findall(r"[a-z0-9]+", fact.lower())) - _STOP_WORDS
    if not q_words:
        return 0.0
    return len(q_words & f_words) / float(len(q_words))


# A definitional predicate ("X is the/a/an ...") marks a fact as a definition
# candidate for definitional queries; it is the lexical prior the old
# hardcoded definitional boost encoded.
_DEFINITIONAL_PREDICATE = re.compile(r"\b(?:is|are)\s+(?:the|a|an)\b", re.IGNORECASE)


def _lexical_score(query: str, fact: str, query_type: str) -> float:
    """Lexical signal: content-word overlap only.

    The former hardcoded definitional boost (+0.5) is removed; definitional
    preference is left to the learned scorer (see recamem.retrieval).
    """
    return _lexical_overlap(query, fact)


def _recency_score(timestamp: Optional[float]) -> float:
    """Exponential recency in [0, 1] with a 30-day half-life."""
    if not timestamp:
        return 0.0
    try:
        age = max(0.0, time.time() - float(timestamp))
    except (TypeError, ValueError):
        return 0.0
    return math.exp(-age / RECENCY_HALF_LIFE_SECONDS)


def _confidence_score(fact: Dict[str, Any]) -> float:
    """Fact-level confidence, clamped to [0, 1] (novelty is the stored proxy)."""
    try:
        value = float(fact.get("novelty", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))



#: Version written into ``config.json``. A newer manifest is refused by name;
#: an older one is read through the explicit upgrade step in load_profile.
PROFILE_SCHEMA_VERSION = 2


class DroidEngine(nn.Module):
    def __init__(self, name: str = "droid-alpha", dim: int = 384,
                 db_path: Optional[str] = None,
                 plastic_adapter: Optional[Union[str, Any]] = None,
                 adapter_kwargs: Optional[Dict[str, Any]] = None,
                 gguf_backend: Optional[Any] = None,
                 byte_rtu_backend: Optional[Any] = None,
                 gguf_config: Optional[Dict[str, Any]] = None,
                 test_time_training: bool = True,
                 surprise_threshold: float = 0.4,
                 K_max: int = 5000,
                 migrations: Optional[List[Dict[str, Any]]] = None):
        super().__init__()
        self.name = name
        self.dim = dim
        self.device = torch.device("cpu")
        # Profile directory, when the engine is bound to one. It is the anchor
        # for the single-writer lock and for schema-versioned manifests.
        self.profile_dir: Optional[Path] = None
        self._writer_tls = threading.local()
        # SLOW tier meta-parameters (opt-in; FAST writes never need them).
        self.slow_tier = None
        self.anchor = OnnxMiniLM()
        adapter_kwargs = adapter_kwargs or {}
        # Store for profile round-trip (task 2.12)
        self.adapter_kwargs = dict(adapter_kwargs)
        # Registered adapter identity, persisted verbatim. An explicitly
        # configured unknown adapter fails instead of silently becoming the
        # default (post-merge 2.6).
        self.adapter_name: Optional[str] = None
        self.migrations: List[Dict[str, Any]] = list(migrations or [])
        # Propagate K_max to plastic adapter for capacity enforcement (task 2.6)
        if "K_max" not in adapter_kwargs and "k_max" not in adapter_kwargs:
            adapter_kwargs["K_max"] = K_max
        if plastic_adapter is not None:
            if isinstance(plastic_adapter, str):
                if get_adapter is None:
                    from src.model.rtu import PlasticAssociativeRTU
                    self.memory = PlasticAssociativeRTU(dim=dim, **adapter_kwargs)
                else:
                    kwargs = {k: v for k, v in dict(adapter_kwargs).items() if k != "dim"}
                    kwargs.setdefault("dim", dim)
                    try:
                        self.memory = get_adapter(plastic_adapter, **kwargs)
                    except KeyError as exc:
                        raise InvalidProfileError(
                            f"Plastic adapter {plastic_adapter!r} is not registered; "
                            f"refusing to fall back to the default. Available: "
                            f"{list_adapters()}"
                        ) from exc
                    self.adapter_name = plastic_adapter
            else:
                self.memory = plastic_adapter
                self.adapter_name = identify_adapter(plastic_adapter)
                if self.adapter_name is None:
                    # Keep the class registered so save/load can round-trip it.
                    registered = getattr(plastic_adapter, "_adapter_type", None)
                    if isinstance(registered, str) and registered:
                        self.adapter_name = registered
                # If passing pre-built memory, sync its K_max to engine's K_max if it supports it
                if hasattr(self.memory, 'K_max'):
                    try:
                        self.memory.K_max = K_max
                    except Exception:
                        pass
        else:
            try:
                from src.model.rtu import PlasticAssociativeRTU
                self.memory = PlasticAssociativeRTU(dim=dim, **adapter_kwargs)
            except Exception as e:
                from src.model.rtu import RTUMemoryBlock
                self.memory = RTUMemoryBlock(dim=dim)
            self.adapter_name = "associative_rtu"
        # Expose anchor mock state for UI and CLI (task 2.10, 2.11)
        try:
            self.anchor_is_mock = bool(getattr(self.anchor, 'is_mock', False))
        except Exception:
            self.anchor_is_mock = False
        self.knowledge = KnowledgeStore(db_path or ":memory:", dim=dim)
        self.knowledge_vectors: Optional[torch.Tensor] = None
        self._discourse_subject: Optional[str] = None
        self.step_count = 0
        self.logs: List[Dict[str, Any]] = []
        self.base_lr = 0.05
        self.last_query: Optional[str] = None
        self.surprise_threshold = surprise_threshold
        self.K_max = K_max
        self.test_time_training = test_time_training
        self.ttt_steps_this_session = 0
        self.max_ttt_steps = 100
        self.ttt_lr = 1e-4
        self.ttt_rate = 1.0              # allowed fraction of updates (1.0 = all)
        self.ttt_attempts = 0
        self.ttt_skipped = 0
        self.ttt_last_skip_reason: Optional[str] = None
        # One lock serialises teach/recall/sleep and worker lifecycle.
        self._engine_lock = threading.RLock()
        self.last_activity = time.time()
        self.sleep_config = json.loads(json.dumps(DEFAULT_SLEEP_CONFIG))
        self.sleep_config["schedule"]["base_lr"] = self.base_lr
        self.lr_schedule = DecreasingBackprop(
            base_lr=self.base_lr,
            decay_factor=float(self.sleep_config["schedule"].get("decay_factor", 0.95)),
            index=0,
        )
        self.mas: Optional[MemoryAwareSynapses] = None
        self.last_sleep_report: Optional[Dict[str, Any]] = None
        self.sleep_cycles = 0
        self.auto_sleep_worker: Optional[AutoSleepWorker] = None
        if DecisionHead is not None:
            try:
                self.decision_head = DecisionHead(dim=dim)
            except Exception:  # noqa: BLE001
                self.decision_head = None
        else:
            self.decision_head = None
        if RetrievalPolicy is not None:
            try:
                self.retrieval_policy = RetrievalPolicy()
            except Exception:  # noqa: BLE001
                self.retrieval_policy = None
        else:
            self.retrieval_policy = None
        self.snapshots: Dict[str, Dict] = {}
        self.current_version: Optional[str] = None
        self.snapshot_manager = None
        if SnapshotManager is not None:
            try:
                self.snapshot_manager = SnapshotManager()
            except Exception:  # noqa: BLE001
                pass
        self.continuous_stream = None
        if byte_rtu_backend is not None:
            if gguf_backend is not None:
                raise ValueError(
                    "Supply only one of gguf_backend and byte_rtu_backend."
                )
            self.gguf_backend = byte_rtu_backend
            self.gguf_config = gguf_config or {}
            self.gguf_config["injection_strategy"] = "prompt"
        else:
            self.gguf_backend = gguf_backend
            self.gguf_config = gguf_config or {}
            if gguf_config and GgufBackend is not None and self.gguf_backend is None:
                try:
                    model_path = gguf_config.get("gguf_model_path") or gguf_config.get("model_path")
                    repo_id = gguf_config.get("gguf_repo_id") or gguf_config.get("repo_id")
                    filename = gguf_config.get("gguf_filename") or gguf_config.get("filename")
                    n_ctx = gguf_config.get("n_ctx", 4096)
                    n_gpu_layers = gguf_config.get("n_gpu_layers", -1)
                    params = gguf_config.get("gguf_params", {})
                    if model_path or repo_id:
                        self.gguf_backend = GgufBackend(model_path=model_path, repo_id=repo_id, filename=filename, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, **params)
                except Exception as e:
                    print(f"Failed to init GGUF backend from config: {e}")
        self._last_retrieval_strategy = "hybrid_rrf"
        self._retrieval_history: List[Dict] = []
        self._last_answer_path = ANSWER_PATH_RECALL
        # External acquisition (adaptation/web-grounded-learning). Disabled
        # until a profile manifest configures it, so an unconfigured profile
        # behaves exactly as it did before the capability existed.
        self.research_config: Dict[str, Any] = {}
        self.search_learner = None
        self.last_research_result: Optional[Dict[str, Any]] = None
        self.last_research_sources: List[str] = []
        # Normalise the defaults so an unconfigured engine reports "disabled"
        # rather than "unknown", and selects no provider.
        self.configure_research({})
        # Manifest fields (change 1.1-1.2). A fresh engine infers its engine
        # from the attached backend; load_profile() overwrites these.
        if isinstance(self.gguf_backend, ByteRtuBackend):
            self.engine = "byte_rtu"
            self.fusion_mode = "prompt_stuffing"
            self.byte_rtu_checkpoint: Optional[str] = "rtu.safetensors"
        else:
            self.engine = "droid"
            self.fusion_mode = "off"
            self.byte_rtu_checkpoint = None
        self.rtu_dim = getattr(self.gguf_backend, "dim", 384)
        self.rtu_layers = getattr(self.gguf_backend, "layers", 8)
        self.rtu_spread = getattr(self.gguf_backend, "spread", 32)

    @property
    def knowledge_bank(self) -> List[Dict[str, Any]]:
        return self.knowledge.get_audit_trail()

    # ------------------------------------------------------------------
    # Answer-path observability (change 1.19)
    # ------------------------------------------------------------------
    @property
    def answer_path(self) -> str:
        """Canonical path that produced the last answer: recall|fused|llm."""
        return self._last_answer_path

    @property
    def legacy_answer_path(self) -> str:
        """Read-only mapping for callers written against the T010 names."""
        return _LEGACY_ANSWER_PATH.get(self._last_answer_path, self._last_answer_path)

    def _set_answer_path(self, path: str) -> None:
        self._last_answer_path = path

    def clear_memory(self):
        """Wipe all learned state: knowledge store, plastic memory, and counters."""
        self.knowledge.clear_all()
        self.memory.reset()
        self.knowledge_vectors = None
        self.step_count = 0

    clear_knowledge = clear_memory

    def log(self, message: str, details: Optional[Dict[str, Any]] = None):
        entry = {"timestamp": time.time(), "message": message}
        if details:
            entry["details"] = details
        self.logs.append(entry)
        if len(self.logs) > 500:
            self.logs = self.logs[-500:]

    # ------------------------------------------------------------------ #
    # Single writer per profile (memory/sleep-consolidation)
    # ------------------------------------------------------------------ #
    @contextlib.contextmanager
    def writer_session(self, operation: str = "write"):
        """Hold the profile's writer lock for the duration of a write.

        Reentrant inside one thread (sleep replays through teach), and refuses a
        concurrent writer with ``ProfileBusyError`` instead of blocking.
        """
        depth = getattr(self._writer_tls, "depth", 0)
        if self.profile_dir is None:
            yield None
            return
        if depth:
            self._writer_tls.depth = depth + 1
            try:
                yield getattr(self._writer_tls, "lock", None)
            finally:
                self._writer_tls.depth -= 1
            return
        lock = ProfileWriterLock(self.profile_dir).acquire()
        self._writer_tls.lock = lock
        self._writer_tls.depth = 1
        try:
            yield lock
        finally:
            self._writer_tls.depth = 0
            self._writer_tls.lock = None
            lock.release()

    def enable_slow_tier(self, tolerance: float = 0.05):
        """Create the SLOW tier for this profile (idempotent)."""
        from recamem.plasticity import SlowTier

        if self.slow_tier is None:
            self.slow_tier = SlowTier(dim=self.dim, tolerance=tolerance)
        return self.slow_tier

    def slow_meta_state(self) -> Optional[Dict[str, Any]]:
        if self.slow_tier is None:
            return None
        return {
            "tolerance": self.slow_tier.tolerance,
            "weights": {k: v.detach().cpu().tolist()
                        for k, v in self.slow_tier.state_dict().items()},
            "train_log": list(self.slow_tier.train_log[-50:]),
        }

    def bind_profile_dir(self, target_dir: Union[str, Path]) -> None:
        """Bind the engine to a profile directory so writer locking applies."""
        self.profile_dir = Path(target_dir)

    def _split_propositions(self, text: str) -> List[str]:
        """Split taught text with the abbreviation-safe segmenter (defect D1).

        Titles (``Dr.``), initials and decimals are not sentence boundaries, so
        "The project lead is Dr. Mina Rostami." stays one proposition.
        """
        props = []
        for sentence in extract_propositions(
            text.strip(), min_length=5, filter_conversational=False
        ):
            s = sentence.strip()
            if len(s) < 5:
                continue
            if len(s.split()) < 3:
                continue
            props.append(s)
        return props

    def _resolve_anaphora(self, propositions: List[str]) -> List[str]:
        resolved = []
        for prop in propositions:
            m = _ANAPHORA_RE.match(prop)
            if m and self._discourse_subject:
                prop = re.sub(r'^(It|They|This|These|Those)\b', self._discourse_subject, prop, count=1)
            subj = _extract_subject(prop)
            if subj:
                self._discourse_subject = subj
            resolved.append(prop)
        return resolved

    def teach(self, text: str, source: str = "user_text",
              surprise_threshold: Optional[float] = None, *,
              origin: Optional[str] = None,
              source_url: Optional[str] = None,
              source_query: Optional[str] = None) -> Dict[str, Any]:
        """Teach under the shared engine lock (serialised with sleep/recall).

        ``origin``/``source_url``/``source_query`` carry provenance for facts
        acquired from an external source; ordinary teaching leaves them unset.
        """
        with self.writer_session("teach"), self._engine_lock:
            self.last_activity = time.time()
            return self._teach_impl(
                text, source, surprise_threshold,
                origin=origin, source_url=source_url, source_query=source_query,
            )

    def _teach_impl(self, text: str, source: str = "user_text",
                    surprise_threshold: Optional[float] = None, *,
                    origin: Optional[str] = None,
                    source_url: Optional[str] = None,
                    source_query: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if surprise_threshold is None:
            surprise_threshold = self.surprise_threshold
        propositions = self._split_propositions(text)
        if not propositions:
            return {"status": "no_propositions", "propositions": 0, "elapsed_ms": 0, "avg_novelty": 0, "memory_norm": 0, "total_knowledge": self.knowledge.active_count()}
        propositions = self._resolve_anaphora(propositions)
        effective_lr = self.base_lr / math.sqrt(1 + self.step_count / 20.0)
        novelty_losses = []
        superseded_total = 0
        episodic_absorbed = 0
        surprises = []
        update_results = []
        for i, prop in enumerate(propositions):
            try:
                e_i = self.anchor.embed(prop)
                if e_i.dim() > 1:
                    e_i = e_i.squeeze(0)
                e_i = e_i / torch.linalg.vector_norm(e_i).clamp(min=1e-6)
            except Exception as e:
                continue
            try:
                if hasattr(self.memory, 'compute_surprise'):
                    _, surprise = self.memory.compute_surprise(e_i)
                else:
                    surprise = 1.0
            except Exception:  # noqa: BLE001
                surprise = 1.0
            surprises.append(surprise)
            novelty_losses.append(surprise)
            should_insert = surprise >= surprise_threshold
            update_result = apply_plastic_update(
                self.memory, e_i, surprise_factor=surprise,
                effective_lr=effective_lr, base_lr=self.base_lr,
            )
            update_results.append(update_result)
            if should_insert:
                conflicts = []
                try:
                    conflicts = [c["id"] for c in self.knowledge.find_similar(e_i, 0.65) if _is_contradiction(prop, c["text"])]
                except Exception:  # noqa: BLE001
                    pass
                try:
                    new_id = self.knowledge.insert_fact(
                        prop, source, time.time(), self.step_count + i, float(surprise), e_i,
                        origin=origin, source_url=source_url, source_query=source_query,
                    )
                    episodic_absorbed += 1
                    for old_id in conflicts:
                        self.knowledge.supersede(old_id, new_id)
                    superseded_total += len(conflicts)
                except Exception as e:
                    pass
        self.step_count += len(propositions)
        elapsed = time.perf_counter() - t0
        avg_novelty = sum(novelty_losses) / max(1, len(novelty_losses))
        avg_surprise = sum(surprises) / max(1, len(surprises))
        try:
            if hasattr(self.memory, 'h_trace'):
                mem_norm = float(torch.norm(self.memory.h_trace).item())
            elif hasattr(self.memory, 'states'):
                mem_norm = float(torch.norm(self.memory.states).item())
            else:
                mem_norm = 0.0
        except Exception:  # noqa: BLE001
            mem_norm = 0.0
        applied_rates = [update.applied_lr for update in update_results]
        applied_lr = applied_rates[0] if applied_rates and all(
            rate == applied_rates[0] for rate in applied_rates
        ) else None
        result = {
            "status": "learned",
            "propositions": len(propositions),
            "episodic_absorbed": episodic_absorbed,
            "elapsed_ms": elapsed * 1000,
            "avg_novelty": avg_novelty,
            "avg_surprise": avg_surprise,
            "surprise_threshold": surprise_threshold,
            "memory_norm": mem_norm,
            "effective_lr": effective_lr,
            "applied_lr": applied_lr,
            "requested_lr": effective_lr,
            "applied_lrs": applied_rates,
            "scheduling_supported": bool(update_results) and all(
                update.scheduling_supported for update in update_results
            ),
            "total_knowledge": self.knowledge.active_count(),
            "superseded": superseded_total
        }
        self.log(f"Successfully learned {len(propositions)} facts in {elapsed*1000:.1f}ms. State norm: {mem_norm:.2f}.", details=result)
        return result

    def recall(self, query: str, top_k: int = 3, threshold: float = 0.50,
               strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._engine_lock:
            self.last_activity = time.time()
            return self._recall_impl(query, top_k, threshold, strategy)

    def _recall_impl(self, query: str, top_k: int = 3, threshold: float = 0.50,
                     strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.knowledge.total_count() == 0:
            return []
        query_type = "specific_fact"
        if self.retrieval_policy:
            try:
                query_type = self.retrieval_policy.classify_query(query)
                if strategy is None:
                    strategy = self.retrieval_policy.select_strategy(query_type)
                self._last_retrieval_strategy = strategy
            except Exception:  # noqa: BLE001
                pass
        if strategy is None:
            strategy = "hybrid_rrf"
        q_emb = self.anchor.embed(query)
        if q_emb.dim() > 1:
            q_emb = q_emb.squeeze(0)
        q_emb = q_emb / torch.linalg.vector_norm(q_emb).clamp(min=1e-6)
        # Associative query projection, computed once per recall inside a
        # guarded block. Absent or cold associative state (empty S) leaves the
        # bonus at zero, so a fresh profile ranks exactly as it did before.
        q_assoc = None
        try:
            retrieve = getattr(self.memory, "retrieve", None)
            S = getattr(self.memory, "S", None)
            cold = True
            if S is not None:
                cold = float(torch.linalg.vector_norm(S).item()) == 0.0
            if retrieve is not None and not cold:
                candidate = retrieve(q_emb)
                q_assoc = candidate / torch.linalg.vector_norm(candidate).clamp(min=1e-8)
        except Exception:
            q_assoc = None
        try:
            matches = self.knowledge.recall(q_emb, top_k=top_k*2, query_text=query, strategy=strategy)
        except TypeError:
            matches = self.knowledge.recall(q_emb, top_k=top_k*2, query_text=query)
        hits: List[Dict[str, Any]] = []
        h_trace = None
        try:
            if hasattr(self.memory, 'h_trace'):
                h_trace = self.memory.h_trace
            elif hasattr(self.memory, 'states'):
                h_trace = self.memory.states
        except Exception:  # noqa: BLE001
            pass
        weights = RECALL_SCORE_WEIGHTS
        for m in matches:
            fact_text = m.get("text", "")
            dense = float(m.get("dense_similarity") or m.get("similarity") or 0.0)
            passed_gate = dense >= DENSE_GATE
            fact_emb = None
            try:
                fact_emb = self.anchor.embed(fact_text)
                if fact_emb.dim() > 1:
                    fact_emb = fact_emb.squeeze(0)
                fact_emb = fact_emb / torch.linalg.vector_norm(fact_emb).clamp(min=1e-6)
            except Exception:
                fact_emb = None
            contextual = 0.0
            if h_trace is not None and fact_emb is not None:
                try:
                    h_norm = h_trace / (torch.linalg.vector_norm(h_trace).clamp(min=1e-8))
                    contextual = torch.dot(h_norm, fact_emb).item()
                except Exception:  # noqa: BLE001
                    contextual = 0.0
            # Associative bonus: relu(dot(q_assoc, fact_emb)) so a negative
            # association can never push a fact below one that passed on merit.
            # The associative bonus only re-ranks candidates that already pass
            # the dense gate (plan 10 decision 6); it can never resurrect an
            # out-of-domain hit on its own.
            associative = 0.0
            if passed_gate and q_assoc is not None and fact_emb is not None:
                try:
                    associative = max(0.0, float(torch.dot(q_assoc, fact_emb).item()))
                except Exception:
                    associative = 0.0
            lexical = _lexical_score(query, fact_text, query_type)
            recency = _recency_score(m.get("timestamp"))
            confidence = _confidence_score(m)
            final_score = score_candidate({
                "dense": dense, "lexical": lexical, "associative": associative,
                "recency": recency, "confidence": confidence, "contextual": contextual,
            }, weights)
            m["similarity"] = final_score
            m["dense_similarity"] = dense
            m["lexical_score"] = lexical
            m["associative_bonus"] = weights["associative"] * associative
            m["recency_score"] = recency
            m["confidence_score"] = confidence
            m["contextual_alignment"] = contextual
            m["original_similarity"] = dense
            m["dense_gate_pass"] = passed_gate
            if final_score >= threshold:
                hits.append(m)
        hits.sort(key=lambda x: (x["dense_gate_pass"], x["similarity"]), reverse=True)
        hits = hits[:top_k]
        ttt_decision = self._consume_ttt_step() if hits else {"allowed": False}
        if ttt_decision.get("allowed"):
            try:
                for hit in hits[:1]:
                    fact_emb = self.anchor.embed(hit["text"])
                    if fact_emb.dim() > 1:
                        fact_emb = fact_emb.squeeze(0)
                    conf = hit.get("similarity", 0.5)
                    if hasattr(self.memory, 'update_associative_memory'):
                        self.memory.update_associative_memory(fact_emb, surprise_factor=conf)
            except Exception as e:
                pass
        self._retrieval_history.append({
            "query": query,
            "query_type": query_type,
            "strategy": strategy,
            "hits": [h["id"] for h in hits],
            "timestamp": time.time()
        })
        if len(self._retrieval_history) > 100:
            self._retrieval_history = self._retrieval_history[-100:]
        return hits

    def _is_duplicate_or_subsumed(self, s1: str, s2: str) -> bool:
        s1_c = re.sub(r'[^\w\s]', '', s1.lower()).strip()
        s2_c = re.sub(r'[^\w\s]', '', s2.lower()).strip()
        if s1_c == s2_c or s1_c in s2_c or s2_c in s1_c:
            return True
        w1 = set(s1_c.split())
        w2 = set(s2_c.split())
        if not w1 or not w2:
            return False
        overlap = len(w1 & w2) / max(len(w1), len(w2))
        return overlap > 0.70

    def chat(self, user_message: str) -> str:
        self.log(f"Chat received: '{user_message}'")
        mode = "auto"
        query = user_message
        lower = user_message.lower().strip()
        if lower == "/sleep" or lower.startswith("/sleep "):
            # Command dispatch happens before teaching/retrieval so the command
            # text is never inserted as a knowledge fact.
            report = self.sleep()
            return self._format_sleep_report(report)
        if lower.startswith("/fast "):
            mode = "fast"
            query = user_message[5:].strip()
        elif lower.startswith("/slow "):
            mode = "slow"
            query = user_message[5:].strip()
        elif lower.startswith("/fast"):
            mode = "fast"
            query = user_message[5:].strip() if len(user_message)>5 else ""
        elif lower.startswith("/slow"):
            mode = "slow"
            query = user_message[5:].strip() if len(user_message)>5 else ""
        is_explicit_teach = lower.startswith(("learn:", "remember:", "note:", "teach:"))
        if is_explicit_teach:
            teach_content = re.sub(r'^(learn|remember|note|teach):\s*', '', user_message, flags=re.IGNORECASE)
            res = self.teach(teach_content, source="chat")
            return f"I have absorbed this into my memory ({res['propositions']} new facts, memory norm: {res['memory_norm']:.2f}). You can now ask me about it!"
        if mode == "auto" and hasattr(self, 'decide_path'):
            try:
                decision = self.decide_path(query)
                path = decision.get("path") or decision.get("action")
                if path in ("fast_recall", "fast", "recall", "chat"):
                    mode = "fast"
                elif path in ("slow_synthesis", "slow", "synthesis"):
                    mode = "slow"
            except Exception:  # noqa: BLE001
                pass
        if mode == "fast":
            hits = self.recall(query, top_k=4, threshold=0.45)
            if not hits and self.last_query:
                contextual_query = f"{self.last_query} {query}"
                hits = self.recall(contextual_query, top_k=4, threshold=0.40)
                if hits:
                    self.log(f"Resolved follow-up query using previous context: '{self.last_query}'.")
            if hits:
                self.last_query = query
                explanation_parts: List[str] = []
                for h in hits:
                    t = h["text"].strip()
                    if not any(self._is_duplicate_or_subsumed(t, existing) for existing in explanation_parts):
                        explanation_parts.append(t)
                synthesized = " ".join(explanation_parts)
                self.log(f"Factual recall hit with similarity {hits[0]['similarity']:.3f}.")
                self._set_answer_path(ANSWER_PATH_RECALL)
                return f"{synthesized}"
            else:
                self.log("No relevant facts found above threshold.")
                researched = self._research_chat_answer(query)
                if researched:
                    return researched
                self._set_answer_path(ANSWER_PATH_RECALL)
                return NO_INFORMATION_MESSAGE
        if mode == "slow":
            if self.gguf_backend is not None and hasattr(self, 'generate_with_memory'):
                result = None
                try:
                    result = self.generate_with_memory(query, strategy="auto")
                except Exception as e:
                    self.log(f"Slow path failed: {e}")
                    result = None
                if isinstance(result, dict):
                    text = result.get("text", "")
                    tier = result.get("confidence_tier", "medium")
                    if isinstance(self.gguf_backend, ByteRtuBackend) and not str(text).strip():
                        # The frozen component emitted nothing: grounded
                        # silence beats an empty answer.
                        self.log("Fused backend produced no bytes; using fallback.")
                        self._set_answer_path(ANSWER_PATH_RECALL)
                        return NO_INFORMATION_MESSAGE
                    # A byte-RTU answer is the fused path; anything else that
                    # generated text is an LLM answer.
                    self._set_answer_path(
                        ANSWER_PATH_FUSED
                        if isinstance(self.gguf_backend, ByteRtuBackend)
                        else ANSWER_PATH_LLM
                    )
                    if tier == "low":
                        return f"I'm uncertain; here's what I found... {text}"
                    return text
                if isinstance(result, str):
                    # No facts -> no free generation (decision 5).
                    self._set_answer_path(ANSWER_PATH_RECALL)
                    return result
        hits = self.recall(query, top_k=4, threshold=0.45)
        if not hits and self.last_query:
            contextual_query = f"{self.last_query} {query}"
            hits = self.recall(contextual_query, top_k=4, threshold=0.40)
            if hits:
                self.log(f"Resolved follow-up query using previous context: '{self.last_query}'.")
        if hits:
            self.last_query = query
            explanation_parts: List[str] = []
            for h in hits:
                t = h["text"].strip()
                if not any(self._is_duplicate_or_subsumed(t, existing) for existing in explanation_parts):
                    explanation_parts.append(t)
            synthesized = " ".join(explanation_parts)
            self.log(f"Factual recall hit with similarity {hits[0]['similarity']:.3f}.")
            self._set_answer_path(ANSWER_PATH_RECALL)
            return f"{synthesized}"
        self.log("No relevant facts found above threshold.")
        researched = self._research_chat_answer(query)
        if researched:
            return researched
        self._set_answer_path(ANSWER_PATH_RECALL)
        return NO_INFORMATION_MESSAGE

    def _frozen_fingerprint(self) -> str:
        """Stable hash of frozen components (anchor + generative decoder).

        Consolidation is allowed to touch plastic parameters only; comparing
        this before and after proves the frozen path was not modified.
        """
        digest = hashlib.sha256()
        targets = [("anchor", getattr(self, "anchor", None))]
        backend = getattr(self, "gguf_backend", None)
        model = getattr(backend, "model", None)
        if model is not None:
            targets.append(("decoder", model))
        for label, module in targets:
            if module is None:
                continue
            params = getattr(module, "parameters", None)
            state = None
            if callable(params):
                try:
                    state = {name: param.detach() for name, param in module.named_parameters()}
                except Exception:
                    state = None
            if isinstance(state, dict):
                digest.update(label.encode())
                for name in sorted(state):
                    tensor = state[name].detach().float().cpu().contiguous()
                    digest.update(name.encode())
                    digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def consolidation_objective(self, sample: int = 16) -> Dict[str, Any]:
        """Measure reconstruction fidelity and orthogonality of plastic state.

        Both numbers are computed, never assumed: ``None`` means the adapter
        offers no such measurement, and the report says so instead of inventing
        a value.
        """
        objective: Dict[str, Any] = {
            "samples": 0,
            "reconstruction_cosine": None,
            "orthogonality_loss": None,
        }
        try:
            vecs, ids = self.knowledge._load_dense()
        except Exception:
            return objective
        if len(ids) == 0:
            return objective
        x = vecs[: min(sample, vecs.shape[0])].detach().float()
        objective["samples"] = int(x.shape[0])
        retrieve = getattr(self.memory, "retrieve", None)
        if callable(retrieve):
            try:
                with torch.no_grad():
                    outputs = torch.stack([retrieve(row) for row in x])
                    objective["reconstruction_cosine"] = float(
                        torch.nn.functional.cosine_similarity(outputs, x, dim=-1).mean()
                    )
            except Exception:
                objective["reconstruction_cosine"] = None
        S = getattr(self.memory, "S", None)
        if S is not None:
            try:
                with torch.no_grad():
                    total = 0.0
                    for head in S:
                        if head.shape[0] != head.shape[1]:
                            continue
                        eye = torch.eye(head.shape[0], dtype=head.dtype, device=head.device)
                        total += float(torch.linalg.matrix_norm(head.T @ head - eye))
                    objective["orthogonality_loss"] = total
            except Exception:
                objective["orthogonality_loss"] = None
        return objective

    def consolidate_memory(self, max_prune_ratio: float = 0.1) -> Dict[str, Any]:
        """Bounded consolidation: measured objective change + audit-safe pruning.

        * The reconstruction/orthogonality objective is measured before and
          after the plastic update, and the delta is reported (4.8).
        * The update runs through the adapter's own ``consolidate()`` when it
          provides one, so only eligible plastic parameters move.
        * Pruning is capped by ``max_prune_ratio`` and only *supersedes* rows,
          so the audit trail and the reversible pre-cycle snapshot keep the
          evidence.
        """
        total = self.knowledge.active_count()
        if total == 0:
            return {
                "status": "no_facts",
                "pruned": 0,
                "total_before": 0,
                "total_after": 0,
                "objective": {"before": None, "after": None, "improved": False},
            }
        facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
        now = time.time()
        tau_half = 86400.0
        lambda_novelty = 0.5
        utilities = []
        for f in facts:
            access = f.get("access_count", 1)
            last = f.get("last_retrieved")
            if last is None:
                last = f.get("timestamp")
            if last is None:
                last = now
            recency = math.exp(-(now - last) / tau_half)
            novelty = f.get("novelty", 0.5)
            utilities.append((f, access * recency + lambda_novelty * novelty))
        utilities.sort(key=lambda item: item[1])

        frozen_before = self._frozen_fingerprint()
        before = self.consolidation_objective()

        plastic_report: Dict[str, Any] = {"status": "skipped", "updated": False}
        consolidate = getattr(self.memory, "consolidate", None)
        if callable(consolidate):
            try:
                plastic_report = dict(consolidate() or {})
            except Exception as exc:
                plastic_report = {"status": "failed", "updated": False, "error": str(exc)}
        if not plastic_report.get("updated"):
            S = getattr(self.memory, "S", None)
            if S is not None:
                try:
                    with torch.no_grad():
                        for head in S:
                            if head.shape[0] != head.shape[1]:
                                continue
                            eye = torch.eye(head.shape[0], dtype=head.dtype, device=head.device)
                            grad = 2 * head @ (head.T @ head - eye)
                            head.sub_(1e-4 * grad)
                    plastic_report = {
                        "status": "consolidated",
                        "updated": True,
                        "method": "orthogonality_regularisation",
                    }
                except Exception as exc:
                    plastic_report = {"status": "failed", "updated": False, "error": str(exc)}

        after = self.consolidation_objective()
        improved = False
        if (before.get("orthogonality_loss") is not None
                and after.get("orthogonality_loss") is not None):
            improved = after["orthogonality_loss"] < before["orthogonality_loss"]
        elif (before.get("reconstruction_cosine") is not None
              and after.get("reconstruction_cosine") is not None):
            improved = after["reconstruction_cosine"] > before["reconstruction_cosine"]

        ratio = max(0.0, min(1.0, float(max_prune_ratio)))
        prune_count = int(total * ratio)
        pruned = 0
        if prune_count > 0:
            for fact, util in utilities[:prune_count]:
                if util >= 0.5:
                    continue
                try:
                    # Audit-safe: the row stays queryable through get_audit_trail.
                    self.knowledge.conn.execute(
                        "UPDATE facts SET superseded=1, valid_until=?, needs_review=1 WHERE id=?",
                        (now, fact["id"]),
                    )
                    pruned += 1
                except Exception:
                    continue
            self.knowledge.conn.commit()

        return {
            "status": "consolidated",
            "total_before": total,
            "pruned": pruned,
            "total_after": self.knowledge.active_count(),
            "pruning": {
                "ratio": ratio,
                "cap": prune_count,
                "bounded": pruned <= prune_count,
                "audit_retained": True,
                "utility_threshold": 0.5,
            },
            "plastic": plastic_report,
            "objective": {"before": before, "after": after, "improved": improved},
            "frozen_unchanged": self._frozen_fingerprint() == frozen_before,
            "recon_loss": before.get("orthogonality_loss"),
        }

    # ------------------------------------------------------------------
    # Sleep consolidation / MAS / decreasing backprop
    # ------------------------------------------------------------------
    def configure_sleep(self, **settings) -> Dict[str, Any]:
        """Update the profile's sleep policy; automatic sleep stays opt-in."""
        unknown = set(settings) - set(DEFAULT_SLEEP_CONFIG)
        if unknown:
            raise ValueError(f"Unknown sleep settings: {sorted(unknown)}")
        schedule = settings.pop("schedule", None)
        for key, value in settings.items():
            self.sleep_config[key] = value
        if schedule is not None:
            self.sleep_config["schedule"] = {**DEFAULT_SLEEP_CONFIG["schedule"], **schedule}
        cfg = self.sleep_config["schedule"]
        self.lr_schedule = DecreasingBackprop(
            base_lr=float(cfg.get("base_lr", self.base_lr)),
            decay_factor=float(cfg.get("decay_factor", 0.95)),
            index=int(cfg.get("index", self.lr_schedule.index)),
            experimental_relu_down=bool(cfg.get("experimental_relu_down", False)),
        )
        return dict(self.sleep_config)

    def _retrieval_quality(self, sample: int = 8) -> Optional[float]:
        """Fraction of sampled facts recovered in their own top-3 recall."""
        facts = self.knowledge.get_active_facts()[:sample]
        if not facts:
            return None
        hits = 0
        for fact in facts:
            try:
                emb = self.anchor.embed(fact["text"])
                if emb.dim() > 1:
                    emb = emb.squeeze(0)
                results = self.knowledge.recall(
                    emb, top_k=3, query_text=fact["text"], strategy="hybrid_rrf"
                )
                if any(int(r["id"]) == int(fact["id"]) for r in results):
                    hits += 1
            except Exception:
                return None
        return hits / len(facts)

    def sleep(self, cancel_event: Optional[threading.Event] = None,
              **overrides) -> Dict[str, Any]:
        """Run one bounded consolidation cycle under the profile writer lock.

        A concurrent writer is refused with ``ProfileBusyError`` rather than
        waiting, and no state is touched by the refused call.
        """
        with self.writer_session("sleep"):
            return self._sleep_impl(cancel_event, **overrides)

    def _sleep_impl(self, cancel_event: Optional[threading.Event] = None,
                    **overrides) -> Dict[str, Any]:
        """Run one bounded consolidation cycle and return a structured report.

        Replay, fact pruning and parameter pruning are reported separately so
        an empty phase is never mistaken for work performed.
        """
        started = time.perf_counter()
        cfg = dict(self.sleep_config)
        cfg.update(overrides)
        warnings: List[str] = []
        errors: List[str] = []
        report: Dict[str, Any] = {
            "status": "completed",
            "duration_ms": 0.0,
            "cycles": self.sleep_cycles,
            "replay": {
                "status": "not_run", "steps": 0, "unique_recovered": 0,
                "duplicates": 0, "coverage": 0.0, "replay_writes": 0,
            },
            "fact_pruning": {"status": "not_run", "pruned": 0, "total_before": 0,
                             "total_after": 0},
            "parameter_pruning": {"status": "disabled", "selected": 0, "total": 0},
            "learning_rate": self.lr_schedule.diagnostics(),
            "updates": {
                "proposed": 0, "committed": 0, "rejected": 0,
                "rejected_domain": None, "reason": None,
            },
            "warnings": warnings,
            "errors": errors,
        }
        if cancel_event is not None and cancel_event.is_set():
            report["status"] = "cancelled"
            report["duration_ms"] = (time.perf_counter() - started) * 1000
            self.last_sleep_report = report
            return report
        with self._engine_lock:
            total_facts = self.knowledge.active_count()
            if total_facts == 0:
                report["status"] = "no_op_empty"
                report["fact_pruning"].update(
                    {"status": "no_facts", "total_before": 0, "total_after": 0}
                )
                report["replay"]["status"] = "no_patterns"
                report["duration_ms"] = (time.perf_counter() - started) * 1000
                self.last_sleep_report = report
                return report

            # Phase 1: autonomous replay (never receives expected patterns).
            try:
                consolidator = SleepConsolidation(
                    self,
                    max_replay_steps=int(cfg.get("max_replay_steps", 128)),
                    max_seconds=float(cfg.get("max_seconds", 10.0)),
                    min_similarity=float(cfg.get("min_similarity", 0.0)),
                    duplicate_cap=int(cfg.get("duplicate_cap", 2)),
                    inhibition_strength=float(cfg.get("inhibition_strength", 0.6)),
                    inhibition_decay=float(cfg.get("inhibition_decay", 0.5)),
                    lr=self.lr_schedule.current_lr,
                    base_lr=max(self.base_lr, 1e-9),
                )
                replay = consolidator.autonomous_retrieval(cancel_event)
                report["replay"] = replay
                warnings.extend(replay.get("warnings", []))
                if replay.get("status") == "cancelled":
                    report["status"] = "cancelled"
            except Exception as exc:
                errors.append(f"replay: {exc}")
                report["replay"]["status"] = "failed"

            # Phase 2: MAS parameter pruning (opt-in).
            if cfg.get("mas_enabled"):
                try:
                    vecs, _ids = self.knowledge._load_dense()
                    inputs = vecs[: min(32, vecs.shape[0])]
                    mas = MemoryAwareSynapses(
                        self.memory,
                        ratio=float(cfg.get("parameter_prune_ratio", 0.1)),
                        quality_fn=self._retrieval_quality,
                    )
                    mas.estimate_importance(inputs)
                    report["parameter_pruning"] = mas.prune()
                    self.mas = mas
                    if report["parameter_pruning"].get("warning"):
                        warnings.append(report["parameter_pruning"]["warning"])
                except Exception as exc:
                    errors.append(f"parameter_pruning: {exc}")
                    report["parameter_pruning"] = {"status": "failed", "selected": 0,
                                                   "total": 0, "error": str(exc)}
            else:
                report["parameter_pruning"] = {
                    "status": "disabled", "selected": 0, "total": 0,
                    "reason": "mas_enabled is false by default",
                }

            # Phase 3: fact-utility pruning (separate counts from parameters).
            if report["status"] != "cancelled":
                try:
                    fact_report = self.consolidate_memory(
                        max_prune_ratio=float(cfg.get("fact_prune_ratio", 0.1))
                    )
                    report["fact_pruning"] = {
                        "status": fact_report.get("status", "unknown"),
                        "pruned": int(fact_report.get("pruned", 0)),
                        "total_before": int(fact_report.get("total_before", total_facts)),
                        "total_after": int(fact_report.get(
                            "total_after", self.knowledge.active_count())),
                    }
                except Exception as exc:
                    errors.append(f"fact_pruning: {exc}")
                    report["fact_pruning"] = {"status": "failed", "pruned": 0,
                                              "total_before": total_facts,
                                              "total_after": total_facts}

            # A committed cycle advances the persisted schedule exactly once.
            committed = (
                not errors and report["status"] == "completed"
                and int(report["replay"].get("steps", 0)) > 0
            )
            if committed:
                self.lr_schedule.advance()
                self.sleep_cycles += 1
            self.sleep_config["schedule"] = self.lr_schedule.state_dict()
            report["cycles"] = self.sleep_cycles
            report["learning_rate"] = self.lr_schedule.diagnostics()
            proposed = int(report["replay"].get("steps", 0))
            report["updates"] = {
                "proposed": proposed,
                "committed": proposed if committed else 0,
                "rejected": 0 if committed else proposed,
                "rejected_domain": None if committed else (
                    "replay" if proposed else None
                ),
                "reason": None if committed else (
                    "no replay step was applied" if not proposed
                    else "cycle did not commit because of errors"
                ),
            }
            if not committed and proposed and report["status"] == "completed":
                report["warnings"].append(
                    "replay produced no committed parameter update"
                )
            if errors:
                report["status"] = "partial_failure"
            report["duration_ms"] = (time.perf_counter() - started) * 1000
            self.last_sleep_report = report
            self.log(f"Sleep cycle {self.sleep_cycles} finished: {report['status']}",
                     details={k: report[k] for k in ("replay", "fact_pruning",
                                                     "parameter_pruning")})
            return report

    @staticmethod
    def _format_sleep_report(report: Dict[str, Any]) -> str:
        replay = report.get("replay", {})
        facts = report.get("fact_pruning", {})
        params = report.get("parameter_pruning", {})
        return (
            f"Sleep cycle {report.get('status')} in "
            f"{report.get('duration_ms', 0.0):.1f} ms: "
            f"replayed {replay.get('steps', 0)} patterns "
            f"({replay.get('unique_recovered', 0)} unique, coverage "
            f"{replay.get('coverage', 0.0):.2f}); pruned {facts.get('pruned', 0)} facts "
            f"and {params.get('selected', 0)} plastic parameters "
            f"(lr index {report.get('learning_rate', {}).get('i', 0)}, "
            f"lr {report.get('learning_rate', {}).get('lr_i', 0.0):.4f})."
        )

    # ------------------------------------------------------------------ #
    # Test-time training budget (post-merge 4.10)
    # ------------------------------------------------------------------ #
    def ttt_status(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.test_time_training),
            "used": int(self.ttt_steps_this_session),
            "max_steps": int(self.max_ttt_steps),
            "attempts": int(self.ttt_attempts),
            "skipped": int(self.ttt_skipped),
            "rate": float(self.ttt_rate),
            "remaining": max(0, int(self.max_ttt_steps) - int(self.ttt_steps_this_session)),
            "last_skip_reason": self.ttt_last_skip_reason,
        }

    def _consume_ttt_step(self) -> Dict[str, Any]:
        """Reserve one TTT update; shared by recall and every feedback path.

        Returns ``{"allowed": bool, "reason": str|None, "used": int}``. A denied
        step never mutates plastic state, so the budget is a hard bound and not
        a best-effort hint.
        """
        self.ttt_attempts += 1
        if not self.test_time_training:
            self.ttt_skipped += 1
            self.ttt_last_skip_reason = "opt_out"
            return {"allowed": False, "reason": "opt_out",
                    "used": self.ttt_steps_this_session}
        if self.ttt_steps_this_session >= self.max_ttt_steps:
            self.ttt_skipped += 1
            self.ttt_last_skip_reason = "exhausted"
            return {"allowed": False, "reason": "exhausted",
                    "used": self.ttt_steps_this_session}
        if self.ttt_rate < 1.0 and (self.ttt_attempts % max(1, int(round(1.0 / self.ttt_rate)))) != 0:
            self.ttt_skipped += 1
            self.ttt_last_skip_reason = "rate_limited"
            return {"allowed": False, "reason": "rate_limited",
                    "used": self.ttt_steps_this_session}
        self.ttt_steps_this_session += 1
        self.ttt_last_skip_reason = None
        return {"allowed": True, "reason": None, "used": self.ttt_steps_this_session}

    def _erase_corrected_fact(self, old_fact_id: int) -> bool:
        """Remove a corrected fact's association from plastic memory."""
        try:
            row = self.knowledge.conn.execute(
                "SELECT text FROM facts WHERE id=?", (int(old_fact_id),)
            ).fetchone()
            if not row:
                return False
            emb = self.anchor.embed(row[0])
            if emb.dim() > 1:
                emb = emb.squeeze(0)
            emb = emb / torch.linalg.vector_norm(emb).clamp(min=1e-6)
            erase = getattr(self.memory, "erase", None)
            if callable(erase):
                erase(emb, strength=1.0)
                return True
        except Exception:
            return False
        return False

    def get_memory_health(self) -> Dict[str, Any]:
        total = self.knowledge.total_count()
        active = self.knowledge.active_count()
        saturation = active / max(1, self.K_max)
        associative_norm = 0.0
        try:
            if hasattr(self.memory, 'S'):
                associative_norm = float(torch.norm(self.memory.S).item())
            elif hasattr(self.memory, 'states'):
                associative_norm = float(torch.norm(self.memory.states).item())
            saturation += associative_norm / 100.0
            saturation = min(1.0, saturation)
        except Exception:  # noqa: BLE001
            pass
        forgetting_rate = 0.0
        try:
            facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
            if facts:
                now = time.time()
                threshold = 0.1
                low_access = 0
                for f in facts:
                    access = f.get("access_count", 1)
                    last = f.get("last_retrieved") or f.get("timestamp") or now
                    acc = access * math.exp(-(now-last)/86400.0)
                    if acc < threshold:
                        low_access += 1
                forgetting_rate = low_access / len(facts)
        except Exception:  # noqa: BLE001
            pass
        interference = 0.0
        try:
            vecs, ids = self.knowledge._load_dense()
            if len(ids) > 1:
                sample_size = min(100, len(ids))
                if len(ids) > 1000:
                    import random
                    indices = random.sample(range(len(ids)), sample_size)
                    sample_vecs = vecs[indices]
                else:
                    sample_vecs = vecs[:sample_size]
                sims = sample_vecs @ sample_vecs.T
                mask = ~torch.eye(sims.shape[0], dtype=torch.bool)
                mean_sim = sims[mask].mean().item()
                interference = max(0.0, mean_sim)
        except Exception:  # noqa: BLE001
            pass
        # Retrieval quality comes from labelled judgements only (4.9): a real
        # precision@k / MRR over persisted feedback, never a constant guess.
        retrieval_health: Dict[str, Any] = {
            "available": False,
            "reason": "not_computed",
            "samples": 0,
            "precision_at_k": None,
            "mrr": None,
        }
        try:
            retrieval_health = self.knowledge.retrieval_health()
        except Exception as exc:
            retrieval_health = {
                "available": False,
                "reason": f"error: {exc}",
                "samples": 0,
                "precision_at_k": None,
                "mrr": None,
            }
        retrieval_quality: Optional[float] = (
            float(retrieval_health["precision_at_k"])
            if retrieval_health.get("available") and retrieval_health.get("precision_at_k") is not None
            else None
        )
        # Approve/reject counts are a separate, unweighted signal: they say how
        # often users accepted a shown fact, not how well ranking performed.
        approval_rate: Optional[float] = None
        approval_samples = 0
        try:
            feedbacks = self.knowledge.get_feedback_log()
            approves = sum(1 for fb in feedbacks if fb["action"] == "approve")
            rejects = sum(1 for fb in feedbacks if fb["action"] == "reject")
            approval_samples = approves + rejects
            if approval_samples:
                approval_rate = approves / approval_samples
        except Exception:
            pass
        # Eviction tracking from capacity enforcement (tasks 2.6, 2.7)
        evictions = 0
        key_count = 0
        try:
            if hasattr(self.memory, 'eviction_count'):
                evictions = int(getattr(self.memory, 'eviction_count', 0))
            if hasattr(self.memory, '_key_registry'):
                key_count = len(getattr(self.memory, '_key_registry', []))
            elif hasattr(self.memory, 'S'):
                # fallback: estimate from active facts if registry not present
                key_count = active
        except Exception:  # noqa: BLE001
            pass
        return {
            "saturation": saturation,
            "associative_norm": associative_norm,
            "forgetting_rate": forgetting_rate,
            "interference": interference,
            "retrieval_quality": retrieval_quality,
            "retrieval_health": retrieval_health,
            "approval_rate": approval_rate,
            "approval_samples": approval_samples,
            "ttt": self.ttt_status(),
            "active_facts": active,
            "total_facts": total,
            "K_max": self.K_max,
            "evictions": evictions,
            "eviction_count": evictions,
            "key_count": key_count,
            "active_key_count": key_count,
        }

    def create_snapshot(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        """Snapshot tensors, the complete database and lifecycle state (4.4).

        Delegates to :class:`SnapshotManager`, which writes the manifest last
        and never aliases live tensors.
        """
        if version_id is None:
            version_id = f"v{int(time.time())}"
        if self.snapshot_manager is not None:
            snapshot = self.snapshot_manager.create_snapshot(
                self, version_id, parent_version=self.current_version
            )
        else:  # pragma: no cover - SnapshotManager is always importable
            plastic_state = (
                clone_state(self.memory.state_dict())
                if hasattr(self.memory, "state_dict") else {}
            )
            snapshot = {
                "plastic_state": plastic_state,
                "metadata": {
                    "version_id": version_id,
                    "timestamp": time.time(),
                    "step_count": self.step_count,
                    "fact_count": self.knowledge.active_count(),
                    "parent_version": self.current_version,
                    "health_metrics": {},
                },
                "persisted": False,
                "artifacts": {},
            }
        self.snapshots[version_id] = snapshot
        self.current_version = version_id
        return snapshot

    def restore_snapshot(self, snapshot_dict_or_version: Union[Dict, str]) -> bool:
        """Staged restore: validate everything before activating (4.4/4.5)."""
        if isinstance(snapshot_dict_or_version, str):
            version_id = snapshot_dict_or_version
        else:
            snapshot = snapshot_dict_or_version
            version_id = (
                snapshot.get("version_id")
                or snapshot.get("version")
                or snapshot.get("metadata", {}).get("version_id")
            )
            self.snapshots.setdefault(version_id, snapshot)
        if self.snapshot_manager is not None:
            if self.snapshot_manager.restore_snapshot(self, version_id):
                self.current_version = version_id
                return True
            # Fall back to in-memory payloads for snapshots created before the
            # manager owned persistence; never silently succeed otherwise.
            if version_id not in self.snapshots:
                return False
        snapshot = self.snapshots.get(version_id)
        if snapshot is None:
            return False
        plastic_state = snapshot.get("plastic_state", {})
        try:
            if hasattr(self.memory, "load_state_dict") and plastic_state:
                self.memory.load_state_dict(clone_state(plastic_state))
        except Exception:
            return False
        meta = snapshot.get("metadata", {})
        self.step_count = meta.get("step_count", self.step_count)
        self.current_version = version_id
        return True

    def rollback_to(self, version_id: str) -> bool:
        """Reversible rollback with a pre-rollback snapshot."""
        return bool(self.rollback(version_id).get("restored"))

    def rollback(self, version_id: str) -> Dict[str, Any]:
        """Roll back through the shared staged path; the live state is saved first."""
        if self.snapshot_manager is None:  # pragma: no cover
            restored = self.restore_snapshot(version_id)
            return {
                "restored": restored,
                "version_id": version_id,
                "pre_rollback_version": None,
                "error": None if restored else "snapshot not found",
            }
        before = self.current_version
        report = self.snapshot_manager.rollback(self, version_id)
        if report.get("restored"):
            self.current_version = f"{version_id}_branch_{int(time.time())}"
            report["restored_from_version"] = before
        return report

    def branch_from(self, version_id: str, new_version_id: str) -> Optional[Dict]:
        if self.snapshot_manager is not None:
            branched = self.snapshot_manager.branch_from(version_id, new_version_id, self)
            if branched is not None:
                self.snapshots[new_version_id] = branched
                self.current_version = new_version_id
                return branched
        snapshot = self.snapshots.get(version_id)
        if snapshot is None:
            if self.snapshot_manager:
                snapshot = self.snapshot_manager.snapshots.get(version_id)
            if snapshot is None:
                return None
        new_snapshot = {"plastic_state": clone_state(snapshot.get("plastic_state", {})), "metadata": {"version_id": new_version_id, "timestamp": time.time(), "step_count": snapshot.get("metadata", {}).get("step_count", 0), "fact_count": snapshot.get("metadata", {}).get("fact_count", 0), "parent_version": version_id, "branch_from": version_id}, "version": new_version_id, "timestamp": time.time(), "step_count": snapshot.get("metadata", {}).get("step_count", 0), "fact_count": snapshot.get("metadata", {}).get("fact_count", 0)}
        self.snapshots[new_version_id] = new_snapshot
        self.current_version = new_version_id
        return new_snapshot

    def list_snapshots(self) -> List[Dict]:
        if self.snapshot_manager is not None:
            try:
                return self.snapshot_manager.list_snapshots(self)
            except Exception:
                pass
        result = []
        for vid, snap in self.snapshots.items():
            meta = snap.get("metadata", {})
            if not meta:
                meta = {"version_id": vid, "timestamp": snap.get("timestamp",0), "step_count": snap.get("step_count",0), "fact_count": snap.get("fact_count",0)}
            result.append(meta)
        result.sort(key=lambda x: x.get("timestamp",0))
        return result

    def _profile_fingerprint(self) -> str:
        """Hash of live plastic tensors and episodic facts (immutability probe)."""
        digest = hashlib.sha256()
        try:
            state = self.memory.state_dict()
        except Exception:
            state = {}
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                digest.update(key.encode())
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        for row in self.knowledge.conn.execute(
            "SELECT id, text, superseded FROM facts ORDER BY id"
        ):
            digest.update(repr(tuple(row)).encode())
        return digest.hexdigest()

    def merge_memory(self, other_droid, policy: str = "require_manual") -> Dict[str, Any]:
        """Fuse another profile into this one without corrupting either side.

        Ordering (post-merge 4.6): policy → compatibility → conflict
        resolution → writes. A manual policy with unresolved conflicts and an
        incompatible adapter/dimension both return a refusal *before* any write
        to either store.
        """
        if MergeEngine is None:  # pragma: no cover - import fallback
            return {"error": "MergeEngine not available", "status": "unavailable",
                    "writes": 0}
        try:
            MergeEngine.validate_policy(policy)
        except ValueError as exc:
            return {"status": "rejected", "error": str(exc), "writes": 0,
                    "conflicts": [], "new_version": None}
        self_dim = getattr(self, "dim", None)
        other_dim = getattr(other_droid, "dim", None)
        if self_dim is not None and other_dim is not None and int(self_dim) != int(other_dim):
            return {
                "status": "rejected",
                "error": f"incompatible embedding dimensions: {self_dim} vs {other_dim}",
                "writes": 0,
                "conflicts": [],
                "new_version": None,
            }
        self_adapter = self.adapter_name or identify_adapter(self.memory)
        other_adapter = getattr(other_droid, "adapter_name", None) or identify_adapter(
            other_droid.memory
        )
        if self_adapter and other_adapter and self_adapter != other_adapter:
            return {
                "status": "rejected",
                "error": f"incompatible plastic adapters: {self_adapter} vs {other_adapter}",
                "writes": 0,
                "conflicts": [],
                "new_version": None,
            }
        source_before = other_droid._profile_fingerprint()
        self_before = self._profile_fingerprint()
        try:
            merge_result = MergeEngine.merge_memories(
                self.knowledge, other_droid.knowledge, policy=policy
            )
            conflicts = merge_result.get("conflicts", [])
            if policy == "require_manual" and conflicts:
                return {
                    "status": "manual_required",
                    "error": "conflicts require explicit resolution; nothing was written",
                    "conflicts": conflicts,
                    "conflict_count": len(conflicts),
                    "resolutions": [],
                    "writes": 0,
                    "new_version": None,
                    "source_unchanged": other_droid._profile_fingerprint() == source_before,
                }
            writes = 0
            skipped = 0
            duplicates = 0
            invalid = 0
            try:
                weight_a = self.knowledge.active_count()
                weight_b = other_droid.knowledge.active_count()
                total = weight_a + weight_b
                w_a, w_b = (0.5, 0.5) if total == 0 else (weight_a / total, weight_b / total)
                fused = MergeEngine.fuse_plastic(
                    self.memory, other_droid.memory, weight_a=w_a, weight_b=w_b
                )
                if fused.get("S") is not None and hasattr(self.memory, "S"):
                    with torch.no_grad():
                        S_fused = fused["S"]
                        if S_fused.shape == self.memory.S.shape:
                            self.memory.S.copy_(S_fused)
                        if "h_trace" in fused and hasattr(self.memory, "h_trace"):
                            h_fused = fused["h_trace"]
                            if h_fused.shape == self.memory.h_trace.shape:
                                self.memory.h_trace.copy_(h_fused)
                        if "momentum" in fused and hasattr(self.memory, "momentum"):
                            mom_fused = fused["momentum"]
                            if mom_fused.shape == self.memory.momentum.shape:
                                self.memory.momentum.copy_(mom_fused)
            except Exception as exc:
                return {
                    "status": "rejected",
                    "error": f"plastic fusion failed: {exc}",
                    "writes": 0,
                    "conflicts": conflicts,
                    "new_version": None,
                    "source_unchanged": other_droid._profile_fingerprint() == source_before,
                }
            # One read of the source, then one batched write. The previous
            # loop committed per fact and swallowed every error, so a merge
            # could take minutes and still report "merged".
            rows = other_droid.knowledge.conn.execute(
                "SELECT f.id, f.text, f.source, f.novelty, f.origin, v.vec"
                " FROM facts f JOIN facts_vecs v ON v.id = f.id"
                " WHERE f.superseded = 0"
            ).fetchall()
            pending = []
            for fact_id, text, source, novelty, origin, blob in rows:
                try:
                    vector = torch.frombuffer(bytearray(blob), dtype=torch.float32).reshape(
                        self.dim
                    )
                except (RuntimeError, ValueError):
                    skipped += 1
                    continue
                pending.append({
                    "text": text,
                    "source": source or "merge",
                    "timestamp": time.time(),
                    "step": self.step_count,
                    "novelty": novelty,
                    "origin": origin,
                    "vec": vector,
                })
            report = self.knowledge.bulk_insert_facts(pending, dedupe_threshold=0.9)
            writes = report["inserted"]
            duplicates = report["duplicate"]
            invalid = report["invalid"]
            new_version = f"merge_{int(time.time())}"
            self.create_snapshot(new_version)
            # Auditable merge (memory/merge-engine#Merge audit trail): the new
            # version records both sources, the policy and every resolution.
            audit = {
                "merge_of": [self.name, other_droid.name],
                "new_version": new_version,
                "policy": policy,
                "conflict_count": len(conflicts),
                "resolutions": merge_result.get("resolutions", []),
                "writes": writes,
                "at": time.time(),
            }
            self.merge_history = list(getattr(self, "merge_history", [])) + [audit]
            self.last_merge = audit
            self.current_version = new_version
            return {
                "status": "merged",
                "new_version": new_version,
                "merge_of": audit["merge_of"],
                "merge_audit": audit,
                "conflicts": conflicts,
                "conflict_count": len(conflicts),
                "resolutions": merge_result.get("resolutions", []),
                "writes": writes,
                "duplicates_skipped": duplicates,
                "invalid_skipped": invalid + skipped,
                "source_unchanged": other_droid._profile_fingerprint() == source_before,
                "target_preexisting": self_before,
            }
        except Exception as exc:  # pragma: no cover - defensive
            return {"status": "error", "error": str(exc), "writes": 0,
                    "conflicts": [], "new_version": None}

    def distill_to_semantic(self) -> List[Dict]:
        if SemanticStore is None:
            return []
        try:
            store = SemanticStore(self.knowledge)
            result = store.distill()
            return result
        except Exception as e:
            return []

    def retrieve_semantic(self, query: str, top_k: int = 3) -> List[Dict]:
        try:
            q_emb = self.anchor.embed(query)
            if q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
            q_emb = q_emb / torch.linalg.vector_norm(q_emb).clamp(min=1e-6)
            if SemanticStore is not None:
                store = SemanticStore(self.knowledge)
                return store.retrieve_semantic(q_emb, top_k=top_k)
            else:
                return self.knowledge.recall_semantic(q_emb, top_k=top_k)
        except Exception as e:
            return []

    def get_contradictions(self) -> List[Dict]:
        if ContradictionResolver is None:
            return []
        try:
            facts = [f for f in self.knowledge.get_audit_trail() if not f.get("superseded")]
            resolver = ContradictionResolver()
            return resolver.detect_contradictions(facts)
        except Exception as e:
            return []

    def approve(self, fact_id: int) -> Dict[str, Any]:
        try:
            self.knowledge.conn.execute("UPDATE facts SET access_count = COALESCE(access_count,0)+1 WHERE id=?", (fact_id,))
            self.knowledge.conn.commit()
            try:
                self.knowledge.log_feedback(fact_id, "approve", strategy_used=self._last_retrieval_strategy)
            except Exception:  # noqa: BLE001
                pass
            if self.retrieval_policy:
                try:
                    last = self._retrieval_history[-1] if self._retrieval_history else {}
                    qt = last.get("query_type", "specific_fact")
                    self.retrieval_policy.update(qt, self._last_retrieval_strategy, reward=1.0)
                except Exception:  # noqa: BLE001
                    pass
            decision = self._consume_ttt_step()
            if decision.get("allowed"):
                try:
                    row = self.knowledge.conn.execute("SELECT text FROM facts WHERE id=?", (fact_id,)).fetchone()
                    if row:
                        text = row[0]
                        emb = self.anchor.embed(text)
                        if emb.dim() > 1:
                            emb = emb.squeeze(0)
                        if hasattr(self.memory, 'update_associative_memory'):
                            self.memory.update_associative_memory(emb, surprise_factor=1.0)
                except Exception as e:
                    pass
            return {
                "status": "approved",
                "fact_id": fact_id,
                "strategy": self._last_retrieval_strategy,
                "ttt": self.ttt_status(),
                "health": self.knowledge.retrieval_health(),
            }
        except Exception as e:
            return {"error": str(e), "fact_id": fact_id}

    def reject(self, fact_id: int) -> Dict[str, Any]:
        try:
            self.knowledge.conn.execute("UPDATE facts SET needs_review=1 WHERE id=?", (fact_id,))
            self.knowledge.conn.commit()
            try:
                self.knowledge.log_feedback(fact_id, "reject", strategy_used=self._last_retrieval_strategy)
            except Exception:  # noqa: BLE001
                pass
            if self.retrieval_policy:
                try:
                    last = self._retrieval_history[-1] if self._retrieval_history else {}
                    qt = last.get("query_type", "specific_fact")
                    self.retrieval_policy.update(qt, self._last_retrieval_strategy, reward=-1.0)
                except Exception:  # noqa: BLE001
                    pass
            return {
                "status": "rejected",
                "fact_id": fact_id,
                "strategy": self._last_retrieval_strategy,
                "ttt": self.ttt_status(),
            }
        except Exception as e:
            return {"error": str(e), "fact_id": fact_id}

    def correct(self, old_fact_id: int, new_text: str) -> Dict[str, Any]:
        try:
            emb = self.anchor.embed(new_text)
            if emb.dim() > 1:
                emb = emb.squeeze(0)
            emb = emb / torch.linalg.vector_norm(emb).clamp(min=1e-6)
            # Erase the superseded association before teaching the correction so
            # the plastic memory cannot keep answering with the old value.
            self._erase_corrected_fact(old_fact_id)
            new_id = self.knowledge.insert_fact(new_text, "correction", time.time(), self.step_count, 1.0, emb)
            self.knowledge.supersede(old_fact_id, new_id)
            try:
                self.knowledge.log_feedback(old_fact_id, "correct", correction_text=new_text, strategy_used=self._last_retrieval_strategy)
            except Exception:  # noqa: BLE001
                pass
            decision = self._consume_ttt_step()
            if decision.get("allowed"):
                try:
                    if hasattr(self.memory, 'update_associative_memory'):
                        self.memory.update_associative_memory(emb, surprise_factor=1.5)
                except Exception as e:
                    pass
            try:
                self.distill_to_semantic()
            except Exception:  # noqa: BLE001
                pass
            return {"status": "corrected", "old_id": old_fact_id, "new_id": new_id, "new_text": new_text}
        except Exception as e:
            return {"error": str(e), "old_id": old_fact_id}

    def get_feedback_log(self, fact_id: Optional[int] = None) -> List[Dict]:
        try:
            return self.knowledge.get_feedback_log(fact_id=fact_id)
        except Exception:  # noqa: BLE001
            return []

    def start_continuous_stream(self, source: Optional[str] = None,
                                max_fpm: int = 10) -> Dict[str, Any]:
        """Start this engine's single stream worker (idempotent under reruns)."""
        with self._engine_lock:
            if ContinuousStream is None:  # pragma: no cover - import fallback
                if self.continuous_stream is None:
                    self.continuous_stream = {
                        "running": True, "source": source, "max_fpm": max_fpm,
                        "stats": {"facts_processed": 0, "facts_absorbed": 0,
                                  "queue_depth": 0, "dropped": 0},
                    }
                return {"status": "started", "source": source}
            stream = self.continuous_stream
            if stream is not None and not isinstance(stream, ContinuousStream):
                # A cached dict stub from an older surface must not shadow a
                # real worker: replace it rather than double-own the stream.
                self.continuous_stream = None
                stream = None
            if stream is None:
                stream = ContinuousStream(
                    self, max_facts_per_minute=max_fpm, surprise_threshold=0.6
                )
                self.continuous_stream = stream
            started = stream.start(log_source=source)
            status = "started" if started else "already_running"
            return {"status": status, "source": source, "max_fpm": max_fpm,
                    "stream": stream.get_status()}

    def stop_continuous_stream(self, timeout: float = 2.0) -> Dict[str, Any]:
        """Bounded stop of this engine's worker; safe to call repeatedly."""
        with self._engine_lock:
            stream = self.continuous_stream
            if not isinstance(stream, ContinuousStream):
                self.continuous_stream = None
                return {"status": "not_running", "joined": True}
            if not stream.running and not stream.is_alive():
                return {"status": "not_running", "joined": True,
                        "stream": stream.get_status()}
            joined = stream.stop(timeout=timeout)
            status = stream.get_status()
            return {"status": "stopped" if joined else "stop_timeout",
                    "joined": joined, "stream": status}

    def pause_stream(self) -> Dict[str, Any]:
        with self._engine_lock:
            stream = self.continuous_stream
            if isinstance(stream, ContinuousStream):
                alive = stream.pause()
                return {"status": "paused", "worker_alive": alive}
            if isinstance(stream, dict):
                stream["running"] = False
                return {"status": "paused"}
            return {"status": "no_stream"}

    def resume_stream(self) -> Dict[str, Any]:
        with self._engine_lock:
            stream = self.continuous_stream
            if isinstance(stream, ContinuousStream):
                alive = stream.resume()
                return {"status": "resumed", "worker_alive": alive}
            if isinstance(stream, dict):
                stream["running"] = True
                return {"status": "resumed"}
            return {"status": "no_stream"}

    def enqueue_stream_text(self, text: str) -> Dict[str, Any]:
        """Queue text for the engine-owned worker (console 'Queue' button)."""
        stream = self.continuous_stream
        if not isinstance(stream, ContinuousStream) or not stream.running:
            return {"status": "not_running"}
        queued = stream.ingest(text)
        return {"status": "queued" if queued else "dropped",
                "queue_depth": stream.get_status()["queue_depth"]}

    def get_stream_status(self) -> Dict[str, Any]:
        stream = self.continuous_stream
        if isinstance(stream, ContinuousStream):
            return stream.get_status()
        if isinstance(stream, dict):
            return {"running": stream.get("running", False),
                    "alive": False,
                    "facts_processed": stream.get("stats", {}).get("facts_processed", 0),
                    "facts_absorbed": stream.get("stats", {}).get("facts_absorbed", 0),
                    "queue_depth": stream.get("stats", {}).get("queue_depth", 0)}
        return {"running": False, "alive": False, "facts_processed": 0,
                "facts_absorbed": 0, "queue_depth": 0}

    def get_stream_stats(self) -> Dict[str, Any]:
        return self.get_stream_status()

    def shutdown(self, timeout: float = 2.0) -> Dict[str, Any]:
        """Stop every owned worker with a bounded join; safe to repeat."""
        report: Dict[str, Any] = {"stream": {"status": "not_running", "joined": True}}
        try:
            report["stream"] = self.stop_continuous_stream(timeout=timeout)
        except Exception as exc:  # never let shutdown mask the original error
            report["stream"] = {"status": "failed", "error": str(exc)}
        worker = getattr(self, "auto_sleep_worker", None)
        if worker is not None:
            try:
                joined = worker.stop(timeout=timeout)
                report["auto_sleep"] = {"status": "stopped" if joined else "stop_timeout",
                                        "joined": joined}
            except Exception as exc:
                report["auto_sleep"] = {"status": "failed", "error": str(exc)}
            self.auto_sleep_worker = None
        report["stream_alive"] = bool(
            isinstance(self.continuous_stream, ContinuousStream)
            and self.continuous_stream.is_alive()
        )
        return report

    def continuous_learn(self, text: str) -> Dict[str, Any]:
        return self.teach(text, source="continuous_stream", surprise_threshold=0.6)

    def generate_with_memory(self, query: str, strategy: str = "auto", facts: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Generate an answer with memory injected by a *capability-checked* path.

        The reported strategy is what executed. An explicit unsupported mode
        raises ``UnsupportedCapabilityError``; automatic selection falls back to
        prompt rendering and records the reason. Hidden injection is applied
        only for the duration of this call and KV prefill is always released,
        including when generation raises.
        """
        if facts is None:
            facts = self.recall(query, top_k=10, threshold=0.3)
        if not facts and isinstance(self.gguf_backend, ByteRtuBackend):
            # Never free-generate on empty retrieval (FR-004): return the same
            # no-information message the extractive path uses.
            return NO_INFORMATION_MESSAGE

        backend = self.gguf_backend
        caps = capabilities_of(backend)
        query_type = "specific_fact"
        if self.retrieval_policy:
            try:
                query_type = self.retrieval_policy.classify_query(query)
            except Exception:
                query_type = "specific_fact"
        explicit = strategy not in (None, "auto")
        selection = resolve_strategy(
            len(facts), query_type, self.gguf_config, caps,
            requested=strategy if explicit else None,
            strict=explicit,
        )
        executed = selection.executed
        prompt = query
        injection_info: Dict[str, Any] = selection.to_dict()
        pending = None
        try:
            if executed == "hidden":
                trace = None
                if hasattr(self.memory, "h_trace"):
                    trace = self.memory.h_trace
                elif hasattr(self.memory, "states"):
                    trace = self.memory.states
                if trace is None:
                    raise UnsupportedCapabilityError("hidden", caps.name, "no trace available")
                declared = getattr(backend, "injection_layer", None)
                layer_idx = None
                if declared is not None:
                    layer_idx = int(declared() if callable(declared) else declared)
                pending = HiddenInjection(backend, trace, layer_idx=layer_idx)
                # Plan a prompt without facts: the trace carries the evidence.
                plan = plan_prompt(query, [], backend=backend)
                prompt = plan.prompt
                injection_info.update(plan.to_dict())
            elif executed == "kv_cache":
                pending = KvPrefill(
                    backend, [str(f.get("text", "")) for f in facts if f.get("text")]
                )
                plan = plan_prompt(query, facts, backend=backend)
                prompt = plan.prompt
                injection_info.update(plan.to_dict())
            else:
                plan = plan_prompt(query, facts, backend=backend)
                prompt = plan.prompt
                injection_info.update(plan.to_dict())
                injection_info["strategy"] = "prompt"
                injection_info["executed"] = "prompt"
                injection_info["applied"] = plan.fact_count > 0
                injection_info["fact_count"] = plan.fact_count
            if injection_info.get("applied") is False and executed == "prompt":
                # Nothing fitted: report it instead of pretending injection ran.
                injection_info["reason"] = injection_info.get("reason") or "no_facts_fit"
        except (UnsupportedCapabilityError, QueryBudgetError, ValueError) as exc:
            raise

        generated_text = ""
        token_logprobs = None
        try:
            if backend is not None:
                if executed == "hidden":
                    with pending:  # hooks live exactly for this generation
                        generated_text = backend.generate(
                            prompt, max_tokens=512, temperature=0.7, top_p=0.9
                        )
                elif executed == "kv_cache":
                    with pending:  # prefill, generate, then reset in __exit__
                        generated_text = backend.generate(
                            prompt, max_tokens=512, temperature=0.7, top_p=0.9
                        )
                else:
                    generated_text = backend.generate(
                        prompt, max_tokens=512, temperature=0.7, top_p=0.9
                    )
            else:
                if facts:
                    generated_text = " ".join([f["text"] for f in facts[:3]])
                else:
                    generated_text = self.chat(query)
        except Exception as e:
            generated_text = f"[Fallback] Based on memory: {' '.join([f['text'] for f in facts[:3]])}"
            injection_info["generation_error"] = str(e)
        finally:
            if pending is not None:
                if hasattr(pending, "remove"):
                    pending.remove()
                elif hasattr(pending, "cleanup"):
                    pending.cleanup()
                # Report the *executed* injection after it ran, never before.
                injection_info.update(
                    {k: v for k, v in pending.to_dict().items()
                     if k in {"applied", "executed", "layer_idx", "hidden_size",
                              "primed_tokens", "tokens", "binding", "cleanup",
                              "trace_norm", "projected_norm", "router_calls"}}
                )
        retrieval_conf = 0.0
        if facts:
            retrieval_conf = max([f.get("similarity", 0) for f in facts]) if facts else 0.0
        generation_conf = 0.7
        try:
            generation_conf = generation_confidence(token_logprobs=token_logprobs, self_consistency=0.8, entropy=0.3)
        except Exception:
            pass
        fused_conf = fuse_confidence(retrieval_conf, generation_conf)
        tier = confidence_tier(fused_conf)
        return {"text": generated_text, "query": query, "strategy": executed,
                "requested_strategy": selection.requested,
                "strategy_reason": selection.reason,
                "facts_used": facts[:5], "retrieval_confidence": retrieval_conf,
                "generation_confidence": generation_conf, "confidence": fused_conf,
                "confidence_tier": tier, "injection_info": injection_info,
                "capabilities": caps.to_dict(), "prompt": prompt}

    # ------------------------------------------------------------------
    # External knowledge acquisition (adaptation/web-grounded-learning)
    # ------------------------------------------------------------------
    @property
    def research_enabled(self) -> bool:
        return bool(self.research_config.get("enabled"))

    def configure_research(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply a research block: enabled flag, provider, and bounds."""
        from recamem.search import SearchLearner, normalize_research_config

        self.research_config = normalize_research_config(config)
        self.search_learner = None
        if self.research_config.get("enabled"):
            self.search_learner = SearchLearner(self, self.research_config)
        status = {
            "enabled": self.research_enabled,
            "provider": self.research_config.get("provider"),
            "available": bool(self.search_learner and self.search_learner.available),
        }
        if self.search_learner is not None and not self.search_learner.available:
            status["reason"] = self.search_learner.unavailable_reason()
        return status

    def research(self, topic: str) -> Dict[str, Any]:
        """Acquire knowledge about ``topic`` from the configured provider.

        Returns the documented contract: the topic, whether a search happened,
        the queries issued, the sources consulted, the facts learned, the
        answer, and the answer path. With no provider configured it returns the
        unavailable result and raises nothing.
        """
        with self._engine_lock:
            if self.search_learner is None and self.research_enabled:
                self.configure_research(self.research_config)
            learner = self.search_learner
            if learner is None or not learner.available:
                reason = (
                    learner.unavailable_reason()
                    if learner is not None
                    else "external acquisition is disabled for this profile"
                )
                result = {
                    "topic": topic,
                    "query": topic,
                    "searched": False,
                    "queries": [],
                    "sources": [],
                    "facts_learned": 0,
                    "facts": [],
                    "results": [],
                    "answer": None,
                    "answer_path": self._last_answer_path,
                    "available": False,
                    "bounded": False,
                    "reason": reason,
                }
                self.last_research_result = result
                return result

            result = learner.research(topic, force=True)
            self.last_research_result = result
            if result.get("facts_learned"):
                self._set_answer_path(ANSWER_PATH_RESEARCHED)
                self.last_research_sources = list(result.get("sources") or [])
                result["answer_path"] = ANSWER_PATH_RESEARCHED
            else:
                result["answer_path"] = self._last_answer_path
            self.log(
                f"Research on '{topic}': searched={result['searched']} "
                f"facts={result['facts_learned']} sources={len(result['sources'])}",
                details={"research": {key: result[key] for key in
                                      ("searched", "facts_learned", "sources", "reason")}},
            )
            return result

    def _research_chat_answer(self, query: str) -> Optional[str]:
        """Research an unanswerable question when the profile allows it.

        Returns ``None`` — and makes no network call — when acquisition is
        disabled, unavailable, bounded, or yielded nothing usable.
        """
        if not self.research_enabled:
            return None
        result = self.research(query)
        if result.get("facts_learned"):
            return result.get("answer")
        return None

    def estimate_confidence(self, retrieval_conf: float, generation_conf: float) -> float:
        return fuse_confidence(retrieval_conf, generation_conf)

    def decide_path(self, query: str) -> Dict[str, Any]:
        if self.decision_head is None:
            q_lower = query.lower()
            if any(w in q_lower for w in ["what is", "who is", "when", "where", "define"]):
                return {"action": "fast_recall", "path": "fast", "confidence": 0.7, "reason": "definitional query heuristic"}
            else:
                return {"action": "slow_synthesis", "path": "slow", "confidence": 0.6, "reason": "synthesis heuristic"}
        try:
            q_emb = self.anchor.embed(query)
            if q_emb.dim() > 1:
                q_emb = q_emb.squeeze(0)
            decision = self.decision_head.decide(q_emb)
            action = decision.get("action", "chat")
            if action in ("recall", "fast_recall", "chat"):
                path = "fast"
            elif action in ("slow_synthesis", "tool", "teach"):
                path = "slow"
            else:
                sims = decision.get("similarities", [])
                names = decision.get("prototype_names", [])
                fast_idx = None
                slow_idx = None
                for i, n in enumerate(names):
                    if "fast" in n:
                        fast_idx = i
                    if "slow" in n:
                        slow_idx = i
                if fast_idx is not None and slow_idx is not None:
                    if sims[fast_idx] > sims[slow_idx]:
                        path = "fast"
                    else:
                        path = "slow"
                else:
                    path = "fast" if decision.get("confidence",0) > 0.6 else "slow"
            decision["path"] = path
            return decision
        except Exception as e:
            return {"action": "fast_recall", "path": "fast", "confidence": 0.5, "error": str(e)}

    def _manifest_checkpoint_name(self) -> Optional[str]:
        """Relative checkpoint filename for this profile, or None."""
        if not isinstance(self.gguf_backend, ByteRtuBackend):
            return None
        declared = getattr(self, "byte_rtu_checkpoint", None) or "rtu.safetensors"
        # The manifest owns a bare relative filename inside the profile dir.
        name = os.path.basename(str(declared))
        return name or "rtu.safetensors"

    @staticmethod
    def _write_manifest(profile_dir: Path, config: Dict[str, Any]) -> None:
        """Write config.json atomically via config.json.tmp + os.replace."""
        tmp = profile_dir / "config.json.tmp"
        with open(tmp, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(str(tmp), str(profile_dir / "config.json"))

    def save_profile(self, target_dir: str):
        """Persist a profile: artifacts first, manifest last.

        Order (change 1.4): knowledge store -> plastic state -> byte-RTU
        checkpoint -> logs/snapshots -> config.json. The manifest is the last
        write and is atomic, so an interrupted save never claims an artifact
        that is not on disk.
        """
        p = Path(target_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.bind_profile_dir(p)
        checkpoint_name = self._manifest_checkpoint_name()
        self.byte_rtu_checkpoint = checkpoint_name

        # 1. Knowledge store.
        try:
            self.knowledge.checkpoint()
            self.knowledge.export_to(str(p / "knowledge.db"))
        except Exception:
            pass

        # 2. Plastic state. Tensors go to safetensors; version/identity and
        # other non-tensor metadata go to a JSON sidecar so a restart restores
        # capacity, eviction counters, gates and schedule state (post-merge 2.7).
        adapter_meta: Dict[str, Any] = {}
        try:
            if hasattr(self.memory, 'state_dict'):
                weights = {}
                sd = self.memory.state_dict()
                for k, v in sd.items():
                    if isinstance(v, torch.Tensor):
                        weights[k] = v.detach().cpu().contiguous()
                    elif isinstance(v, (str, int, float, bool, list, tuple, type(None))):
                        adapter_meta[k] = v
                # ``h_trace`` is already in the adapter state dict; do not
                # also write it as ``states`` (safetensors rejects two keys
                # that share memory). Legacy states-only readers still work.
                if hasattr(self.memory, 'h_trace'):
                    weights.setdefault(
                        "h_trace", self.memory.h_trace.detach().cpu().contiguous()
                    )
                elif hasattr(self.memory, 'states'):
                    weights.setdefault(
                        "states", self.memory.states.detach().cpu().contiguous()
                    )
                if hasattr(self.memory, 'decay'):
                    weights["decay"] = self.memory.decay.detach().cpu().contiguous()
                if hasattr(self.memory, 'proj'):
                    weights["proj_weight"] = self.memory.proj.weight.detach().cpu().contiguous()
                save_file(weights, str(p / "memory.safetensors"))
            else:
                weights = {
                    "states": self.memory.states.detach().cpu().contiguous(),
                    "decay": self.memory.decay.detach().cpu().contiguous(),
                    "proj_weight": self.memory.proj.weight.detach().cpu().contiguous(),
                }
                save_file(weights, str(p / "memory.safetensors"))
        except Exception:
            try:
                if hasattr(self.memory, 'h_trace'):
                    save_file({"states": self.memory.h_trace.detach().cpu()}, str(p / "memory.safetensors"))
            except Exception:
                pass
        # Sidecar is written after the tensors it describes, and only when
        # there is metadata to record.
        if adapter_meta:
            adapter_meta.setdefault("adapter_name", self.adapter_name)
            adapter_meta.setdefault("adapter_version", getattr(self.memory, "_adapter_version", 1))
            try:
                tmp = p / "adapter_meta.json.tmp"
                with open(tmp, "w") as f:
                    json.dump(adapter_meta, f, indent=2)
                os.replace(tmp, p / "adapter_meta.json")
            except Exception:
                pass

        # 3. Byte-RTU checkpoint (byte_rtu / hybrid profiles only).
        if checkpoint_name is not None:
            checkpoint_path = p / checkpoint_name
            self.gguf_backend.checkpoint = str(checkpoint_path)
            self.gguf_backend.model.save(str(checkpoint_path))

        # 4. Logs, snapshots, retrieval policy (not named by the manifest).
        try:
            with open(p / "train_log.json", "w") as f:
                json.dump(self.logs[-200:], f, indent=2)
        except Exception:
            pass
        try:
            if self.snapshots:
                with open(p / "snapshots.json", "w") as f:
                    serializable = {}
                    for vid, snap in self.snapshots.items():
                        meta = snap.get("metadata", {})
                        serializable[vid] = meta
                    json.dump(serializable, f, indent=2)
        except Exception:
            pass

        # 4b. MAS importance/masks (reversible pruning evidence).
        mas_manifest: Optional[Dict[str, Any]] = None
        if self.mas is not None and getattr(self.mas, "importance", None):
            try:
                self.mas.save_state(
                    str(p / "mas.safetensors"), str(p / "mas_meta.json")
                )
                mas_manifest = {"file": "mas.safetensors", "metadata": "mas_meta.json"}
            except Exception:
                mas_manifest = None

        # 4c. Snapshot directories travel with the profile so a rollback still
        # works after a restart (post-merge 4.4/4.5).
        try:
            if self.snapshot_manager is not None:
                self.snapshot_manager.relocate(str(p / "snapshots"))
        except Exception:
            pass

        # 5. Manifest last.
        config: Dict[str, Any] = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "name": self.name,
            "dim": self.dim,
            "step_count": self.step_count,
            "base_lr": self.base_lr,
            "total_facts": self.knowledge.active_count(),
            "updated_at": time.time(),
            "surprise_threshold": self.surprise_threshold,
            "K_max": self.K_max,
            "test_time_training": self.test_time_training,
            "current_version": self.current_version,
        }
        # A configured research block is part of the profile: the provider name,
        # the bounds and the enabled flag round-trip. An unconfigured profile
        # writes no block, which loads as disabled.
        if self.research_config and (
            self.research_config.get("enabled")
            or self.research_config.get("provider")
        ):
            config["research"] = dict(self.research_config)
        # Explicit registered identity, never a class-name substring guess.
        identity = self.adapter_name or identify_adapter(self.memory)
        config["plastic_adapter"] = identity or "associative_rtu"
        config["adapter_identity_source"] = "registry" if identity else "default"
        if getattr(self.memory, "_adapter_version", None) is not None:
            config["adapter_version"] = int(self.memory._adapter_version)
        if self.migrations:
            config["migrations"] = self.migrations
        if getattr(self, "merge_history", None):
            config["merge_history"] = list(self.merge_history)
        if getattr(self, "merge_of", None):
            config["merge_of"] = list(self.merge_of)
        # Write constructor kwargs for round-trip (task 2.12)
        try:
            if hasattr(self, 'adapter_kwargs') and self.adapter_kwargs:
                config["adapter_kwargs"] = dict(self.adapter_kwargs)
            elif "adapter_kwargs" in locals():
                # fallback if attribute not set
                pass
        except Exception:  # noqa: BLE001
            pass
        if self.gguf_config:
            config.update(self.gguf_config)
        if self.gguf_backend and hasattr(self.gguf_backend, 'model_path'):
            config["gguf_model_path"] = self.gguf_backend.model_path
        config["sleep"] = dict(self.sleep_config)
        config["decreasing_backprop"] = self.lr_schedule.state_dict()
        if mas_manifest is not None:
            config["mas_state"] = mas_manifest
        slow_state = self.slow_meta_state()
        if slow_state is not None:
            (p / "slow_tier.json").write_text(json.dumps(slow_state, indent=2) + "\n")
            config["has_slow_tier"] = True
        if self.retrieval_policy:
            try:
                policy_path = p / "retrieval_policy.json"
                self.retrieval_policy.persist(str(policy_path))
                config["has_retrieval_policy"] = True
            except Exception:
                pass

        # Manifest fields owned by this change (1.1-1.2).
        config["engine"] = self.engine
        config["fusion_mode"] = self.fusion_mode
        config["rtu_dim"] = int(self.rtu_dim)
        config["rtu_layers"] = int(self.rtu_layers)
        config["rtu_spread"] = int(self.rtu_spread)
        if checkpoint_name is not None:
            config["byte_rtu_checkpoint"] = checkpoint_name
            # Legacy alias, kept for one release (read-only on load).
            config["rtu_checkpoint"] = checkpoint_name
            if isinstance(self.gguf_backend, ByteRtuBackend):
                config["injection_strategy"] = "prompt"
        else:
            config["byte_rtu_checkpoint"] = None
            config["rtu_checkpoint"] = None
        self._write_manifest(p, config)
        self.log(f"Profile saved to {target_dir}")

    def _capture_live_profile_state(self) -> Dict[str, Any]:
        """Snapshot everything a failed load must be able to put back."""
        state: Dict[str, Any] = {"memory": self.memory, "knowledge": self.knowledge}
        try:
            state["plastic"] = clone_state(self.memory.state_dict())
        except Exception:
            state["plastic"] = None
        policy = self.retrieval_policy
        state["policy"] = (
            clone_state(policy.state_dict()) if policy is not None
            and hasattr(policy, "state_dict") else None
        )
        try:
            state["schedule"] = clone_state(self.lr_schedule.state_dict())
        except Exception:
            state["schedule"] = None
        for name in ("engine", "fusion_mode", "rtu_dim", "rtu_layers", "rtu_spread",
                     "name", "step_count", "base_lr", "surprise_threshold", "K_max",
                     "test_time_training", "current_version", "gguf_config",
                     "adapter_name", "adapter_kwargs", "migrations",
                     "byte_rtu_checkpoint", "sleep_config", "logs", "gguf_backend",
                     "mas", "byte_rtu_backend"):
            state[name] = getattr(self, name, None)
        state["merge_history"] = clone_state(getattr(self, "merge_history", []))
        return state

    def _restore_live_profile_state(self, state: Dict[str, Any]) -> None:
        self.memory = state.get("memory", self.memory)
        if state.get("plastic") is not None and hasattr(self.memory, "load_state_dict"):
            try:
                self.memory.load_state_dict(clone_state(state["plastic"]))
            except Exception:
                pass
        if state.get("knowledge") is not None:
            if self.knowledge is not state["knowledge"]:
                try:
                    self.knowledge.close()
                except Exception:
                    pass
            self.knowledge = state["knowledge"]
        policy = self.retrieval_policy
        if state.get("policy") is not None and policy is not None and hasattr(
            policy, "load_state_dict"
        ):
            try:
                policy.load_state_dict(state["policy"])
            except Exception:
                pass
        if state.get("schedule") is not None:
            try:
                self.lr_schedule = DecreasingBackprop().load_state_dict(state["schedule"])
            except Exception:
                pass
        for name in ("engine", "fusion_mode", "rtu_dim", "rtu_layers", "rtu_spread",
                     "name", "step_count", "base_lr", "surprise_threshold", "K_max",
                     "test_time_training", "current_version", "gguf_config",
                     "adapter_name", "adapter_kwargs", "migrations",
                     "byte_rtu_checkpoint", "sleep_config", "logs", "gguf_backend",
                     "mas", "byte_rtu_backend"):
            if name in state:
                setattr(self, name, state[name])
        if "merge_history" in state:
            self.merge_history = clone_state(state["merge_history"])

    @staticmethod
    def _validate_profile_artifacts(p: Path, config: Dict[str, Any],
                                    config_file: Path) -> None:
        """Validate every referenced artifact before anything is activated."""
        db_file = p / "knowledge.db"
        if db_file.exists():
            try:
                conn = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)
                try:
                    integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
                    tables = {
                        row[0] for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        )
                    }
                finally:
                    conn.close()
            except Exception as exc:
                raise InvalidProfileError(
                    f"Profile database {db_file} is unreadable: {exc}"
                ) from exc
            if integrity != "ok":
                raise InvalidProfileError(
                    f"Profile database {db_file} failed integrity_check: {integrity}"
                )
            if "facts" not in tables:
                raise InvalidProfileError(
                    f"Profile database {db_file} has no facts table"
                )
        for name in ("retrieval_policy.json", "snapshots.json", "train_log.json",
                     "adapter_meta.json"):
            path = p / name
            if not path.exists():
                continue
            try:
                with open(path, "r") as handle:
                    json.load(handle)
            except Exception as exc:
                raise InvalidProfileError(
                    f"Profile artifact {path} is malformed: {exc}"
                ) from exc
        schedule_cfg = config.get("decreasing_backprop")
        if isinstance(schedule_cfg, dict):
            try:
                DecreasingBackprop().load_state_dict(schedule_cfg)
            except Exception as exc:
                raise InvalidProfileError(
                    f"Profile manifest {config_file} declares an invalid "
                    f"decreasing-backprop schedule: {exc}"
                ) from exc
        mas_manifest = config.get("mas_state")
        if isinstance(mas_manifest, dict) and mas_manifest.get("file"):
            mas_path = p / str(mas_manifest["file"])
            if not mas_path.is_file():
                raise InvalidProfileError(
                    f"Profile manifest {config_file} declares MAS state {mas_path} "
                    "which is missing"
                )
            meta_path = p / str(mas_manifest.get("metadata", "mas_meta.json"))
            if meta_path.exists():
                try:
                    with open(meta_path, "r") as handle:
                        json.load(handle)
                except Exception as exc:
                    raise InvalidProfileError(
                        f"MAS metadata {meta_path} is malformed: {exc}"
                    ) from exc

    def load_profile(self, target_dir: str) -> bool:
        """Load a profile transactionally (post-merge 2.8).

        Every referenced artifact is validated before activation, and any
        failure restores the previously active profile instead of leaving a
        half-swapped engine behind.
        """
        live = self._capture_live_profile_state()
        try:
            return self._load_profile_inner(target_dir)
        except Exception:
            self._restore_live_profile_state(live)
            raise

    def _load_profile_inner(self, target_dir: str) -> bool:
        """Load a profile, failing loudly on a missing or invalid artifact.

        Change 1.5/1.16/1.23: a missing/corrupt manifest raises
        ``InvalidProfileError``; a manifest that names an absent checkpoint
        raises ``MissingCheckpointError``. There is no return-False path and
        no silent random initialisation.
        """
        p = Path(target_dir)
        self.bind_profile_dir(p)
        config_file = p / "config.json"
        mem_file = p / "memory.safetensors"
        if not config_file.is_file():
            raise InvalidProfileError(
                f"Profile manifest not found: {config_file}"
            )
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise InvalidProfileError(
                f"Profile manifest {config_file} is not valid JSON: {e}"
            ) from e
        except OSError as e:
            raise InvalidProfileError(
                f"Profile manifest {config_file} is unreadable: {e}"
            ) from e
        if not isinstance(config, dict):
            raise InvalidProfileError(
                f"Profile manifest {config_file} must be a JSON object, "
                f"got {type(config).__name__}"
            )
        declared_version = int(config.get("schema_version", 1) or 1)
        if declared_version > PROFILE_SCHEMA_VERSION:
            raise InvalidProfileError(
                f"Profile manifest {config_file} declares schema version "
                f"{declared_version}, newer than the supported "
                f"{PROFILE_SCHEMA_VERSION}; refusing to load"
            )
        if declared_version < PROFILE_SCHEMA_VERSION:
            # Explicit upgrade step: the older manifest is read as-is and the
            # upgrade is recorded so the next save writes the current version.
            self.migrations.append(
                {
                    "kind": "profile_schema",
                    "migration": "profile_schema_version",
                    "from": declared_version,
                    "to": PROFILE_SCHEMA_VERSION,
                    "source": str(config_file),
                }
            )
        if not mem_file.is_file():
            raise InvalidProfileError(
                f"Profile artifact missing: {mem_file} (plastic state)"
            )
        # Stage 1: validate every artifact before any live state is replaced.
        self._validate_profile_artifacts(p, config, config_file)

        declared_checkpoint = config.get("byte_rtu_checkpoint") or config.get("rtu_checkpoint")
        engine = config.get("engine")
        if engine is None:
            # Compatibility: infer from the artifact the manifest declares.
            engine = "byte_rtu" if declared_checkpoint else "droid"
        if engine not in ("droid", "byte_rtu", "hybrid"):
            raise InvalidProfileError(
                f"Profile manifest {config_file} declares unknown engine {engine!r}"
            )
        self.engine = engine
        self.fusion_mode = config.get(
            "fusion_mode", "prompt_stuffing" if engine != "droid" else "off"
        )
        self.rtu_dim = int(config.get("rtu_dim", 384))
        self.rtu_layers = int(config.get("rtu_layers", 8))
        self.rtu_spread = int(config.get("rtu_spread", 32))

        checkpoint_path = None
        if declared_checkpoint:
            declared_path = Path(str(declared_checkpoint))
            if declared_path.is_absolute() or ".." in declared_path.parts:
                raise InvalidProfileError(
                    f"Profile manifest {config_file} declares checkpoint "
                    f"{declared_checkpoint!r} outside the profile directory"
                )
            checkpoint_path = p / declared_path
            if not checkpoint_path.is_file():
                raise MissingCheckpointError(
                    f"Profile {p.name!r} declares checkpoint "
                    f"{declared_checkpoint!r} but {checkpoint_path} is missing."
                )
        self.byte_rtu_checkpoint = str(declared_checkpoint) if declared_checkpoint else None

        self.name = config.get("name", self.name)
        self.step_count = config.get("step_count", 0)
        self.base_lr = config.get("base_lr", self.base_lr)
        self.surprise_threshold = config.get("surprise_threshold", self.surprise_threshold)
        self.K_max = config.get("K_max", self.K_max)
        self.test_time_training = config.get("test_time_training", self.test_time_training)
        self.current_version = config.get("current_version", self.current_version)
        # External acquisition is read from the manifest and defaults to
        # disabled when the block is absent (web-grounded-learning 5.4).
        research_block = config.get("research")
        if research_block is None:
            self.configure_research({})
        elif not isinstance(research_block, dict):
            raise InvalidProfileError(
                f"Profile manifest {config_file} declares a non-object research block"
            )
        else:
            self.configure_research(research_block)
        # Keep the pinned injection strategy: it is a profile setting, not a
        # GGUF-prefixed one, and dropping it re-enables the hidden/kv stubs.
        self.gguf_config = {k: v for k, v in config.items() if k.startswith("gguf_")}
        if "injection_strategy" in config:
            self.gguf_config["injection_strategy"] = config["injection_strategy"]
        # Explicit adapter identity is authoritative (post-merge 2.5/2.6).
        explicit_identity = "plastic_adapter" in config
        adapter_name = config.get("plastic_adapter", "associative_rtu")
        adapter_kwargs = config.get("adapter_kwargs", {}) or {}
        if not isinstance(adapter_kwargs, dict):
            raise InvalidProfileError(
                f"Profile manifest {config_file} declares non-object adapter_kwargs"
            )
        current_identity = identify_adapter(self.memory) or getattr(
            self.memory, "_adapter_type", None
        )
        if explicit_identity:
            if not isinstance(adapter_name, str) or not adapter_name:
                raise InvalidProfileError(
                    f"Profile manifest {config_file} declares an empty plastic_adapter"
                )
            if get_adapter is None:
                raise InvalidProfileError(
                    "Cannot load a profile with an explicit plastic_adapter: "
                    "the adapter registry is unavailable"
                )
            try:
                from recamem.plastic_adapter import get_adapter_class
                get_adapter_class(adapter_name)
            except KeyError as exc:
                raise InvalidProfileError(
                    f"Profile manifest {config_file} names unsupported plastic_adapter "
                    f"{adapter_name!r}; refusing to substitute the default adapter"
                ) from exc
            if current_identity != adapter_name:
                constructor = {k: v for k, v in dict(adapter_kwargs).items() if k != "dim"}
                constructor.setdefault("dim", self.dim)
                try:
                    self.memory = get_adapter(adapter_name, **constructor)
                except Exception as exc:
                    raise InvalidProfileError(
                        f"Could not construct plastic adapter {adapter_name!r} from "
                        f"{config_file}: {exc}"
                    ) from exc
            self.adapter_name = adapter_name
        else:
            # Legacy manifest: documented default, recorded as a versioned
            # migration rather than silently treated as authored state.
            self.adapter_name = "associative_rtu"
            record = {"migration": "adapter_identity_default", "version": 1,
                      "from": "absent", "to": "associative_rtu"}
            if record not in self.migrations:
                self.migrations.append(record)
        self.adapter_kwargs = dict(adapter_kwargs)
        # Merge provenance round-trips with the profile it describes.
        if isinstance(config.get("merge_history"), list):
            self.merge_history = list(config["merge_history"])
        if isinstance(config.get("merge_of"), list):
            self.merge_of = list(config["merge_of"])
        elif getattr(self, "merge_history", None):
            self.merge_of = list(self.merge_history[-1].get("merge_of", []))

        weights: Dict[str, Any] = {}
        try:
            weights = dict(load_file(str(mem_file)))
        except Exception as exc:
            raise InvalidProfileError(
                f"Profile artifact {mem_file} is unreadable or malformed: {exc}"
            ) from exc
        # Merge the versioned metadata sidecar that safetensors cannot carry.
        meta_file = p / "adapter_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                if not isinstance(meta, dict):
                    raise ValueError("adapter metadata must be a JSON object")
            except Exception as exc:
                raise InvalidProfileError(
                    f"Adapter metadata {meta_file} is malformed: {exc}"
                ) from exc
            for key, value in meta.items():
                weights.setdefault(key, value)

        previous_state = None
        try:
            previous_state = clone_state(self.memory.state_dict())
        except Exception:
            previous_state = None
        try:
            with torch.no_grad():
                if hasattr(self.memory, 'load_state_dict'):
                    try:
                        self.memory.load_state_dict(weights)
                    except Exception:
                        adapter_type = getattr(self.memory, "_adapter_type", "")
                        if adapter_type in ("delta_rtu", "sparse_delta", "fly_model"):
                            # Versioned delta/sparse states must never be
                            # reinterpreted through the legacy key path.
                            raise
                        if "states" in weights and hasattr(self.memory, 'h_trace'):
                            self.memory.h_trace.copy_(weights["states"])
                        elif "states" in weights and hasattr(self.memory, 'states'):
                            self.memory.states.copy_(weights["states"])
                        if "decay" in weights and hasattr(self.memory, 'decay'):
                            self.memory.decay.copy_(weights["decay"])
                        if "proj_weight" in weights and hasattr(self.memory, 'proj'):
                            self.memory.proj.weight.copy_(weights["proj_weight"])
                else:
                    if "states" in weights:
                        self.memory.states.copy_(weights["states"])
                    if "decay" in weights:
                        self.memory.decay.copy_(weights["decay"])
                    if "proj_weight" in weights:
                        self.memory.proj.weight.copy_(weights["proj_weight"])
        except Exception as exc:
            # Staged failure: restore the previously active plastic state.
            if previous_state is not None:
                try:
                    self.memory.load_state_dict(previous_state)
                except Exception:
                    pass
            raise InvalidProfileError(
                f"Plastic state {mem_file} is incompatible with adapter "
                f"{self.adapter_name!r}: {exc}"
            ) from exc
        db_file = p / "knowledge.db"
        know_file = p / "knowledge.json"
        # Build the replacement store first; the live store is closed only once
        # the new one is fully populated (transactional load, post-merge 2.8).
        if db_file.exists():
            new_knowledge = KnowledgeStore(str(db_file), dim=self.dim)
        else:
            new_knowledge = KnowledgeStore(":memory:", dim=self.dim)
            if know_file.exists():
                with open(know_file, "r") as f:
                    legacy_facts = json.load(f)
                legacy_vecs = weights.get("knowledge_vectors") if weights else None
                for i, fact in enumerate(legacy_facts):
                    vec = None
                    if legacy_vecs is not None and i < len(legacy_vecs):
                        vec = legacy_vecs[i].float()
                    if vec is None:
                        vec = self.anchor.embed(fact["text"])
                        if vec.dim() > 1:
                            vec = vec.squeeze(0)
                    new_knowledge.insert_fact(
                        fact["text"],
                        fact.get("source", "legacy"),
                        fact.get("timestamp", time.time()),
                        fact.get("step", i),
                        fact.get("novelty", 0.0),
                        vec,
                    )
        previous_knowledge = self.knowledge
        self.knowledge = new_knowledge
        try:
            if previous_knowledge is not None and previous_knowledge is not new_knowledge:
                previous_knowledge.close()
        except Exception:
            pass
        log_file = p / "train_log.json"
        if log_file.exists():
            with open(log_file, "r") as f:
                self.logs = json.load(f)
        if self.snapshot_manager is not None:
            try:
                self.snapshot_manager.base_dir = p / "snapshots"
            except Exception:
                pass
        snap_file = p / "snapshots.json"
        if snap_file.exists():
            try:
                with open(snap_file, "r") as f:
                    snap_meta = json.load(f)
                for vid, meta in snap_meta.items():
                    if vid not in self.snapshots:
                        self.snapshots[vid] = {"metadata": meta, "plastic_state": {}}
            except Exception:
                pass
        policy_file = p / "retrieval_policy.json"
        if policy_file.exists() and self.retrieval_policy:
            try:
                self.retrieval_policy.load(str(policy_file))
            except Exception:
                pass

        # Sleep policy, learning-rate schedule and MAS state.
        sleep_cfg = config.get("sleep")
        if isinstance(sleep_cfg, dict):
            merged = json.loads(json.dumps(DEFAULT_SLEEP_CONFIG))
            merged.update(sleep_cfg)
            self.sleep_config = merged
        schedule_cfg = config.get("decreasing_backprop")
        if isinstance(schedule_cfg, dict):
            try:
                self.lr_schedule = DecreasingBackprop().load_state_dict(schedule_cfg)
            except Exception as exc:
                raise InvalidProfileError(
                    f"Profile {config_file} declares an invalid decreasing-backprop "
                    f"schedule: {exc}"
                ) from exc
            self.sleep_config["schedule"] = self.lr_schedule.state_dict()
        mas_manifest = config.get("mas_state")
        if isinstance(mas_manifest, dict) and mas_manifest.get("file"):
            mas_path = p / str(mas_manifest["file"])
            meta_path = p / str(mas_manifest.get("metadata", "mas_meta.json"))
            if not mas_path.is_file():
                raise InvalidProfileError(
                    f"Profile {config_file} declares MAS state {mas_path} which is missing"
                )
            try:
                mas = MemoryAwareSynapses(
                    self.memory,
                    ratio=float((json.loads(meta_path.read_text())
                                 if meta_path.exists() else {}).get("ratio", 0.1)),
                    quality_fn=self._retrieval_quality,
                )
                mas.load_state(str(mas_path),
                               str(meta_path) if meta_path.exists() else None)
                self.mas = mas
            except Exception as exc:
                raise InvalidProfileError(
                    f"MAS state {mas_path} is incompatible with adapter "
                    f"{self.adapter_name!r}: {exc}"
                ) from exc

        # Re-attach the generative backend *after* the artifacts are restored.
        if checkpoint_path is not None:
            gguf_path = self.gguf_config.get("gguf_model_path") or self.gguf_config.get("model_path")
            use_gguf = (
                self.engine == "droid"
                or (self.engine == "hybrid" and self.fusion_mode == "off")
            ) and bool(gguf_path) and GgufBackend is not None
            if use_gguf:
                try:
                    self.gguf_backend = GgufBackend(model_path=gguf_path)
                except Exception:
                    self.gguf_backend = None
            else:
                self.gguf_backend = ByteRtuBackend(
                    checkpoint=str(checkpoint_path),
                    dim=self.rtu_dim,
                    layers=self.rtu_layers,
                    spread=self.rtu_spread,
                )
                self.gguf_config["injection_strategy"] = "prompt"
        elif self.gguf_config and GgufBackend is not None:
            try:
                model_path = self.gguf_config.get("gguf_model_path") or self.gguf_config.get("model_path")
                repo_id = self.gguf_config.get("gguf_repo_id") or self.gguf_config.get("repo_id")
                filename = self.gguf_config.get("gguf_filename") or self.gguf_config.get("filename")
                if model_path or repo_id:
                    self.gguf_backend = GgufBackend(model_path=model_path, repo_id=repo_id, filename=filename)
            except Exception:
                pass
        # SLOW tier meta-parameters round-trip with the profile.
        slow_path = p / "slow_tier.json"
        if slow_path.is_file():
            try:
                payload = json.loads(slow_path.read_text())
                tier = self.enable_slow_tier(float(payload.get("tolerance", 0.05)))
                from recamem.plasticity import SlowTier
                weights = {k: torch.tensor(v) for k, v in payload.get("weights", {}).items()}
                if weights:
                    tier.load_state_dict(weights)
                tier.train_log = list(payload.get("train_log", []))
            except (OSError, ValueError, RuntimeError) as exc:
                raise InvalidProfileError(
                    f"Profile {p.name!r} has an unreadable slow_tier.json: {exc}"
                ) from exc
        self.log(f"Profile loaded from {target_dir}")
        return True
