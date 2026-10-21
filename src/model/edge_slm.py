"""Provenance-aware on-device SLM catalog and resource-constrained advice.

Implements ``model/edge-optimization``.  The registry is metadata only: listing
or recommending a model never downloads or loads weights.  Source-reported
figures are labelled as such and are never presented as local measurements;
unknown device-specific throughput stays unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

REGISTRY_VERSION = "1.0.0"
CATALOG_VERIFIED = "2026-10-21"

_TASK_TAGS = ("general_chat", "reasoning", "summarization", "coding", "extraction")


@dataclass(frozen=True)
class ModelEntry:
    key: str
    family: str
    display_name: str
    params_b: float
    ram_gb: float
    ram_source: str  # "source-reported" | "estimate"
    quantizations: Tuple[str, ...]
    default_quantization: str
    task_tags: Tuple[str, ...]
    gguf_repo_id: str
    gguf_filename: str
    tok_s: Optional[Tuple[float, float]] = None
    tok_s_source: Optional[str] = None
    quality: Dict[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()
    provenance_url: str = ""
    verified: str = CATALOG_VERIFIED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "display_name": self.display_name,
            "params_b": self.params_b,
            "ram_gb": self.ram_gb,
            "ram_source": self.ram_source,
            "quantizations": list(self.quantizations),
            "default_quantization": self.default_quantization,
            "task_tags": list(self.task_tags),
            "tok_s": list(self.tok_s) if self.tok_s else None,
            "tok_s_source": self.tok_s_source or "unknown",
            "quality": dict(self.quality),
            "notes": list(self.notes),
            "model_id": self.gguf_repo_id,
            "filename": self.gguf_filename,
            "provenance": {
                "type": self.ram_source,
                "url": self.provenance_url,
                "verified": self.verified,
                "local_measurement": False,
            },
        }

    def gguf_config(self, n_ctx: int = 4096) -> Dict[str, Any]:
        return {
            "gguf_repo_id": self.gguf_repo_id,
            "gguf_filename": self.gguf_filename,
            "gguf_quantization": self.default_quantization,
            "gguf_n_ctx": int(n_ctx),
        }


def _entry(key: str, family: str, name: str, params_b: float, ram_gb: float,
           ram_source: str, tags: Sequence[str], repo: str, filename: str,
           quality: Optional[Dict[str, Any]] = None,
           tok_s: Optional[Tuple[float, float]] = None,
           tok_s_source: Optional[str] = None,
           notes: Sequence[str] = (), url: str = "",
           quant: str = "Q4_K_M") -> ModelEntry:
    return ModelEntry(
        key=key, family=family, display_name=name, params_b=params_b,
        ram_gb=ram_gb, ram_source=ram_source,
        quantizations=(quant,), default_quantization=quant,
        task_tags=tuple(tags), gguf_repo_id=repo, gguf_filename=filename,
        tok_s=tok_s, tok_s_source=tok_s_source, quality=quality or {},
        notes=tuple(notes), provenance_url=url,
    )


_GGUF_URL = "https://ggufloader.github.io/2025-07-07-top-10-gguf-models-i5-16gb.html"
_SURVEY_URL = "https://v-chandra.github.io/on-device-llms/"


class EdgeSLMRegistry:
    """Versioned catalog of 13 small models with provenance and `recommend()`."""

    VERSION = REGISTRY_VERSION

    def __init__(self, entries: Optional[Sequence[ModelEntry]] = None):
        self._entries: Dict[str, ModelEntry] = {}
        for entry in entries if entries is not None else self._default_catalog():
            self._entries[entry.key] = entry

    # -- catalog --------------------------------------------------------
    @staticmethod
    def _default_catalog() -> List[ModelEntry]:
        return [
            _entry("llama-3.2-1b", "llama-3.2", "Llama 3.2 1B Instruct", 1.0,
                   1.5, "source-reported",
                   ["general_chat", "summarization", "extraction"],
                   "bartowski/Llama-3.2-1B-Instruct-GGUF",
                   "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                   tok_s=(30.0, 50.0), tok_s_source="source-reported",
                   quality={"source_label": "source-reported"},
                   notes=("source-reported ~1.5 GB RAM and 30-50 tokens/s on the cited i5/16 GB system",),
                   url=_GGUF_URL),
            _entry("llama-3.2-3b", "llama-3.2", "Llama 3.2 3B Instruct", 3.0,
                   2.6, "estimate", ["general_chat", "reasoning", "summarization"],
                   "bartowski/Llama-3.2-3B-Instruct-GGUF",
                   "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                   tok_s=None, notes=("throughput on this device is unknown",)),
            _entry("qwen2.5-0.5b", "qwen2.5", "Qwen2.5 0.5B Instruct", 0.5,
                   0.5, "estimate", ["general_chat", "extraction"],
                   "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                   "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
            _entry("qwen2.5-1.5b", "qwen2.5", "Qwen2.5 1.5B Instruct", 1.5,
                   1.2, "estimate", ["reasoning", "general_chat", "coding"],
                   "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                   "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                   notes=("tagged reasoning-oriented by the cited on-device survey",),
                   url=_SURVEY_URL),
            _entry("qwen2.5-3b", "qwen2.5", "Qwen2.5 3B Instruct", 3.0,
                   2.4, "estimate", ["reasoning", "coding", "general_chat"],
                   "Qwen/Qwen2.5-3B-Instruct-GGUF",
                   "qwen2.5-3b-instruct-q4_k_m.gguf"),
            _entry("smollm2-135m", "smollm2", "SmolLM2 135M Instruct", 0.135,
                   0.2, "estimate", ["general_chat", "extraction"],
                   "HuggingFaceTB/SmolLM2-135M-Instruct-GGUF",
                   "smollm2-135m-instruct-q4_k_m.gguf"),
            _entry("smollm2-360m", "smollm2", "SmolLM2 360M Instruct", 0.36,
                   0.35, "estimate", ["general_chat", "extraction"],
                   "HuggingFaceTB/SmolLM2-360M-Instruct-GGUF",
                   "smollm2-360m-instruct-q4_k_m.gguf"),
            _entry("smollm2-1.7b", "smollm2", "SmolLM2 1.7B Instruct", 1.7,
                   1.5, "estimate", ["general_chat", "summarization", "reasoning"],
                   "HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
                   "smollm2-1.7b-instruct-q4_k_m.gguf",
                   notes=("reported 11T-token training corpus",),
                   url=_SURVEY_URL),
            _entry("smollm2-3b", "smollm2", "SmolLM2 3B (preview)", 3.0,
                   2.5, "estimate", ["general_chat", "summarization"],
                   "HuggingFaceTB/SmolLM2-3B-Instruct-GGUF",
                   "smollm2-3b-instruct-q4_k_m.gguf"),
            _entry("gemma-3-270m", "gemma-3", "Gemma 3 270M Instruct", 0.27,
                   0.3, "estimate", ["general_chat", "extraction"],
                   "bartowski/gemma-3-270m-it-GGUF",
                   "gemma-3-270m-it-Q4_K_M.gguf"),
            _entry("gemma-3-1b", "gemma-3", "Gemma 3 1B Instruct", 1.0,
                   0.8, "source-reported",
                   ["general_chat", "reasoning", "summarization"],
                   "bartowski/gemma-3-1b-it-GGUF",
                   "gemma-3-1b-it-Q4_K_M.gguf",
                   quality={"gsm8k_pct": 62.8, "source_label": "source-reported"},
                   notes=("source-reported ~0.8 GB and 62.8% GSM8K",),
                   url=_SURVEY_URL),
            _entry("gemma-3-4b", "gemma-3", "Gemma 3 4B Instruct", 4.0,
                   3.3, "estimate", ["reasoning", "general_chat", "summarization"],
                   "bartowski/gemma-3-4b-it-GGUF",
                   "gemma-3-4b-it-Q4_K_M.gguf"),
            _entry("phi-4-mini", "phi-4", "Phi-4-mini Instruct", 3.8,
                   3.0, "estimate", ["reasoning", "coding", "general_chat"],
                   "bartowski/microsoft_Phi-4-mini-instruct-GGUF",
                   "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"),
        ]

    def all(self) -> List[ModelEntry]:
        return sorted(self._entries.values(), key=lambda entry: entry.key)

    def families(self) -> List[str]:
        return sorted({entry.family for entry in self._entries.values()})

    def get(self, key: str) -> ModelEntry:
        try:
            return self._entries[key]
        except KeyError as exc:
            raise KeyError(
                f"Unknown model {key!r}; available: {sorted(self._entries)}"
            ) from exc

    def __len__(self) -> int:
        return len(self._entries)

    def metadata(self) -> Dict[str, Any]:
        return {
            "version": self.VERSION,
            "verified": CATALOG_VERIFIED,
            "count": len(self._entries),
            "families": self.families(),
            "provenance_note": (
                "RAM/throughput figures are source-reported or estimates, not "
                "local measurements; unknown throughput is reported as unknown."
            ),
        }

    # -- recommendation -------------------------------------------------
    def recommend(self, ram_gb: Optional[float] = None,
                  min_tok_s: Optional[float] = None,
                  task: Optional[str] = None) -> Dict[str, Any]:
        if ram_gb is not None and ram_gb < 0:
            raise ValueError("ram_gb must be non-negative")
        if min_tok_s is not None and min_tok_s < 0:
            raise ValueError("min_tok_s must be non-negative")
        if task is not None and task not in _TASK_TAGS:
            raise ValueError(f"task must be one of {list(_TASK_TAGS)}, got {task!r}")

        excluded: List[Dict[str, str]] = []
        candidates: List[Dict[str, Any]] = []
        for entry in self.all():
            if ram_gb is not None and entry.ram_gb > ram_gb:
                excluded.append({"key": entry.key, "reason": "ram_exceeds_budget"})
                continue
            if min_tok_s is not None:
                if entry.tok_s is None:
                    excluded.append({"key": entry.key,
                                     "reason": "throughput_unknown_for_hard_minimum"})
                    continue
                if entry.tok_s[1] < min_tok_s:
                    excluded.append({"key": entry.key,
                                     "reason": "throughput_below_minimum"})
                    continue
            score, rationale = self._score(entry, task, ram_gb, min_tok_s)
            candidates.append({
                "key": entry.key,
                "model_id": entry.gguf_repo_id,
                "filename": entry.gguf_filename,
                "quantization": entry.default_quantization,
                "params_b": entry.params_b,
                "ram_gb": entry.ram_gb,
                "ram_source": entry.ram_source,
                "tok_s": list(entry.tok_s) if entry.tok_s else None,
                "tok_s_source": entry.tok_s_source or "unknown",
                "task_tags": list(entry.task_tags),
                "score": round(score, 4),
                "rationale": rationale,
                "provenance": {
                    "type": entry.ram_source,
                    "url": entry.provenance_url,
                    "verified": entry.verified,
                    "local_measurement": False,
                },
                "gguf_config": entry.gguf_config(),
            })

        candidates.sort(key=lambda item: (-item["score"], item["ram_gb"], item["key"]))
        status = "ok" if candidates else "no_match"
        return {
            "status": status,
            "registry_version": self.VERSION,
            "constraints": {"ram_gb": ram_gb, "min_tok_s": min_tok_s, "task": task},
            "recommendations": candidates,
            "excluded": excluded,
            "provenance_note": (
                "Ranked from catalog metadata; source-reported values are not "
                "local benchmarks."
            ),
        }

    @staticmethod
    def _score(entry: ModelEntry, task: Optional[str],
               ram_gb: Optional[float],
               min_tok_s: Optional[float]) -> Tuple[float, List[str]]:
        score = 0.0
        rationale: List[str] = []
        if task and task in entry.task_tags:
            score += 2.0
            rationale.append(f"tagged for {task}")
        elif task:
            rationale.append(f"not tagged for {task}")
        if task == "reasoning":
            gsm8k = entry.quality.get("gsm8k_pct")
            if isinstance(gsm8k, (int, float)):
                score += float(gsm8k) / 100.0
                rationale.append(
                    f"source-reported GSM8K {gsm8k}% (provenance: source-reported)"
                )
        if task == "general_chat" and 0.3 <= entry.params_b <= 3.0:
            score += 0.5
            rationale.append("mid-size model for general chat")
        if ram_gb is not None:
            headroom = max(0.0, ram_gb - entry.ram_gb)
            score += min(0.5, headroom / max(1.0, ram_gb))
            rationale.append(f"fits {ram_gb:.1f} GB with {headroom:.1f} GB headroom")
        if min_tok_s is not None and entry.tok_s is not None:
            score += 0.1 if entry.tok_s[1] >= min_tok_s else 0.0
            rationale.append(
                f"source-reported throughput {entry.tok_s[0]:.0f}-{entry.tok_s[1]:.0f} tok/s"
            )
        if entry.tok_s is None:
            rationale.append("throughput unknown (not claimed)")
        score += 0.01 * min(entry.params_b, 5.0) * 0.1
        return score, rationale

    # -- backend hand-off ----------------------------------------------
    def gguf_config(self, key_or_entry, n_ctx: int = 4096) -> Dict[str, Any]:
        entry = (self.get(key_or_entry) if isinstance(key_or_entry, str)
                 else key_or_entry)
        config = entry.gguf_config(n_ctx)
        # No download and no model load happens here on purpose.
        config["load_requested"] = False
        return config


__all__ = ["EdgeSLMRegistry", "ModelEntry", "REGISTRY_VERSION"]
