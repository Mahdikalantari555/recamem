"""Track-5 isolated lifelong memory and adaptation arms (`M0`..`M5`).

Invariants enforced across arms:
1. `M0`..`M4` NEVER mutate any `nn.Parameter` of the attached `CandidateLM` during `ingest()`.
2. `M5` (`online_weight_plasticity`) operates strictly on an isolated deep-copied `plastic_model`
   snapshot and never mutates the caller's original `base_model` in place.
3. Every arm reports exact static parameter bytes + dynamic store/state bytes via `footprint_bytes()`.
"""

from __future__ import annotations

import copy
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CandidateLM, MemoryFootprint, TokenizerAdapter
from src.model.cores.objectives import compute_shifted_ce
from src.model.delta_memory import DeltaPlasticAdapter, SparseDeltaAdapter
from src.model.hybrid_retrieval import ExactDenseIndex
from src.model.knowledge_store import KnowledgeStore


def _state_nbytes(state: Any) -> int:
    if state is None:
        return 0
    if isinstance(state, torch.Tensor):
        return int(state.numel() * state.element_size())
    if isinstance(state, Mapping):
        return sum(_state_nbytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(_state_nbytes(v) for v in state)
    return 0


def _extract_subject_key(text: str, metadata: Optional[Mapping[str, Any]] = None) -> str:
    if metadata and metadata.get("entity_key"):
        return str(metadata["entity_key"]).strip().lower()
    if metadata and metadata.get("subject"):
        return str(metadata["subject"]).strip().lower()
    # Heuristic subject extraction before 'is', 'equals', ':', or '='
    m = re.match(r"^\s*([A-Za-z0-9_\-\s]{2,48}?)\s+(?:is|are|equals|was|has|=|:)\b", text)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip().lower())
    words = re.findall(r"[a-z0-9]+", text.lower())
    return " ".join(words[:3]) if words else text.strip().lower()[:32]


@torch.no_grad()
def _encode_text_mean_and_nll(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    text: str,
) -> Tuple[torch.Tensor, float, torch.Tensor]:
    was_training = model.training
    model.eval()
    try:
        ids = tokenizer.encode(text, add_eos=True)
        if not ids:
            ids = [tokenizer.eos_id]
        device = model.embed.weight.device
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)
        # Temporarily bypass memory hook when computing clean backbone embedding/surprise
        saved_hook = model.memory_layer_hook
        model.memory_layer_hook = None
        try:
            details = model(t_ids, return_details=True)
        finally:
            model.memory_layer_hook = saved_hook
        h = details["hidden_states"][0]  # (T, d)
        vec = F.normalize(h.float().mean(dim=0), dim=-1, eps=1e-6)
        nll = float(compute_shifted_ce(details["logits"], t_ids, pad_id=tokenizer.pad_id).item())
        return vec, nll, t_ids
    finally:
        if was_training:
            model.train()


class PersistentCoreStateArm(nn.Module):
    """`M0` (`persistent_core_state`): streams facts into recurrent/KV core state with zero index."""

    arm_id = "M0"

    def __init__(self, model: CandidateLM, tokenizer: TokenizerAdapter) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self._carried_state: Any = None
        self._streamed_chunks: int = 0
        self._recent_stream_log: Dict[str, str] = {}

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            ids = self.tokenizer.encode(text + "\n", add_eos=False)
            if not ids:
                ids = [self.tokenizer.sep_id]
            t_ids = torch.tensor([ids], dtype=torch.long, device=self.base_model.embed.weight.device)
            details = self.base_model(t_ids, state=self._carried_state, return_details=True)
            self._carried_state = details["state"]
            self._streamed_chunks += 1
            subj = _extract_subject_key(text, metadata)
            self._recent_stream_log[subj] = text
            return {
                "arm_id": self.arm_id,
                "ingested": True,
                "streamed_chunks": self._streamed_chunks,
                "state_bytes": _state_nbytes(self._carried_state),
            }
        finally:
            if was_training:
                self.base_model.train()

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        del top_k
        was_training = self.base_model.training
        self.base_model.eval()
        try:
            ids = self.tokenizer.encode(query_text, add_eos=False) or [self.tokenizer.sep_id]
            t_ids = torch.tensor([ids], dtype=torch.long, device=self.base_model.embed.weight.device)
            details = self.base_model(t_ids, state=self._carried_state, return_details=True)
            # Greedy 16-token continuation from carried state to measure pure state-only recall
            st = details["state"]
            last_logits = details["logits"][0, -1, :]
            gen_ids: List[int] = []
            for _ in range(16):
                nxt = int(torch.argmax(last_logits).item())
                if nxt == self.tokenizer.eos_id or nxt == self.tokenizer.pad_id:
                    break
                gen_ids.append(nxt)
                step_out = self.base_model.step(torch.tensor([nxt], device=t_ids.device), state=st)
                st = step_out["state"]
                last_logits = step_out["logits"][0]
            continuation = self.tokenizer.decode(gen_ids)
            return {
                "arm_id": self.arm_id,
                "hits": [{"text": continuation, "score": 1.0}] if continuation else [],
                "state_continuation": continuation,
                "logits_last": details["logits"][0, -1, :].detach().cpu(),
            }
        finally:
            if was_training:
                self.base_model.train()

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": key_or_subject})
        self._recent_stream_log.pop(key_or_subject.strip().lower(), None)
        return {"arm_id": self.arm_id, "status": "stream_overwrite_only"}

    def reset(self) -> None:
        self._carried_state = None
        self._streamed_chunks = 0
        self._recent_stream_log.clear()
        self.base_model.reset_state()

    def footprint_bytes(self) -> MemoryFootprint:
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=0,
            dynamic_store_bytes=_state_nbytes(self._carried_state),
            active_items=self._streamed_chunks,
        )

    teach = ingest
    recall = query


class EngramNGramHashArm(nn.Module):
    """`M1` (`engram_ngram_hash`): conditional n-gram (`n in {2,3}`) hash table gated at layers 2 & L/2."""

    arm_id = "M1"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        hash_buckets: int = 2048,
        max_param_cap: int = 2_500_000,
        attach_hook: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.hash_buckets = int(hash_buckets)
        n_layers = int(getattr(model.core, "n_layers", 4))
        self.target_layers = sorted({min(1, n_layers - 1), n_layers // 2})

        # Static conditional n-gram table + context gate parameters (checked against max_param_cap)
        table_params = self.hash_buckets * self.d_model + len(self.target_layers) * (self.d_model + 1)
        if table_params > int(max_param_cap):
            raise ValueError(
                f"EngramNGramHashArm table_params={table_params} exceeds max_param_cap={max_param_cap}"
            )
        self.static_engram = nn.Embedding(self.hash_buckets, self.d_model)
        nn.init.normal_(self.static_engram.weight, mean=0.0, std=0.02)
        self.layer_gates = nn.ModuleDict(
            {str(l_idx): nn.Linear(self.d_model, 1, bias=True) for l_idx in self.target_layers}
        )
        for gate in self.layer_gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.constant_(gate.bias, -2.0)

        # Dynamic non-parameter slot memory updated online during `ingest()`
        self.register_buffer("dynamic_slots", torch.zeros(self.hash_buckets, self.d_model, dtype=torch.float32))
        self._slot_texts: Dict[int, Dict[str, Any]] = {}
        self._subject_buckets: Dict[str, int] = {}

        if attach_hook:
            self.base_model.memory_layer_hook = self._layer_hook

    def _ngram_buckets(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, T) -> causal 2-gram + 3-gram rolling hash mod hash_buckets
        b, t = token_ids.shape
        t0 = token_ids
        t1 = F.pad(token_ids[:, :-1], (1, 0), value=0)
        t2 = F.pad(token_ids[:, :-2], (2, 0), value=0)
        hashed = (t0 * 1315423911 + t1 * 2654435761 + t2 * 97531) % self.hash_buckets
        return hashed.long()

    def _text_bucket(self, key_text: str) -> int:
        ids = self.tokenizer.encode(key_text.lower(), add_eos=False)
        acc = 2166136261
        for tok in ids[:16]:
            acc = ((acc ^ int(tok)) * 16777619) & 0xFFFFFFFF
        return int(acc % self.hash_buckets)

    def _layer_hook(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if layer_idx not in self.target_layers or token_ids is None:
            return hidden
        buckets = self._ngram_buckets(token_ids)
        mem_vec = self.static_engram(buckets) + self.dynamic_slots[buckets]
        gate = torch.sigmoid(self.layer_gates[str(layer_idx)](hidden))
        return hidden + gate * mem_vec

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        bucket = self._text_bucket(subj)
        self.dynamic_slots[bucket].copy_(vec)
        self._slot_texts[bucket] = {"text": text, "subject": subj, "vec": vec.detach().cpu()}
        self._subject_buckets[subj] = bucket
        return {"arm_id": self.arm_id, "ingested": True, "bucket": bucket, "subject": subj}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._slot_texts:
            return {"arm_id": self.arm_id, "hits": []}
        subj = _extract_subject_key(query_text)
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        q_cpu = q_vec.detach().cpu()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        exact_bucket = self._subject_buckets.get(subj)
        for b_idx, item in self._slot_texts.items():
            sim = float(torch.dot(q_cpu, item["vec"]).item())
            if exact_bucket is not None and b_idx == exact_bucket:
                sim += 0.50  # exact n-gram subject hash hit
            scored.append((sim, {"text": item["text"], "score": sim, "bucket": b_idx}))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return {"arm_id": self.arm_id, "hits": [h for _, h in scored[:top_k]]}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = key_or_subject.strip().lower()
        old_bucket = self._subject_buckets.pop(subj, self._text_bucket(subj))
        self.dynamic_slots[old_bucket].zero_()
        self._slot_texts.pop(old_bucket, None)
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": subj})
        return {"arm_id": self.arm_id, "erased_bucket": old_bucket}

    def reset(self) -> None:
        self.dynamic_slots.zero_()
        self._slot_texts.clear()
        self._subject_buckets.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        dyn_b = len(self._slot_texts) * self.d_model * 4 + sum(
            len(v["text"].encode("utf-8")) for v in self._slot_texts.values()
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._slot_texts),
        )

    teach = ingest
    recall = query


class SurpriseResidualEpisodicArm(nn.Module):
    """`M2` (`surprise_residual_episodic`): Candidate C NLL-surprise write gate + top-k residual read."""

    arm_id = "M2"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        surprise_threshold: float = 0.50,
        max_entries: int = 2048,
        top_k: int = 4,
        attach_hook: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.surprise_threshold = float(surprise_threshold)
        self.max_entries = int(max_entries)
        self.default_top_k = int(top_k)
        n_layers = int(getattr(model.core, "n_layers", 4))
        self.inject_layer = n_layers // 2

        self.read_gate = nn.Linear(self.d_model, 1, bias=True)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -1.5)

        self._entries: List[Dict[str, Any]] = []
        if attach_hook:
            self.base_model.memory_layer_hook = self._layer_hook

    def _layer_hook(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        token_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        del token_ids
        if layer_idx != self.inject_layer or not self._entries:
            return hidden
        keys = torch.stack([e["key"] for e in self._entries], dim=0).to(hidden.device, dtype=hidden.dtype)
        vals = torch.stack([e["val"] for e in self._entries], dim=0).to(hidden.device, dtype=hidden.dtype)
        q = F.normalize(hidden, dim=-1, eps=1e-6)  # (B, T, d)
        sims = torch.matmul(q, keys.transpose(0, 1))  # (B, T, M)
        k = min(self.default_top_k, keys.shape[0])
        top_vals, top_idx = torch.topk(sims, k=k, dim=-1)
        weights = F.softmax(top_vals * 8.0, dim=-1)  # (B, T, k)
        gathered = vals[top_idx]  # (B, T, k, d)
        retrieved = torch.sum(gathered * weights.unsqueeze(-1), dim=2)
        gate = torch.sigmoid(self.read_gate(hidden))
        return hidden + gate * retrieved

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        vec, surprise_nll, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        force = bool(metadata.get("force", False)) if metadata else False
        if not force and surprise_nll < self.surprise_threshold:
            return {
                "arm_id": self.arm_id,
                "ingested": False,
                "surprise_nll": surprise_nll,
                "reason": "below_surprise_threshold",
            }
        subj = _extract_subject_key(text, metadata)
        # Supersede prior conflicting entry for same entity subject
        self._entries = [e for e in self._entries if e["subject"] != subj]
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        self._entries.append(
            {
                "subject": subj,
                "text": text,
                "key": vec.detach().cpu(),
                "val": vec.detach().cpu(),
                "surprise": surprise_nll,
            }
        )
        return {"arm_id": self.arm_id, "ingested": True, "surprise_nll": surprise_nll, "subject": subj}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._entries:
            return {"arm_id": self.arm_id, "hits": []}
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        q_cpu = q_vec.detach().cpu()
        q_words = set(re.findall(r"[a-z0-9]+", query_text.lower()))
        scored = []
        for e in self._entries:
            cos = float(torch.dot(q_cpu, e["key"]).item())
            e_words = set(re.findall(r"[a-z0-9]+", e["text"].lower()))
            lex_overlap = len(q_words & e_words) / float(max(1, len(q_words)))
            score = 0.65 * cos + 0.35 * lex_overlap
            scored.append((score, {"text": e["text"], "score": score, "surprise": e["surprise"]}))
        scored.sort(key=lambda p: p[0], reverse=True)
        return {"arm_id": self.arm_id, "hits": [h for _, h in scored[:top_k]]}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = key_or_subject.strip().lower()
        before = len(self._entries)
        self._entries = [e for e in self._entries if e["subject"] != subj]
        removed = before - len(self._entries)
        if new_text:
            res = self.ingest(new_text, metadata={"entity_key": subj, "force": True})
            res["removed"] = removed
            return res
        return {"arm_id": self.arm_id, "removed": removed}

    def reset(self) -> None:
        self._entries.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        dyn_b = sum(
            int(e["key"].numel() * 4 + e["val"].numel() * 4 + len(e["text"].encode("utf-8")))
            for e in self._entries
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._entries),
        )

    teach = ingest
    recall = query


class HybridKnowledgeRRFArm(nn.Module):
    """`M3` (`hybrid_knowledge_rrf`): wraps `KnowledgeStore` (FTS5 BM25 + dense MVM + RRF)."""

    arm_id = "M3"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        injection_mode: str = "prompt_stuffing",  # "prompt_stuffing" | "gated_residual"
        db_path: str = ":memory:",
    ) -> None:
        super().__init__()
        if injection_mode not in {"prompt_stuffing", "gated_residual"}:
            raise ValueError(f"Unsupported injection_mode {injection_mode!r}")
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.injection_mode = injection_mode
        self.db_path = db_path
        self._init_store()
        self._step = 0

        if self.injection_mode == "gated_residual":
            self.residual_gate = nn.Linear(self.d_model, 1, bias=True)
            nn.init.zeros_(self.residual_gate.weight)
            nn.init.constant_(self.residual_gate.bias, -1.5)
        else:
            self.residual_gate = None

    def _init_store(self) -> None:
        self.store = KnowledgeStore(db_path=self.db_path, dim=self.d_model)
        self.store.configure_retrieval(dense_index=ExactDenseIndex(dim=self.d_model))
        self._subject_to_fact_id: Dict[str, int] = {}

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        vec, novelty, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        self._step += 1
        old_id = self._subject_to_fact_id.get(subj)
        fid = self.store.insert_fact(
            text=text,
            source=str((metadata or {}).get("domain", "tournament")),
            timestamp=time.time(),
            step=self._step,
            novelty=novelty,
            vec=vec.detach().cpu(),
            origin="user_text",
        )
        if old_id is not None and old_id != fid:
            self.store.supersede(old_id, fid)
        self._subject_to_fact_id[subj] = fid
        return {"arm_id": self.arm_id, "ingested": True, "fact_id": fid, "superseded_old_id": old_id}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, query_text)
        rows = self.store.recall(
            query_vec=q_vec.detach().cpu(),
            top_k=top_k,
            query_text=query_text,
            strategy="hybrid_rrf",
        )
        hits = [{"text": r["text"], "score": float(r.get("fused_score", 0.0)), "id": r["id"]} for r in rows]
        stuffed_prompt = (
            "\n".join(f"[Memory] {h['text']}" for h in hits) + f"\n[Query] {query_text}"
            if (hits and self.injection_mode == "prompt_stuffing")
            else query_text
        )
        return {
            "arm_id": self.arm_id,
            "injection_mode": self.injection_mode,
            "hits": hits,
            "augmented_prompt": stuffed_prompt,
        }

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = key_or_subject.strip().lower()
        old_id = self._subject_to_fact_id.get(subj)
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": subj})
        if old_id is not None:
            self.store.supersede(old_id, None)
            self._subject_to_fact_id.pop(subj, None)
        return {"arm_id": self.arm_id, "superseded_old_id": old_id}

    def reset(self) -> None:
        self._init_store()
        self._step = 0

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.parameters())
        active_rows = self.store.conn.execute(
            "SELECT text FROM facts WHERE superseded = 0"
        ).fetchall()
        n_active = len(active_rows)
        text_bytes = sum(len(str(r[0]).encode("utf-8")) for r in active_rows)
        vec_bytes = n_active * self.d_model * 4
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(text_bytes + vec_bytes),
            active_items=n_active,
        )

    teach = ingest
    recall = query


class SparseDeltaAdapterArm(nn.Module):
    """`M4` (`sparse_delta_adapter`): wraps `DeltaPlasticAdapter` / `SparseDeltaAdapter` fast weights."""

    arm_id = "M4"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        use_sparse_expansion: bool = True,
        code_dim: int = 512,
        top_k_code: int = 64,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.d_model = int(model.d_model)
        self.use_sparse_expansion = bool(use_sparse_expansion)

        if self.use_sparse_expansion:
            self.adapter: nn.Module = SparseDeltaAdapter(
                dim=self.d_model,
                code_dim=code_dim,
                top_k=min(top_k_code, code_dim),
                heads=heads,
            )
        else:
            self.adapter = DeltaPlasticAdapter(dim=self.d_model, heads=heads)
        # Freeze adapter projection parameters so online writes strictly update fast-weight buffer `S`!
        for p in self.adapter.parameters():
            p.requires_grad_(False)

        self._value_book: Dict[str, Dict[str, Any]] = {}

    def _encode_sparse_or_dense(self, vec: torch.Tensor) -> torch.Tensor:
        if isinstance(self.adapter, SparseDeltaAdapter):
            return self.adapter.coder.encode(vec)
        return vec

    def _delta_core(self) -> DeltaPlasticAdapter:
        if isinstance(self.adapter, SparseDeltaAdapter):
            return self.adapter.delta
        return self.adapter  # type: ignore[return-value]

    @torch.no_grad()
    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        subj = _extract_subject_key(text, metadata)
        key_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, subj)
        val_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, text)
        k_code = self._encode_sparse_or_dense(key_vec.detach().cpu())
        v_code = self._encode_sparse_or_dense(val_vec.detach().cpu())
        delta = self._delta_core()
        # Use beta=0.95 so contradicting writes cleanly overwrite the key's association in `S`
        write_info = delta.write_association(k_code, v_code, beta=torch.tensor(0.95))
        readback = F.normalize(delta.read(k_code), dim=-1, eps=1e-6)
        self._value_book[subj] = {
            "text": text,
            "val_readback": readback.detach().clone(),
            "key_vec": key_vec.detach().cpu(),
        }
        return {"arm_id": self.arm_id, "ingested": True, "subject": subj, **write_info}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        if not self._value_book:
            return {"arm_id": self.arm_id, "hits": []}
        subj = _extract_subject_key(query_text)
        q_vec, _, _ = _encode_text_mean_and_nll(self.base_model, self.tokenizer, subj)
        q_code = self._encode_sparse_or_dense(q_vec.detach().cpu())
        delta = self._delta_core()
        recalled_val = F.normalize(delta.read(q_code), dim=-1, eps=1e-6)

        scored = []
        for s_key, entry in self._value_book.items():
            assoc_sim = float(torch.dot(recalled_val, entry["val_readback"]).item())
            key_sim = float(torch.dot(q_vec.detach().cpu(), entry["key_vec"]).item())
            score = 0.70 * assoc_sim + 0.30 * key_sim + (0.25 if s_key == subj else 0.0)
            scored.append((score, {"text": entry["text"], "score": score, "subject": s_key}))
        scored.sort(key=lambda p: p[0], reverse=True)
        return {"arm_id": self.arm_id, "hits": [h for _, h in scored[:top_k]]}

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        subj = key_or_subject.strip().lower()
        if new_text:
            return self.ingest(new_text, metadata={"entity_key": subj})
        entry = self._value_book.pop(subj, None)
        if entry is not None:
            k_code = self._encode_sparse_or_dense(entry["key_vec"])
            self._delta_core().erase(k_code, strength=1.0)
        return {"arm_id": self.arm_id, "erased": entry is not None}

    def reset(self) -> None:
        self._delta_core().reset()
        self._value_book.clear()

    def footprint_bytes(self) -> MemoryFootprint:
        static_b = sum(p.numel() * p.element_size() for p in self.adapter.parameters())
        delta = self._delta_core()
        dyn_b = int(delta.S.numel() * delta.S.element_size()) + sum(
            len(v["text"].encode("utf-8")) for v in self._value_book.values()
        )
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(static_b),
            dynamic_store_bytes=int(dyn_b),
            active_items=len(self._value_book),
        )

    teach = ingest
    recall = query


class OnlineWeightPlasticityArm(nn.Module):
    """`M5` (`online_weight_plasticity`): snapshot-guarded online SGD + diagonal Fisher EWC.

    Safety contract: `base_model` is NEVER mutated in place; all online gradient steps
    apply exclusively to `self.plastic_model`, which can be reset to `_clean_snapshot` in O(1).
    """

    arm_id = "M5"

    def __init__(
        self,
        model: CandidateLM,
        tokenizer: TokenizerAdapter,
        lr: float = 5e-3,
        steps_per_ingest: int = 3,
        ewc_lambda: float = 10.0,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.tokenizer = tokenizer
        self.lr = float(lr)
        self.steps_per_ingest = int(steps_per_ingest)
        self.ewc_lambda = float(ewc_lambda)

        # Isolated plastic copy + immutable clean snapshot
        self.plastic_model: CandidateLM = copy.deepcopy(model)
        self._clean_snapshot: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in self.plastic_model.state_dict().items()
        }
        self._anchor_params: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in self.plastic_model.named_parameters() if p.requires_grad
        }
        self._fisher_diag: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, dtype=torch.float32)
            for n, p in self.plastic_model.named_parameters()
            if p.requires_grad
        }
        self._ingested_count = 0

    def ingest(self, text: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        del metadata
        ids = self.tokenizer.encode(text, add_eos=True)
        if len(ids) < 2:
            ids = ids + [self.tokenizer.eos_id]
        device = self.plastic_model.embed.weight.device
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)

        self.plastic_model.train()
        opt = torch.optim.SGD(self.plastic_model.parameters(), lr=self.lr)
        final_loss = 0.0
        for _ in range(self.steps_per_ingest):
            opt.zero_grad(set_to_none=True)
            logits = self.plastic_model(t_ids)
            ce = compute_shifted_ce(logits, t_ids, pad_id=self.tokenizer.pad_id)
            ewc = torch.zeros((), device=device)
            if self._ingested_count > 0 and self.ewc_lambda > 0.0:
                for n, p in self.plastic_model.named_parameters():
                    if p.requires_grad:
                        ewc = ewc + (self._fisher_diag[n] * (p - self._anchor_params[n]).pow(2)).sum()
            loss = ce + 0.5 * self.ewc_lambda * ewc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.plastic_model.parameters(), 1.0)
            # Update running empirical diagonal Fisher & anchor lightly
            with torch.no_grad():
                for n, p in self.plastic_model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self._fisher_diag[n].mul_(0.9).add_(p.grad.pow(2), alpha=0.1)
            opt.step()
            final_loss = float(ce.detach().item())

        with torch.no_grad():
            for n, p in self.plastic_model.named_parameters():
                if p.requires_grad:
                    self._anchor_params[n].mul_(0.8).add_(p.data, alpha=0.2)

        self.plastic_model.eval()
        self._ingested_count += 1
        return {"arm_id": self.arm_id, "ingested": True, "final_ce_loss": final_loss}

    @torch.no_grad()
    def query(self, query_text: str, top_k: int = 4) -> Dict[str, Any]:
        del top_k
        self.plastic_model.eval()
        ids = self.tokenizer.encode(query_text, add_eos=False) or [self.tokenizer.sep_id]
        device = self.plastic_model.embed.weight.device
        t_ids = torch.tensor([ids], dtype=torch.long, device=device)
        details = self.plastic_model(t_ids, return_details=True)
        st = details["state"]
        last_logits = details["logits"][0, -1, :]
        gen_ids: List[int] = []
        for _ in range(16):
            nxt = int(torch.argmax(last_logits).item())
            if nxt in {self.tokenizer.eos_id, self.tokenizer.pad_id}:
                break
            gen_ids.append(nxt)
            step_out = self.plastic_model.step(torch.tensor([nxt], device=device), state=st)
            st = step_out["state"]
            last_logits = step_out["logits"][0]
        continuation = self.tokenizer.decode(gen_ids)
        return {
            "arm_id": self.arm_id,
            "hits": [{"text": continuation, "score": 1.0}] if continuation else [],
            "plastic_continuation": continuation,
            "logits_last": details["logits"][0, -1, :].detach().cpu(),
        }

    def erase_or_supersede(self, key_or_subject: str, new_text: Optional[str] = None) -> Dict[str, Any]:
        del key_or_subject
        if new_text:
            return self.ingest(new_text)
        return {"arm_id": self.arm_id, "status": "no_targeted_erase_in_weight_plasticity"}

    def reset(self) -> None:
        self.plastic_model.load_state_dict(self._clean_snapshot)
        for n, p in self.plastic_model.named_parameters():
            if p.requires_grad:
                self._anchor_params[n].copy_(p.detach())
                self._fisher_diag[n].zero_()
        self._ingested_count = 0

    def footprint_bytes(self) -> MemoryFootprint:
        param_b = sum(p.numel() * p.element_size() for p in self.plastic_model.parameters())
        fisher_b = sum(t.numel() * t.element_size() for t in self._fisher_diag.values())
        return MemoryFootprint(
            arm_id=self.arm_id,
            static_param_bytes=int(param_b),
            dynamic_store_bytes=int(fisher_b),
            active_items=self._ingested_count,
        )

    teach = ingest
    recall = query


def build_memory_arm(
    arm_id: str,
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    **kwargs: Any,
) -> nn.Module:
    key = arm_id.strip().upper()
    if key in {"M0", "PERSISTENT_CORE_STATE"}:
        return PersistentCoreStateArm(model, tokenizer)
    if key in {"M1", "ENGRAM_NGRAM_HASH"}:
        return EngramNGramHashArm(model, tokenizer, **kwargs)
    if key in {"M2", "SURPRISE_RESIDUAL_EPISODIC"}:
        return SurpriseResidualEpisodicArm(model, tokenizer, **kwargs)
    if key in {"M3", "HYBRID_KNOWLEDGE_RRF"}:
        return HybridKnowledgeRRFArm(model, tokenizer, **kwargs)
    if key in {"M4", "SPARSE_DELTA_ADAPTER"}:
        return SparseDeltaAdapterArm(model, tokenizer, **kwargs)
    if key in {"M5", "ONLINE_WEIGHT_PLASTICITY"}:
        return OnlineWeightPlasticityArm(model, tokenizer, **kwargs)
    raise ValueError(f"Unknown memory arm {arm_id!r}; expected M0..M5")
