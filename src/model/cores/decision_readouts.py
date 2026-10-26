"""Track-3 System-1 non-autoregressive decision readouts: D1, D2, D3, and D4 (Split-Specialist)."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.cores.contracts import CandidateLM, DecisionResult
from src.model.cores.param_matcher import TIER_BUDGETS, solve_core_config
from src.model.cores.registry import build_candidate_lm
from src.model.decision_head import DecisionHead, OptionConditionedHead


def _normalized_entropy_confidence(probs: torch.Tensor) -> float:
    p = probs.detach().float().reshape(-1).clamp(min=1e-12)
    k = p.numel()
    if k <= 1:
        return 1.0
    ent = float(-(p * torch.log(p)).sum().item())
    return max(0.0, min(1.0, 1.0 - ent / math.log(float(k))))


def _mean_pool(h: torch.Tensor) -> torch.Tensor:
    if h.dim() == 3:
        return h.mean(dim=1)
    if h.dim() == 2:
        return h.mean(dim=0, keepdim=True)
    return h.view(1, -1)


def _make_decision_result(
    logits: torch.Tensor,
    option_names: Sequence[str],
    include_null: bool,
    temperature: float,
    t0: float,
    mode: str,
) -> DecisionResult:
    cal_logits = logits / max(1e-4, float(temperature))
    probs = F.softmax(cal_logits, dim=-1)
    idx = int(torch.argmax(probs).item())
    is_null = include_null and idx == len(option_names)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    return DecisionResult(
        selected_index=-1 if is_null else idx,
        selected_option="null" if is_null else str(option_names[idx]),
        probabilities=[float(v) for v in probs.detach().cpu().tolist()],
        logits=[float(v) for v in logits.detach().cpu().tolist()],
        confidence=float(probs[idx].item()),
        is_null_or_abstain=is_null,
        latency_ms=latency_ms,
        mode=mode,
    )


class ZeroShotPrototypeReadout(nn.Module):
    """`D1` (`zero_shot_prototype`): zero-shot cosine prototype readout wrapping `DecisionHead`."""

    readout_id = "D1"

    def __init__(
        self,
        d_model: int,
        null_threshold: float = 0.25,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.null_threshold = float(null_threshold)
        self.logit_scale = float(logit_scale)
        self.legacy_head = DecisionHead(dim=self.d_model)

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        if not (2 <= len(option_hiddens) <= 255):
            raise ValueError(f"Expected K in [2, 255] options, got {len(option_hiddens)}")
        t0 = time.perf_counter()
        q = F.normalize(_mean_pool(query_hidden)[0], dim=-1, eps=1e-6)
        opt_vecs = torch.stack(
            [F.normalize(_mean_pool(oh)[0], dim=-1, eps=1e-6) for oh in option_hiddens],
            dim=0,
        )  # (K, d)
        cos_sims = torch.mv(opt_vecs, q)  # (K,)
        names = list(option_names)
        if include_null:
            # Null score equals the rejection cosine threshold so max(cos) < threshold selects 'null'
            logits = torch.cat(
                [cos_sims * self.logit_scale, torch.tensor([self.null_threshold * self.logit_scale], device=q.device)]
            )
            names.append("null")
        else:
            logits = cos_sims * self.logit_scale

        return _make_decision_result(logits, option_names, include_null, temperature, t0, self.readout_id)


class ClosedFormRidgeReadout(nn.Module):
    """`D2` (`closed_form_ridge`): closed-form Ridge regression in bilinear option space."""

    readout_id = "D2"

    def __init__(
        self,
        d_model: int,
        lambda_reg: float = 1.0,
        null_threshold: float = 0.25,
        logit_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.lambda_reg = float(lambda_reg)
        self.null_threshold = float(null_threshold)
        self.logit_scale = float(logit_scale)
        self.register_buffer("w_ridge", torch.eye(self.d_model, dtype=torch.float32))
        self.fitted = False
        self.legacy_head = DecisionHead(dim=self.d_model)

    @torch.no_grad()
    def fit_ridge(
        self,
        query_features: torch.Tensor,
        target_option_features: torch.Tensor,
        lambda_reg: Optional[float] = None,
    ) -> torch.Tensor:
        """Fit closed-form Ridge projection `W = (X^T X + lambda I)^{-1} X^T Y_opt`."""
        reg = float(lambda_reg if lambda_reg is not None else self.lambda_reg)
        x = F.normalize(query_features.float(), dim=-1, eps=1e-6)
        y = F.normalize(target_option_features.float(), dim=-1, eps=1e-6)
        w = self.legacy_head.train_probe(x, y, lambda_reg=reg)
        self.w_ridge.copy_(w)
        self.fitted = True
        return self.w_ridge

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        if not (2 <= len(option_hiddens) <= 255):
            raise ValueError(f"Expected K in [2, 255] options, got {len(option_hiddens)}")
        t0 = time.perf_counter()
        q_raw = F.normalize(_mean_pool(query_hidden)[0], dim=-1, eps=1e-6)
        q_proj = F.normalize(q_raw @ self.w_ridge, dim=-1, eps=1e-6)
        opt_vecs = torch.stack(
            [F.normalize(_mean_pool(oh)[0], dim=-1, eps=1e-6) for oh in option_hiddens],
            dim=0,
        )
        sims = torch.mv(opt_vecs, q_proj)
        if include_null:
            logits = torch.cat(
                [sims * self.logit_scale, torch.tensor([self.null_threshold * self.logit_scale], device=q_raw.device)]
            )
        else:
            logits = sims * self.logit_scale

        return _make_decision_result(logits, option_names, include_null, temperature, t0, self.readout_id)


class OptionConditionedReadout(nn.Module):
    """`D3` (`option_conditioned`): Candidate A dynamic option-conditioned head + learned null."""

    readout_id = "D3"

    def __init__(self, d_model: int, proj_dim: Optional[int] = None) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.head = OptionConditionedHead(d_model=self.d_model, proj_dim=proj_dim)

    def forward_logits(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        include_null: bool = True,
    ) -> torch.Tensor:
        return self.head.forward_logits(query_hidden, list(option_hiddens), include_null=include_null)

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        t0 = time.perf_counter()
        logits_b = self.forward_logits(query_hidden, option_hiddens, include_null=include_null)
        logits = logits_b[0]
        return _make_decision_result(logits, option_names, include_null, temperature, t0, self.readout_id)


class SplitSpecialistSystem(nn.Module):
    """`D4` (`split_specialist`): Equal-budget LM + separate specialist encoder + `D3` head.

    Enforces `±2.0%` total trainable parameter parity (`lm_model + specialist_model + readout`)
    against the unified budget `target_total_params` so Shared-vs-Split comparisons never
    grant free parameter capacity to the split baseline.
    """

    readout_id = "D4"

    def __init__(
        self,
        core_id: str,
        target_total_params: int | str,
        vocab_size: int = 4096,
        lm_budget_share: float = 0.78,
        tolerance: float = 0.02,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(target_total_params, str):
            total_budget = TIER_BUDGETS[target_total_params]
        else:
            total_budget = int(target_total_params)
        if not (0.25 <= lm_budget_share <= 0.90):
            raise ValueError(f"lm_budget_share={lm_budget_share} must lie in [0.25, 0.90]")

        # Solve LM and Specialist sub-budgets iteratively so (LM + Specialist + D3 head) lands within ±2%
        share_grid = [lm_budget_share] + [s for s in (0.72, 0.65, 0.58, 0.50) if abs(s - lm_budget_share) > 1e-4]
        best_pair: Optional[Tuple[float, CandidateLM, CandidateLM, OptionConditionedReadout]] = None

        for share in share_grid:
            lm_target = int(round(total_budget * share))
            try:
                lm_cfg = solve_core_config(core_id, tier_or_budget=lm_target, vocab_size=vocab_size, tolerance=0.08)
            except ValueError:
                continue
            cand_lm: CandidateLM = build_candidate_lm(lm_cfg, chunk_size=chunk_size)
            lm_actual = sum(p.numel() for p in cand_lm.parameters() if p.requires_grad)

            remaining = max(32_000, total_budget - lm_actual)
            for head_reserve_frac in (0.02, 0.04, 0.08, 0.12, 0.16, 0.20):
                spec_target = max(24_000, int(round(remaining * (1.0 - head_reserve_frac))))
                try:
                    spec_cfg = solve_core_config(
                        core_id,
                        tier_or_budget=spec_target,
                        vocab_size=vocab_size,
                        tolerance=0.15,
                    )
                except ValueError:
                    continue
                spec_lm = build_candidate_lm(spec_cfg, chunk_size=chunk_size)
                readout = OptionConditionedReadout(d_model=spec_cfg.d_model)
                tot = (
                    lm_actual
                    + sum(p.numel() for p in spec_lm.parameters() if p.requires_grad)
                    + sum(p.numel() for p in readout.parameters() if p.requires_grad)
                )
                rel_err = abs(tot - total_budget) / float(total_budget)
                if best_pair is None or rel_err < best_pair[0]:
                    best_pair = (rel_err, cand_lm, spec_lm, readout)
            if best_pair is not None and best_pair[0] <= tolerance:
                break

        if best_pair is None or best_pair[0] > tolerance:
            best_err = best_pair[0] if best_pair is not None else float("inf")
            raise ValueError(
                f"SplitSpecialistSystem could not match budget={total_budget} within ±{tolerance:.1%} "
                f"(best error={best_err:.2%})"
            )

        self.relative_budget_error = best_pair[0]
        self.target_total_params = total_budget
        self.lm_model: CandidateLM = best_pair[1]
        self.specialist_model: CandidateLM = best_pair[2]
        self.readout: OptionConditionedReadout = best_pair[3]

    def total_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_for_decision(self, token_ids: torch.Tensor) -> torch.Tensor:
        details = self.specialist_model(token_ids, return_details=True)
        return details["hidden_states"]

    def score_token_options(
        self,
        query_ids: torch.Tensor,
        option_id_list: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        q_h = self.encode_for_decision(query_ids)
        opt_h = [self.encode_for_decision(oid) for oid in option_id_list]
        return self.readout.score_options(
            q_h,
            opt_h,
            option_names=option_names,
            include_null=include_null,
            temperature=temperature,
        )

    def score_options(
        self,
        query_hidden: torch.Tensor,
        option_hiddens: Sequence[torch.Tensor],
        option_names: Sequence[str],
        include_null: bool = True,
        temperature: float = 1.0,
    ) -> DecisionResult:
        return self.readout.score_options(
            query_hidden,
            option_hiddens,
            option_names=option_names,
            include_null=include_null,
            temperature=temperature,
        )


def build_decision_readout(readout_id: str, d_model: int, **kwargs: Any) -> nn.Module:
    key = readout_id.strip().upper()
    if key in {"D1", "ZERO_SHOT_PROTOTYPE"}:
        return ZeroShotPrototypeReadout(d_model=d_model, **kwargs)
    if key in {"D2", "CLOSED_FORM_RIDGE"}:
        return ClosedFormRidgeReadout(d_model=d_model, **kwargs)
    if key in {"D3", "OPTION_CONDITIONED"}:
        return OptionConditionedReadout(d_model=d_model, **kwargs)
    raise ValueError(f"Unknown readout_id {readout_id!r}; use SplitSpecialistSystem directly for D4")
