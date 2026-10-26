"""Deterministic parameter-budget solver across all seven candidate core families.

Ensures every candidate core (`C0`..`C6`) instantiated inside ``CandidateLM`` at a
shared vocabulary size matches the requested tier budget (`micro` = 2M,
`small` = 6M, `target` = 17M, or custom integer budget) within `±2.0%`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

TIER_BUDGETS: Dict[str, int] = {
    "smoke": 250_000,
    "micro": 2_000_000,
    "small": 6_000_000,
    "target": 17_000_000,
}

CORE_ALIASES: Dict[str, str] = {
    "c0": "C0",
    "unstacked_rtu": "C0",
    "c1": "C1",
    "stacked_gated_rtu": "C1",
    "c2": "C2",
    "selective_ssm": "C2",
    "c3": "C3",
    "retnet_decay_matrix": "C3",
    "c4": "C4",
    "gated_delta_matrix": "C4",
    "c5": "C5",
    "needle_san": "C5",
    "c6": "C6",
    "softmax_transformer": "C6",
}

ALL_CORE_IDS: Tuple[str, ...] = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
DEFAULT_TOLERANCE = 0.02  # ±2.0%


def canonical_core_id(core_id: str) -> str:
    key = core_id.strip().lower()
    if key in CORE_ALIASES:
        return CORE_ALIASES[key]
    upper = core_id.strip().upper()
    if upper in ALL_CORE_IDS:
        return upper
    raise ValueError(f"Unknown core_id {core_id!r}; expected one of {ALL_CORE_IDS}")


@dataclass(frozen=True)
class MatchedCoreConfig:
    """Solved dimensions for a candidate core under a parameter budget."""

    core_id: str
    tier: str
    target_params: int
    estimated_params: int
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    conv_kernel: int = 4
    use_conv: bool = True
    window_size: int = 256
    state_dim: int = 16

    @property
    def relative_error(self) -> float:
        return abs(self.estimated_params - self.target_params) / float(self.target_params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "core_id": self.core_id,
            "tier": self.tier,
            "target_params": int(self.target_params),
            "estimated_params": int(self.estimated_params),
            "relative_error": float(self.relative_error),
            "vocab_size": int(self.vocab_size),
            "d_model": int(self.d_model),
            "n_layers": int(self.n_layers),
            "n_heads": int(self.n_heads),
            "d_ff": int(self.d_ff),
            "conv_kernel": int(self.conv_kernel),
            "use_conv": bool(self.use_conv),
            "window_size": int(self.window_size),
            "state_dim": int(self.state_dim),
        }


def _swiglu_params(d_model: int, d_ff: int) -> int:
    # RMSNorm (d) + W_gate (d * d_ff) + W_up (d * d_ff) + W_down (d_ff * d), no bias
    return d_model + 3 * d_model * d_ff


def estimate_candidate_lm_params(
    core_id: str,
    vocab_size: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    use_conv: bool = True,
    conv_kernel: int = 4,
    state_dim: int = 16,
) -> int:
    """Exact trainable parameter count of ``CandidateLM(core, vocab_size, tied=True)``."""
    cid = canonical_core_id(core_id)
    # CandidateLM shell: tied embedding (vocab * d) + final RMSNorm (d)
    shell = vocab_size * d_model + d_model

    if cid == "C0":
        # Each unstacked RTU layer: decay (d) + LayerNorm (2*d) + weights (d*d)
        # Optional channel MLP (SwiGLU) when d_ff > 0 to allow exact budget tuning
        per_layer = d_model + 2 * d_model + d_model * d_model
        if d_ff > 0:
            per_layer += _swiglu_params(d_model, d_ff)
        return shell + n_layers * per_layer

    if cid == "C1":
        # Candidate B stacked gated RTU:
        # pre_norm (d) + W_g (d*d + d bias) + W_in (d*d) + LN(s) (2*d) + W_out (d*d)
        per_layer = d_model + (d_model * d_model + d_model) + d_model * d_model + 2 * d_model + d_model * d_model
        if d_ff > 0:
            per_layer += _swiglu_params(d_model, d_ff)
        return shell + n_layers * per_layer

    if cid == "C2":
        # Selective diagonal SSM:
        # pre_norm (d) + in_proj (d -> 2*d_inner, d_inner = d) = 2*d*d
        # depthwise conv1d on d_inner (d * conv_kernel + d bias)
        # dt_B_C_proj: d -> (d + 2*state_dim) = d*(d + 2*state_dim) + d (dt_bias)
        # A_log (d * state_dim) + D_skip (d) + out_proj (d * d)
        # + SwiGLU FFN
        conv_p = d_model * conv_kernel + d_model
        ssm_p = (
            d_model
            + 2 * d_model * d_model
            + conv_p
            + d_model * (d_model + 2 * state_dim)
            + d_model
            + d_model * state_dim
            + d_model
            + d_model * d_model
        )
        per_layer = ssm_p + _swiglu_params(d_model, d_ff)
        return shell + n_layers * per_layer

    if cid == "C3":
        # RetNet-style pure-decay matrix recurrence:
        # pre_norm (d) + W_q, W_k, W_v, W_g, W_o (5 * d * d) + head_decay_logit (n_heads)
        # + group/rms norm on o (d) + SwiGLU FFN
        rec_p = d_model + 5 * d_model * d_model + n_heads + d_model
        per_layer = rec_p + _swiglu_params(d_model, d_ff)
        return shell + n_layers * per_layer

    if cid == "C4":
        # Candidate A Gated-Delta Matrix Core:
        # pre_norm (d) + W_q, W_k, W_v, W_g, W_o (5 * d * d)
        # + optional depthwise Conv4 on q, k, v: 3 * (d * conv_kernel)
        # + W_alpha (d * n_heads + n_heads) + W_beta (d * n_heads + n_heads)
        # + o_norm (d) + SwiGLU FFN
        conv_p = 3 * (d_model * conv_kernel) if use_conv else 0
        gates_p = 2 * (d_model * n_heads + n_heads)
        rec_p = d_model + 5 * d_model * d_model + conv_p + gates_p + d_model
        per_layer = rec_p + _swiglu_params(d_model, d_ff)
        return shell + n_layers * per_layer

    if cid == "C5":
        # Needle-style SAN (sliding-window GQA with n_kv_heads = max(1, n_heads // 2),
        # QK-RMSNorm (2 * head_dim), W_q (d*d), W_k (d*d_kv), W_v (d*d_kv), W_o (d*d),
        # pre_norm (d) + Hadamard-gated channel mixer: norm(d) + W_h1(d*d_ff) + W_h2(d*d_ff) + W_hout(d_ff*d))
        head_dim = d_model // n_heads
        n_kv = max(1, n_heads // 2)
        d_kv = n_kv * head_dim
        attn_p = d_model + 2 * d_model * d_model + 2 * d_model * d_kv + 2 * head_dim
        mixer_p = d_model + 3 * d_model * d_ff
        return shell + n_layers * (attn_p + mixer_p)

    if cid == "C6":
        # Standard Softmax Transformer (MHA + RoPE + QK-RMSNorm + pre-RMSNorm + SwiGLU)
        head_dim = d_model // n_heads
        attn_p = d_model + 4 * d_model * d_model + 2 * head_dim
        return shell + n_layers * (attn_p + _swiglu_params(d_model, d_ff))

    raise ValueError(f"Unsupported core_id {cid!r}")


def solve_core_config(
    core_id: str,
    tier_or_budget: str | int = "small",
    vocab_size: int = 4096,
    tolerance: float = DEFAULT_TOLERANCE,
    use_conv: bool = True,
    window_size: int = 256,
) -> MatchedCoreConfig:
    """Solve deterministic `(d_model, n_layers, n_heads, d_ff)` within `±tolerance`."""
    cid = canonical_core_id(core_id)
    if isinstance(tier_or_budget, str):
        key = tier_or_budget.strip().lower()
        if key not in TIER_BUDGETS:
            raise ValueError(f"Unknown tier {tier_or_budget!r}; expected one of {tuple(TIER_BUDGETS)}")
        tier_name = key
        target = TIER_BUDGETS[key]
    else:
        target = int(tier_or_budget)
        tier_name = "custom"

    if target <= 0 or vocab_size <= 0:
        raise ValueError("target_params and vocab_size must be positive")
    if tolerance <= 0.0:
        raise ValueError("tolerance must be positive")

    # Minimum possible model: d_model=16, n_layers=1
    min_possible = vocab_size * 16 + 16 + 16 * 16
    if target < min_possible * (1.0 - tolerance):
        raise ValueError(
            f"Target budget {target} is unsatisfiable for vocab_size={vocab_size} "
            f"(minimum viable model has >= {min_possible} parameters)"
        )

    layer_candidates = (1, 2, 4, 6, 8, 10, 12, 16)
    head_dims = (8, 16, 24, 32, 48, 64)
    head_counts = (2, 4, 6, 8, 12)

    best: Optional[MatchedCoreConfig] = None

    for n_layers in layer_candidates:
        for n_heads in head_counts:
            for head_dim in head_dims:
                d_model = n_heads * head_dim
                if d_model < 16 or d_model > 1024:
                    continue
                if d_model < 32 and target >= 200_000:
                    continue
                if n_layers == 1 and target >= 200_000:
                    continue
                # Base count with minimal d_ff
                min_dff = 0 if cid in ("C0", "C1") else max(32, d_model // 2)
                base_est = estimate_candidate_lm_params(
                    cid,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_ff=min_dff,
                    use_conv=use_conv,
                )
                if base_est > target * (1.0 + tolerance):
                    continue

                # Solve analytically for optimal integer d_ff >= min_dff
                per_dff_coeff = 3 * d_model * n_layers
                if min_dff == 0:
                    # Also account for the RMSNorm (d_model * n_layers) added when d_ff > 0
                    rem_zero = abs(base_est - target) / float(target)
                    cand_zero = MatchedCoreConfig(
                        core_id=cid,
                        tier=tier_name,
                        target_params=target,
                        estimated_params=base_est,
                        vocab_size=vocab_size,
                        d_model=d_model,
                        n_layers=n_layers,
                        n_heads=n_heads,
                        d_ff=0,
                        use_conv=use_conv,
                        window_size=window_size,
                    )
                    if rem_zero <= tolerance and (best is None or _config_rank(cand_zero) < _config_rank(best)):
                        best = cand_zero

                    base_with_norm = base_est + d_model * n_layers
                    needed = target - base_with_norm
                    d_ff = max(32, int(round(needed / float(per_dff_coeff))))
                else:
                    base_without_dff = base_est - 3 * d_model * min_dff * n_layers
                    needed = target - base_without_dff
                    d_ff = max(min_dff, int(round(needed / float(per_dff_coeff))))

                # Cap d_ff at sensible aspect ratio (<= 5 * d_model) so width/depth stay balanced
                if d_ff > 5 * d_model:
                    continue

                est = estimate_candidate_lm_params(
                    cid,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    use_conv=use_conv,
                )
                cand = MatchedCoreConfig(
                    core_id=cid,
                    tier=tier_name,
                    target_params=target,
                    estimated_params=est,
                    vocab_size=vocab_size,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    use_conv=use_conv,
                    window_size=window_size,
                )
                if cand.relative_error <= tolerance:
                    if best is None or _config_rank(cand) < _config_rank(best):
                        best = cand

    if best is None:
        raise ValueError(
            f"Could not match parameter budget {target} (±{tolerance*100:.1f}%) "
            f"for core {cid} with vocab_size={vocab_size}"
        )
    return best


def _config_rank(cfg: MatchedCoreConfig) -> Tuple[float, float, float]:
    """Prefer canonical aspect ratios (`d_ff ~ 2.67 * d_model`, balanced depth) and tiny error."""
    target_layers = 4 if cfg.target_params < 1_000_000 else (
        4 if cfg.target_params <= 2_500_000 else (6 if cfg.target_params <= 8_000_000 else 8)
    )
    ideal_dff = 0.0 if cfg.core_id in ("C0", "C1") and cfg.d_ff == 0 else 2.67 * cfg.d_model
    dff_aspect_pen = abs(cfg.d_ff - ideal_dff) / max(1.0, float(cfg.d_model))
    layer_pen = abs(cfg.n_layers - target_layers) * 0.25
    return (round(layer_pen + 0.15 * dff_aspect_pen, 4), cfg.relative_error, -cfg.d_model)


def solve_all_cores_for_tier(
    tier_or_budget: str | int = "small",
    vocab_size: int = 4096,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, MatchedCoreConfig]:
    return {
        cid: solve_core_config(cid, tier_or_budget=tier_or_budget, vocab_size=vocab_size, tolerance=tolerance)
        for cid in ALL_CORE_IDS
    }
