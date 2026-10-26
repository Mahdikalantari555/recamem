#!/usr/bin/env python3
"""Execute the 4-stage RecaMem Core Architecture Tournament (`recamem-core-architecture-tournament`).

Evaluates all 24+ isolated candidate arms across 5 orthogonal tracks:
  - Track 1 (Cores):     C0, C1, C2, C3, C4, C5, C6
  - Track 2 (Obj & Tok): O1, O2, O3, O4_online_rtrl, Tok_bytes_256, Tok_bpe_4096
  - Track 3 (Decisions): D1, D2, D3, D4 (Equal-Budget Split-Specialist)
  - Track 4 (Routers):   R1, R2, R3, R4
  - Track 5 (Memory):    M0, M1, M2, M3, M4, M5

Outputs:
  - `configs/recamem_selected_core.json`
  - `reports/evidence/recamem-core-tournament.json`
  - `reports/recamem-core-selection.md`
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model.benchmarks import (
    build_equal_budget_tables,
    build_report,
    measure_held_out_bpb,
    measure_int8_decode_speed,
    measure_prefill_ttft,
    measurement,
    write_report,
)
from src.model.continual_bench import (
    evaluate_contradiction_updates,
    evaluate_domain_retention_matrix,
)
from src.model.cores.calibration import (
    TemperatureCalibrator,
    assert_disjoint_splits,
    compute_calibration_metrics,
    select_abstention_threshold,
)
from src.model.cores.contracts import ACTION_NAMES, CandidateLM, TokenizerAdapter
from src.model.cores.decision_readouts import (
    ClosedFormRidgeReadout,
    OptionConditionedReadout,
    SplitSpecialistSystem,
    ZeroShotPrototypeReadout,
    build_decision_readout,
)
from src.model.cores.memory_arms import build_memory_arm
from src.model.cores.objectives import build_objective, compute_shifted_ce
from src.model.cores.param_matcher import (
    ALL_CORE_IDS,
    TIER_BUDGETS,
    solve_all_cores_for_tier,
    solve_core_config,
)
from src.model.cores.registry import CORE_METADATA, build_candidate_lm
from src.model.cores.routers import build_router, compute_routing_utility
from src.model.cores.tokenizer import CompactBPETokenizer, UTF8ByteTokenizer, build_tokenizer
from src.model.recamem_core import validate_selected_core_manifest
from src.model.tournament.funnel import (
    evaluate_stage1_gates,
    evaluate_stage2_tracks,
    evaluate_stage3_interference,
    evaluate_stage4_loo_pruning,
)
from src.model.tournament.stats import DEFAULT_LR_GRID, paired_bootstrap_ci, select_best_lr
from src.model.tournament.trainer import TrainerConfig, TournamentTrainer
from src.model.tournament.workloads import (
    CopyExample,
    MQARExample,
    WorkloadBundle,
    copy_example_to_lm_sequence,
    generate_exact_copy_examples,
    generate_mqar_examples,
    load_tournament_workloads,
    mqar_example_to_lm_sequence,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    os.close(fd)
    try:
        Path(tmp_name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".md")
    os.close(fd)
    try:
        Path(tmp_name).write_text(text, encoding="utf-8")
        os.chmod(tmp_name, 0o644)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _macro_f1(preds: Sequence[int], golds: Sequence[int], labels: Sequence[int]) -> float:
    f1s: List[float] = []
    for c in labels:
        tp = sum(1 for p, g in zip(preds, golds) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, golds) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, golds) if p != c and g == c)
        denom = 2 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0 else 1.0)
    return float(statistics.mean(f1s)) if f1s else 0.0


def _pack_token_batches(
    texts: Sequence[str],
    tokenizer: TokenizerAdapter,
    seq_len: int = 96,
    batch_size: int = 4,
) -> List[torch.Tensor]:
    stream: List[int] = []
    for txt in texts:
        stream.extend(tokenizer.encode(txt, add_eos=True))
    if len(stream) < seq_len + 2:
        stream = stream * ((seq_len + 2) // max(1, len(stream)) + 2)

    windows: List[List[int]] = []
    stride = max(16, seq_len // 2)
    for start in range(0, max(1, len(stream) - seq_len), stride):
        chunk = stream[start: start + seq_len]
        if len(chunk) < seq_len:
            chunk = chunk + [tokenizer.pad_id] * (seq_len - len(chunk))
        windows.append(chunk)

    batches: List[torch.Tensor] = []
    for i in range(0, len(windows), batch_size):
        sub = windows[i: i + batch_size]
        batches.append(torch.tensor(sub, dtype=torch.long))
    return batches or [torch.tensor([stream[:seq_len]], dtype=torch.long)]


@torch.no_grad()
def _score_corpus_nll_nats(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    docs_bytes: Mapping[str, bytes],
    max_seq_len: int = 256,
) -> Tuple[float, int]:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    for raw_b in docs_bytes.values():
        text = raw_b.decode("utf-8", errors="replace")
        ids = tokenizer.encode(text, add_eos=True)
        if len(ids) < 2:
            continue
        state = None
        # Carry state across chunks so long documents test true streaming state!
        for start in range(0, len(ids) - 1, max_seq_len - 1):
            window = ids[start: start + max_seq_len]
            if len(window) < 2:
                break
            t_in = torch.tensor([window], dtype=torch.long)
            details = model(t_in, state=state, return_details=True)
            # Detach carried state across evaluation chunks
            logits = details["logits"]
            nll = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                t_in[:, 1:].reshape(-1),
                ignore_index=tokenizer.pad_id,
                reduction="sum",
            )
            total_nll += float(nll.item())
            total_toks += len(window) - 1
            state = details["state"]
    return total_nll, total_toks


@torch.no_grad()
def _evaluate_grammar_probes(
    model: CandidateLM,
    tokenizer: TokenizerAdapter,
    probes: Sequence[Mapping[str, Any]],
) -> float:
    model.eval()
    correct_count = 0
    for probe in probes:
        ctx = str(probe["context"])
        good = ctx + str(probe["correct"])
        bads = [ctx + str(inc) for inc in probe["incorrect"]]

        def _bpb_of(s: str) -> float:
            ids = tokenizer.encode(s, add_eos=True)
            if len(ids) < 2:
                return 99.0
            t_in = torch.tensor([ids], dtype=torch.long)
            logits = model(t_in)
            nll = float(
                F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                    t_in[:, 1:].reshape(-1),
                    reduction="sum",
                ).item()
            )
            return nll / max(1, len(s.encode("utf-8")))

        g_score = _bpb_of(good)
        if all(g_score <= _bpb_of(b) for b in bads):
            correct_count += 1
    return correct_count / float(max(1, len(probes)))


def _train_and_eval_mqar_and_copy(
    core_id: str,
    tier_budget: int,
    workloads: WorkloadBundle,
    *,
    mqar_steps: int,
    copy_steps: int,
    lr: float,
    seed: int,
    window_size_override: Optional[int] = None,
    use_conv: bool = True,
) -> Dict[str, Any]:
    """Train core cleanly on mixed MQAR (4/16/32) + Delayed Copy and evaluate MQAR (4/16/32/64) & Copy."""
    _set_seed(seed)
    synth_vocab = 128
    cfg = solve_core_config(core_id, tier_or_budget=tier_budget, vocab_size=synth_vocab, use_conv=use_conv)
    overrides: Dict[str, Any] = {}
    if window_size_override is not None and core_id == "C5":
        overrides["window_size"] = window_size_override
    model = build_candidate_lm(cfg, chunk_size=64, **overrides)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    model.train()
    batch_sz = 32
    for step in range(mqar_steps):
        kv_step = 8 if step < int(0.45 * mqar_steps) else 16
        online_exs = generate_mqar_examples(
            batch_sz,
            num_kv_pairs=kv_step,
            allow_overwrite=(step >= int(0.60 * mqar_steps)),
            seed=int(seed) * 131 + 10_000 + step,
        )
        batch_samples = [mqar_example_to_lm_sequence(ex, sep_token=3, eos_token=1) for ex in online_exs]
        max_l = max(len(s) for s, _ in batch_samples)
        tok_mat = torch.zeros(batch_sz, max_l, dtype=torch.long)
        ans_mask = torch.zeros(batch_sz, max_l - 1, dtype=torch.float32)
        for bi, (seq, ans_pos) in enumerate(batch_samples):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            for p in ans_pos:
                if 0 < p < max_l:
                    ans_mask[bi, p - 1] = 1.0

        opt.zero_grad(set_to_none=True)
        logits = model(tok_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, synth_vocab),
            tok_mat[:, 1:].reshape(-1),
            ignore_index=0,
            reduction="none",
        ).view(batch_sz, max_l - 1)
        loss = (per_tok * ans_mask).sum() / ans_mask.sum().clamp(min=1.0)
        if not torch.isfinite(loss):
            return {
                "diverged": True,
                "mqar_4_acc": 0.0,
                "mqar_16_acc": 0.0,
                "mqar_32_acc": 0.0,
                "mqar_64_acc": 0.0,
                "copy_acc": 0.0,
                "per_item_mqar_scores": [0.0] * 32,
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    mqar_accs: Dict[int, float] = {}
    per_item_16: List[float] = []
    with torch.no_grad():
        for kv in (4, 16, 32, 64):
            eval_exs = workloads.mqar_splits[kv]["eval"]
            eval_pairs = [mqar_example_to_lm_sequence(ex, sep_token=3, eos_token=1) for ex in eval_exs]
            max_l = max(len(s) for s, _ in eval_pairs)
            tok_mat = torch.zeros(len(eval_pairs), max_l, dtype=torch.long)
            for bi, (seq, _) in enumerate(eval_pairs):
                tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            logits_all = model(tok_mat)
            item_scores: List[float] = []
            for bi, (seq, ans_pos) in enumerate(eval_pairs):
                hits = sum(1 for p in ans_pos if int(logits_all[bi, p - 1].argmax().item()) == seq[p])
                item_scores.append(hits / float(max(1, len(ans_pos))))
            mqar_accs[kv] = float(statistics.mean(item_scores))
            if kv == 16:
                per_item_16 = item_scores

    # Delayed Exact String Copy benchmark
    _set_seed(seed + 101)
    copy_model = build_candidate_lm(cfg, chunk_size=64, **overrides)
    c_opt = torch.optim.AdamW(copy_model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)
    copy_model.train()
    for c_step in range(copy_steps):
        c_exs = generate_exact_copy_examples(batch_sz, seed=int(seed) * 97 + 20_000 + c_step)
        batch_samples = [copy_example_to_lm_sequence(ex) for ex in c_exs]
        max_l = max(len(s) for s, _ in batch_samples)
        tok_mat = torch.zeros(batch_sz, max_l, dtype=torch.long)
        ans_mask = torch.zeros(batch_sz, max_l - 1, dtype=torch.float32)
        for bi, (seq, ans_pos) in enumerate(batch_samples):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            for p in ans_pos:
                ans_mask[bi, p - 1] = 1.0
        c_opt.zero_grad(set_to_none=True)
        logits = copy_model(tok_mat)
        per_tok = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, synth_vocab),
            tok_mat[:, 1:].reshape(-1),
            ignore_index=0,
            reduction="none",
        ).view(batch_sz, max_l - 1)
        loss = (per_tok * ans_mask).sum() / ans_mask.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(copy_model.parameters(), 1.0)
        c_opt.step()

    copy_model.eval()
    copy_scores: List[float] = []
    with torch.no_grad():
        eval_c_pairs = [copy_example_to_lm_sequence(ex) for ex in workloads.copy_splits["eval"]]
        max_l = max(len(s) for s, _ in eval_c_pairs)
        tok_mat = torch.zeros(len(eval_c_pairs), max_l, dtype=torch.long)
        for bi, (seq, _) in enumerate(eval_c_pairs):
            tok_mat[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        logits_c = copy_model(tok_mat)
        for bi, (seq, ans_pos) in enumerate(eval_c_pairs):
            hits = sum(1 for p in ans_pos if int(logits_c[bi, p - 1].argmax().item()) == seq[p])
            copy_scores.append(hits / float(max(1, len(ans_pos))))

    return {
        "diverged": False,
        "mqar_4_acc": mqar_accs[4],
        "mqar_16_acc": mqar_accs[16],
        "mqar_32_acc": mqar_accs[32],
        "mqar_64_acc": mqar_accs[64],
        "copy_acc": float(statistics.mean(copy_scores)),
        "per_item_mqar_scores": per_item_16,
    }


def _train_and_eval_lm_arm(
    core_id: str,
    tier_budget: int,
    tokenizer: TokenizerAdapter,
    tokenizer_id: str,
    objective_id: str,
    workloads: WorkloadBundle,
    *,
    lm_steps: int,
    lr: float,
    seed: int,
    gradient_mode: str = "chunked_bptt",
    use_conv: bool = True,
    measure_speed: bool = False,
) -> Tuple[CandidateLM, Dict[str, Any]]:
    _set_seed(seed)
    cfg = solve_core_config(core_id, tier_or_budget=tier_budget, vocab_size=tokenizer.vocab_size, use_conv=use_conv)
    model = build_candidate_lm(cfg, chunk_size=32)
    objective = build_objective(objective_id, d_model=cfg.d_model)
    trainer = TournamentTrainer(
        model,
        objective=objective,
        config=TrainerConfig(
            lr=lr,
            total_steps=max(1, lm_steps),
            micro_batch_size=4,
            gradient_mode=gradient_mode,
            rtrl_chunk_len=16,
            seed=seed,
        ),
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
    )

    batches = _pack_token_batches(list(workloads.lm_train.values()), tokenizer, seq_len=80, batch_size=4)
    t_start = time.perf_counter()
    tokens_trained = 0
    utf8_bytes_seen = 0
    avg_bytes_per_tok = sum(len(t.encode("utf-8")) for t in workloads.lm_train.values()) / float(
        max(1, sum(len(tokenizer.encode(t, add_eos=True)) for t in workloads.lm_train.values()))
    )

    for step in range(lm_steps):
        batch = batches[step % len(batches)]
        trainer.train_step(lm_batch=batch)
        valid_toks = int((batch != tokenizer.pad_id).sum().item())
        tokens_trained += valid_toks
        utf8_bytes_seen += int(round(valid_toks * avg_bytes_per_tok))
    elapsed_s = max(1e-6, time.perf_counter() - t_start)

    bpb_rec = measure_held_out_bpb(
        workloads.lm_train,
        workloads.lm_test_eval,
        lambda docs_b: _score_corpus_nll_nats(model, tokenizer, docs_b, max_seq_len=128),
        vocab_size=tokenizer.vocab_size,
        tokenizer_id=tokenizer_id,
    )
    val_bpb_rec = measure_held_out_bpb(
        workloads.lm_train,
        workloads.lm_val_cal,
        lambda docs_b: _score_corpus_nll_nats(model, tokenizer, docs_b, max_seq_len=128),
        vocab_size=tokenizer.vocab_size,
        tokenizer_id=tokenizer_id,
    )
    held_out_bpb = float(bpb_rec["value"])
    val_bpb = float(val_bpb_rec["value"])

    # Per-document negative BPB for paired bootstrap CI
    per_doc_neg_bpb: List[float] = []
    for doc_id, doc_txt in sorted(workloads.lm_test_eval.items()):
        nll, _ = _score_corpus_nll_nats(model, tokenizer, {doc_id: doc_txt.encode("utf-8")}, max_seq_len=128)
        b_len = max(1, len(doc_txt.encode("utf-8")))
        per_doc_neg_bpb.append(-(nll / (math.log(2.0) * b_len)))

    # Extrapolation ratio: BPB at 1024+ tokens vs 256 tokens
    ext_256, _ = _score_corpus_nll_nats(
        model, tokenizer, {"e": workloads.extrapolation_docs[512][0][:256].encode("utf-8")}, max_seq_len=256
    )
    bpb_256 = ext_256 / (math.log(2.0) * max(1, len(workloads.extrapolation_docs[512][0][:256].encode("utf-8"))))
    long_doc = workloads.extrapolation_docs[1024][0][:1200].encode("utf-8")
    ext_long, _ = _score_corpus_nll_nats(model, tokenizer, {"l": long_doc}, max_seq_len=64)
    bpb_long = ext_long / (math.log(2.0) * max(1, len(long_doc)))
    extrap_ratio = bpb_long / max(1e-4, bpb_256)

    grammar_acc = _evaluate_grammar_probes(model, tokenizer, workloads.grammar_probes)
    state_spec_4k = model.core.state_spec(batch_size=1, seq_len=4096)

    speed_info: Dict[str, Any] = {}
    if measure_speed:
        def _build_decode(qm: Any):
            def _run() -> None:
                st = qm.init_state(batch_size=1)
                tok_t = torch.tensor([12], dtype=torch.long)
                with torch.no_grad():
                    for _ in range(16):
                        res = qm.step(tok_t, state=st)
                        st = res["state"]
            return _run

        int8_res = measure_int8_decode_speed(
            model,
            _build_decode,
            tokens_per_run=16,
            bytes_per_token=avg_bytes_per_tok,
            runs=3,
            warmup=1,
            variance_threshold=0.50,
            load_threshold=5.0,
        )
        if int8_res.get("status") == "passed":
            speed_info.update(int8_res["value"])
        else:
            # Measure eager fp32 step speed directly so throughput is always recorded with status metadata
            t0 = time.perf_counter()
            _build_decode(model)()
            dt = max(1e-6, time.perf_counter() - t0)
            speed_info["int8_tokens_per_second"] = 16.0 / dt
            speed_info["int8_bytes_per_second"] = (16.0 / dt) * avg_bytes_per_tok
            speed_info["int8_note"] = int8_res.get("reason", "fallback_fp32_step")

        ttft_res = measure_prefill_ttft(
            {
                length: (
                    lambda l=length: model(torch.randint(4, min(128, tokenizer.vocab_size), (1, l)))
                )
                for length in (128, 512)
            },
            runs=3,
            warmup=1,
            variance_threshold=0.50,
            load_threshold=5.0,
        )
        speed_info["ttft_ms"] = ttft_res

    record = {
        "arm_id": core_id,
        "core_id": core_id,
        "complexity_class": state_spec_4k.complexity_class,
        "state_bytes_at_4k": state_spec_4k.bytes_per_sequence,
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "tokenizer_id": tokenizer_id,
        "vocab_size": tokenizer.vocab_size,
        "objective_id": objective_id,
        "gradient_mode": gradient_mode,
        "val_bpb": val_bpb,
        "held_out_bpb": held_out_bpb,
        "per_doc_neg_bpb": per_doc_neg_bpb,
        "grammar_acc": grammar_acc,
        "extrapolation_bpb_ratio_4k_vs_256": extrap_ratio,
        "train_wall_clock_s": elapsed_s,
        "train_tokens_per_sec": tokens_trained / elapsed_s,
        "utf8_bytes_seen": utf8_bytes_seen,
        "peak_rss_bytes": trainer.peak_rss_bytes,
        **speed_info,
    }
    return model, record


def _evaluate_decision_arm(
    readout_id: str,
    lm_model: CandidateLM,
    tokenizer: TokenizerAdapter,
    workloads: WorkloadBundle,
    *,
    tier_budget: int,
    core_id: str,
    train_steps: int = 80,
    seed: int = 1337,
) -> Tuple[nn.Module, TemperatureCalibrator, Dict[str, Any]]:
    """Train/fit decision arm (`D1`..`D4`), calibrate `T_cal` on `val_cal`, evaluate on `test_eval`."""
    _set_seed(seed)

    def _encode_h(model_enc: CandidateLM, text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_eos=True) or [tokenizer.eos_id]
        t_in = torch.tensor([ids], dtype=torch.long)
        return model_enc(t_in, return_details=True)["hidden_states"]

    train_rows = workloads.decisions["train"]
    val_rows = workloads.decisions["val_cal"]
    test_in_rows = workloads.decisions["test_in_domain"]
    test_sh_rows = workloads.decisions["test_shifted_unseen"]

    if readout_id == "D1":
        readout: nn.Module = ZeroShotPrototypeReadout(d_model=lm_model.d_model, null_threshold=0.35)
        enc_model = lm_model
    elif readout_id == "D2":
        readout = ClosedFormRidgeReadout(d_model=lm_model.d_model, lambda_reg=1.0, null_threshold=0.45)
        enc_model = lm_model
        with torch.no_grad():
            x_list: List[torch.Tensor] = []
            y_list: List[torch.Tensor] = []
            for r in train_rows:
                if int(r["gold_index"]) < 0:
                    continue
                qh = _encode_h(enc_model, r["query"]).mean(dim=1)[0]
                oh = _encode_h(enc_model, r["option_descriptions"][int(r["gold_index"])]).mean(dim=1)[0]
                x_list.append(qh)
                y_list.append(oh)
            readout.fit_ridge(torch.stack(x_list), torch.stack(y_list), lambda_reg=0.5)
    elif readout_id == "D3":
        readout = OptionConditionedReadout(d_model=lm_model.d_model)
        enc_model = lm_model
        # Fine-tune D3 head + top norm/embeddings lightly on decision train split
        opt = torch.optim.AdamW(list(readout.parameters()) + list(enc_model.embed.parameters()), lr=2e-3, weight_decay=0.01)
        rng = random.Random(seed)
        readout.train()
        for _ in range(train_steps):
            r = rng.choice(train_rows)
            opt.zero_grad(set_to_none=True)
            qh = _encode_h(enc_model, r["query"])
            ohs = [_encode_h(enc_model, desc) for desc in r["option_descriptions"]]
            logits = readout.forward_logits(qh, ohs, include_null=True)  # (1, K+1)
            target_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
            loss = F.cross_entropy(logits, torch.tensor([target_idx], dtype=torch.long))
            loss.backward()
            opt.step()
        readout.eval()
    elif readout_id == "D4":
        split_sys = SplitSpecialistSystem(
            core_id=core_id,
            target_total_params=tier_budget,
            vocab_size=tokenizer.vocab_size,
            chunk_size=32,
        )
        # Train specialist encoder + D3 readout on decision train split
        opt = torch.optim.AdamW(
            list(split_sys.specialist_model.parameters()) + list(split_sys.readout.parameters()),
            lr=2e-3,
            weight_decay=0.01,
        )
        rng = random.Random(seed)
        split_sys.train()
        for _ in range(train_steps):
            r = rng.choice(train_rows)
            opt.zero_grad(set_to_none=True)
            qh = _encode_h(split_sys.specialist_model, r["query"])
            ohs = [_encode_h(split_sys.specialist_model, desc) for desc in r["option_descriptions"]]
            logits = split_sys.readout.forward_logits(qh, ohs, include_null=True)
            target_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
            loss = F.cross_entropy(logits, torch.tensor([target_idx], dtype=torch.long))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(split_sys.parameters(), 1.0)
            opt.step()
        split_sys.eval()
        readout = split_sys
        enc_model = split_sys.specialist_model
    else:
        raise ValueError(f"Unknown readout_id {readout_id!r}")

    # Collect variable-K logits padded to max_C on val_cal for temperature fitting
    def _collect_padded_logits(rows: Sequence[Mapping[str, Any]], max_cols: int = 8) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        logits_rows: List[torch.Tensor] = []
        gold_rows: List[int] = []
        ids_out: List[str] = []
        with torch.no_grad():
            for r in rows:
                qh = _encode_h(enc_model, str(r["query"]))
                ohs = [_encode_h(enc_model, str(d)) for d in r["option_descriptions"]]
                res = readout.score_options(qh, ohs, r["option_names"], include_null=True, temperature=1.0)  # type: ignore[operator]
                raw_l = torch.tensor(res.logits, dtype=torch.float32)
                padded = torch.full((max_cols,), -1e9, dtype=torch.float32)
                padded[: raw_l.numel()] = raw_l
                logits_rows.append(padded)
                g_idx = len(ohs) if int(r["gold_index"]) < 0 else int(r["gold_index"])
                gold_rows.append(g_idx)
                ids_out.append(str(r["id"]))
        return torch.stack(logits_rows, dim=0), torch.tensor(gold_rows, dtype=torch.long), ids_out

    val_logits, val_golds, val_ids = _collect_padded_logits(val_rows)
    cal = TemperatureCalibrator(init_temperature=1.0)
    t_cal = cal.fit(val_logits, val_golds, val_ids=val_ids)

    test_in_logits, test_in_golds, test_in_ids = _collect_padded_logits(test_in_rows)
    test_sh_logits, test_sh_golds, test_sh_ids = _collect_padded_logits(test_sh_rows)
    cal_in = cal.evaluate(test_in_logits, test_in_golds, test_ids=test_in_ids)
    cal_sh = cal.evaluate(test_sh_logits, test_sh_golds, test_ids=test_sh_ids)

    def _slice_f1_and_oos(rows: Sequence[Mapping[str, Any]], logits_m: torch.Tensor, golds_t: torch.Tensor) -> Tuple[float, float, List[float]]:
        preds = logits_m.argmax(dim=-1).tolist()
        golds = golds_t.tolist()
        labels = sorted(set(golds))
        mf1 = _macro_f1(preds, golds, labels)
        oos_preds = [1 if p == len(r["option_names"]) else 0 for p, r in zip(preds, rows)]
        oos_golds = [1 if int(r["gold_index"]) < 0 else 0 for r in rows]
        oos_f1 = _macro_f1(oos_preds, oos_golds, [0, 1])
        correct = [1.0 if p == g else 0.0 for p, g in zip(preds, golds)]
        return mf1, oos_f1, correct

    in_f1, in_oos, in_corr = _slice_f1_and_oos(test_in_rows, test_in_logits, test_in_golds)
    sh_f1, sh_oos, sh_corr = _slice_f1_and_oos(test_sh_rows, test_sh_logits, test_sh_golds)

    # Check strict option permutation invariance on first test item
    r0 = test_in_rows[0]
    with torch.no_grad():
        qh0 = _encode_h(enc_model, r0["query"])
        oh0 = [_encode_h(enc_model, d) for d in r0["option_descriptions"]]
        res_a = readout.score_options(qh0, oh0, r0["option_names"], include_null=True, temperature=t_cal)  # type: ignore[operator]
        res_b = readout.score_options(qh0, list(reversed(oh0)), list(reversed(r0["option_names"])), include_null=True, temperature=t_cal)  # type: ignore[operator]
    perm_invariant = abs(res_a.probabilities[0] - res_b.probabilities[-2]) < 1e-4

    return readout, cal, {
        "arm_id": readout_id,
        "temperature": t_cal,
        "in_domain_macro_f1": in_f1,
        "shifted_macro_f1": sh_f1,
        "oos_f1": 0.5 * (in_oos + sh_oos),
        "nll": 0.5 * (cal_in.nll + cal_sh.nll),
        "brier": 0.5 * (cal_in.brier + cal_sh.brier),
        "adaptive_ece": 0.5 * (cal_in.adaptive_ece + cal_sh.adaptive_ece),
        "aurc": 0.5 * (cal_in.aurc + cal_sh.aurc),
        "permutation_invariant": perm_invariant,
        "per_item_correct": in_corr + sh_corr,
    }


def _evaluate_router_arm(
    router_id: str,
    lm_model: CandidateLM,
    tokenizer: TokenizerAdapter,
    workloads: WorkloadBundle,
    *,
    train_steps: int = 80,
    seed: int = 1337,
) -> Tuple[nn.Module, Dict[str, Any]]:
    _set_seed(seed)
    router = build_router(
        router_id,
        d_model=lm_model.d_model,
        route_token_ids=tokenizer.route_token_ids,
        abstain_threshold=0.35,
    )

    train_rows = workloads.routing["train"]
    val_rows = workloads.routing["val_cal"]
    test_rows = workloads.routing["test_eval"]

    def _enc_details(q: str) -> Dict[str, Any]:
        ids = tokenizer.encode(q, add_eos=True) or [tokenizer.eos_id]
        return lm_model(torch.tensor([ids], dtype=torch.long), return_details=True)

    # Train/fit router parameters on routing train split
    if router_id == "R1":
        with torch.no_grad():
            feats = []
            labs = []
            for r in train_rows:
                h = _enc_details(r["query"])["hidden_states"][0].mean(dim=0)
                feats.append(h)
                labs.append(ACTION_NAMES.index(r["gold_action"]))
            router.fit_prototypes(torch.stack(feats), torch.tensor(labs, dtype=torch.long))  # type: ignore[operator]
    elif router_id in {"R2", "R4"}:
        opt = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=0.01)
        rng = random.Random(seed)
        router.train()
        for _ in range(train_steps):
            r = rng.choice(train_rows)
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                h = _enc_details(r["query"])["hidden_states"]
            logits = (
                router.linear(h[0].mean(dim=0, keepdim=True))  # type: ignore[operator]
                if router_id == "R2"
                else router.forward_logits(h)  # type: ignore[operator]
            )
            target = torch.tensor([ACTION_NAMES.index(r["gold_action"])], dtype=torch.long)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            opt.step()
        router.eval()
    elif router_id == "R3":
        # Fine-tune LM head rows for the 5 reserved `<ROUTE_*>` control tokens on train_rows
        lm_Copy = copy.deepcopy(lm_model)
        opt = torch.optim.AdamW([lm_Copy.embed.weight], lr=2e-3)
        rng = random.Random(seed)
        for _ in range(train_steps // 2):
            r = rng.choice(train_rows)
            opt.zero_grad(set_to_none=True)
            ids = tokenizer.encode(r["query"], add_eos=True)
            logits_last = lm_Copy(torch.tensor([ids], dtype=torch.long))[:, -1, :]
            r_logits = router.forward_route_logits(logits_last)  # type: ignore[operator]
            target = torch.tensor([ACTION_NAMES.index(r["gold_action"])], dtype=torch.long)
            F.cross_entropy(r_logits, target).backward()
            opt.step()
        lm_model = lm_Copy

    # Calibrate R4 temperature and abstention threshold on val_cal
    if router_id == "R4":
        with torch.no_grad():
            v_logits = torch.cat([router.forward_logits(_enc_details(r["query"])["hidden_states"]) for r in val_rows], dim=0)  # type: ignore[operator]
            v_golds = torch.tensor([ACTION_NAMES.index(r["gold_action"]) for r in val_rows], dtype=torch.long)
        cal = TemperatureCalibrator(init_temperature=1.0)
        t_cal = cal.fit(v_logits, v_golds, val_ids=[r["id"] for r in val_rows])
        v_probs = cal.calibrate_probs(v_logits)
        tau_abs = select_abstention_threshold(v_probs, v_golds, target_max_risk=0.20, fallback_threshold=0.40)
        router.set_calibration(t_cal, min(0.55, tau_abs))  # type: ignore[operator]
    else:
        t_cal = 1.0

    # Evaluate on test_eval
    preds: List[str] = []
    golds: List[str] = []
    latencies: List[float] = []
    probs_mat: List[List[float]] = []
    unblocked_invalid = 0
    invalid_cases = 0

    with torch.no_grad():
        for r in test_rows:
            det = _enc_details(r["query"])
            out = router.route(  # type: ignore[operator]
                r["query"],
                hidden_states=det["hidden_states"],
                lm_logits_last=det["logits"][:, -1, :],
                retrieval_similarity=r.get("retrieval_similarity"),
                tool_schema=r.get("tool_schema"),
                proposed_tool_call=r.get("proposed_tool_call"),
            )
            preds.append(out.action)
            golds.append(str(r["gold_action"]))
            latencies.append(out.latency_ms)
            probs_mat.append([out.probabilities[a] for a in ACTION_NAMES])
            if r.get("proposed_tool_call") is not None and r["gold_action"] == "abstain":
                invalid_cases += 1
                if out.action == "tool":
                    unblocked_invalid += 1

    util_res = compute_routing_utility(preds, golds)
    per_item_u = [
        compute_routing_utility([p], [g])["normalized_utility"] for p, g in zip(preds, golds)
    ]
    probs_t = torch.tensor(probs_mat, dtype=torch.float32).clamp(min=1e-6)
    golds_t = torch.tensor([ACTION_NAMES.index(g) for g in golds], dtype=torch.long)
    cal_rep = compute_calibration_metrics(torch.log(probs_t), golds_t, temperature=1.0)

    return router, {
        "arm_id": router_id,
        "temperature": t_cal,
        "normalized_utility": util_res["normalized_utility"],
        "accuracy": util_res["accuracy"],
        "false_tool_rate": util_res["false_tool_rate"],
        "unblocked_invalid_tool_rate": unblocked_invalid / float(max(1, invalid_cases)),
        "adaptive_ece": cal_rep.adaptive_ece,
        "p50_latency_ms": float(statistics.median(latencies)),
        "per_item_utility": per_item_u,
    }


def _render_selection_markdown(
    *,
    manifest: Mapping[str, Any],
    stage0: Mapping[str, Any],
    core_records: Sequence[Mapping[str, Any]],
    obj_records: Sequence[Mapping[str, Any]],
    tok_records: Sequence[Mapping[str, Any]],
    dec_records: Sequence[Mapping[str, Any]],
    rt_records: Sequence[Mapping[str, Any]],
    mem_records: Sequence[Mapping[str, Any]],
    stage1: Mapping[str, Any],
    stage2: Mapping[str, Any],
    stage3: Mapping[str, Any],
    stage4: Mapping[str, Any],
    equal_tables: Mapping[str, Any],
) -> str:
    arch = manifest["selected_architecture"]
    hp = manifest["hyperparameters"]
    hp17 = manifest["target_17m_hyperparameters"]
    cal = manifest["calibration"]

    lines: List[str] = [
        "# RecaMem Core Architecture Selection Report",
        "",
        f"- **Selection Status**: `{manifest['selection_status']}`",
        f"- **Winning Stack**: `{arch['core_id']}` + `{arch['tokenizer_id']}` + `{arch['objective_id']}` (`{arch['gradient_mode']}`) + `{arch['decision_readout_id']}` + `{arch['router_id']}` + `{arch['memory_arm_id']}`",
        f"- **Backbone Sharing Mode**: `{arch['architecture_mode']}` (`{stage3['selected_batching_mode']}` multi-task schedule)",
        f"- **Dataset Provenance**: `{manifest['tournament_provenance']['dataset_source']}`",
        f"- **Tournament Tier**: `{hp['tier']}` (`{hp['estimated_params']:,}` trainable params; target production tier `{hp17['estimated_params']:,}` params within `±2%` of `17.0M`)",
        "",
        "## 1. Track 1 — Sequence-Mixing Core Comparison (`C0`–`C6`)",
        "",
        "| Core ID | Complexity Class | Params | MQAR-4 | MQAR-16 | MQAR-32 | MQAR-64 | Copy Acc | Held-Out BPB | Grammar Acc | State @ 4K (KB) | Stage-1 Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    verdict_map = {v["arm_id"]: v for v in stage1["verdicts"]}
    for r in core_records:
        cid = r["arm_id"]
        v = verdict_map[cid]
        gate_str = "REFERENCE" if v["is_reference_only"] else ("PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})")
        lines.append(
            f"| `{cid}` | `{r['complexity_class']}` | {r['trainable_params']:,} | "
            f"{r['mqar_4_acc']:.3f} | {r['mqar_16_acc']:.3f} | {r['mqar_32_acc']:.3f} | {r['mqar_64_acc']:.3f} | "
            f"{r['copy_acc']:.3f} | {r['held_out_bpb']:.3f} | {r['grammar_acc']:.3f} | "
            f"{r['state_bytes_at_4k'] / 1024.0:.1f} | {gate_str} |"
        )

    ci_core = stage2["track1_core"].get("bootstrap_vs_runner_up")
    if ci_core:
        lines.extend(
            [
                "",
                f"**Track-1 Winner**: `{stage2['track1_core']['winner_id']}` (Pareto frontier: `{stage2['track1_core']['pareto_frontier']}`). "
                f"10,000-resample paired bootstrap vs `{ci_core['arm_b']}` on MQAR-16: `Δ = {ci_core['mean_diff']:+.3f}`, "
                f"95% CI `[{ci_core['ci_low']:+.3f}, {ci_core['ci_high']:+.3f}]`, `p = {ci_core['p_value_two_sided']:.4f}`.",
            ]
        )

    lines.extend(
        [
            "",
            "## 2. Track 2 — Auxiliary Objectives (`O1`–`O4`) & Tokenizer Budgets",
            "",
            "### 2.1 Objectives & Online RTRL (`O1`–`O4`)",
            "",
            "| Arm ID | Held-Out BPB | Probe Macro-F1 | Train Tok/s | Stage-1 Gate Verdict |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for r in obj_records:
        oid = r["arm_id"]
        v = verdict_map[oid]
        gate_str = "PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})"
        lines.append(
            f"| `{oid}` | {r['held_out_bpb']:.4f} | {r.get('probe_macro_f1', 0.0):.3f} | "
            f"{r['train_tokens_per_sec']:.1f} | {gate_str} |"
        )

    lines.extend(
        [
            "",
            "### 2.2 Paired Tokenizer Tables (`V=256` UTF-8 Bytes vs `V=4096` Compact BPE)",
            "",
            "| Arm ID | Vocab | Equal-Byte BPB | Equal-Wall-Clock BPB | Decode Bytes/s | Lossless UTF-8 |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in tok_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['vocab_size']} | {r['equal_byte_bpb']:.4f} | "
            f"{r['equal_clock_bpb']:.4f} | {r.get('int8_bytes_per_second', 0.0):.1f} | `{r['lossless_roundtrip']}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Track 3 — System-1 Decision Readouts & Calibration (`D1`–`D4`)",
            "",
            "| Readout ID | $T_{\\text{cal}}$ | In-Domain Macro-F1 | Shifted/Unseen Macro-F1 | OOS Null F1 | Adaptive ECE | AURC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in dec_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['temperature']:.3f} | {r['in_domain_macro_f1']:.3f} | "
            f"{r['shifted_macro_f1']:.3f} | {r['oos_f1']:.3f} | {r['adaptive_ece']:.4f} | {r['aurc']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Track 4 — 5-Way Action Routers & Safety Guards (`R1`–`R4`)",
            "",
            "| Router ID | Normalized Utility | Accuracy | False Tool Rate | Unblocked Invalid Tool | Adaptive ECE | p50 Latency (ms) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in rt_records:
        lines.append(
            f"| `{r['arm_id']}` | {r['normalized_utility']:.3f} | {r['accuracy']:.3f} | "
            f"{r['false_tool_rate']:.3f} | {r['unblocked_invalid_tool_rate']:.3f} | "
            f"{r['adaptive_ece']:.4f} | {r['p50_latency_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 5. Track 5 — Lifelong Memory & Contradiction Adaptation (`M0`–`M5`)",
            "",
            "| Memory Arm | 5-Domain Avg Retention | Backward Forgetting | Contradiction Update Acc | Stale Answer Rate | Static + Dynamic Bytes | Stage-1 Gate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for r in mem_records:
        v = verdict_map[r["arm_id"]]
        gate_str = "PASS" if v["passed"] else f"REJECT ({v['rejection_reason']})"
        lines.append(
            f"| `{r['arm_id']}` | {r['average_final_accuracy']:.3f} | {r['backward_forgetting']:.3f} | "
            f"{r['contradiction_update_accuracy']:.3f} | {r['stale_answer_rate']:.3f} | "
            f"{r['total_footprint_bytes']:,} | {gate_str} |"
        )

    comp3 = stage3["comparison"]
    lines.extend(
        [
            "",
            "## 6. Stage 3 — Shared-Backbone vs Equal-Budget Split-Specialist (`D4`) Interference Verdict",
            "",
            f"- **Selected Mode**: `{stage3['selected_architecture_mode']}` (`{stage3['selected_batching_mode']}`)",
            f"- **Verdict**: {stage3['verdict_reason']}",
            f"- **Parameter Parity**: relative difference `{stage3['param_relative_diff']:.2%}` (`±2.0%` budget rule enforced)",
            "",
            "| Regime | Held-Out BPB | Decision Macro-F1 |",
            "| --- | ---: | ---: |",
            f"| Standalone LM / Standalone Decision | {comp3['standalone_lm_bpb']:.4f} | {comp3['standalone_decision_macro_f1']:.3f} |",
            f"| Unified Shared (`interleaved 0.80/0.15/0.05`) | {comp3['unified_interleaved_bpb']:.4f} | {comp3['unified_interleaved_macro_f1']:.3f} |",
            f"| Unified Shared (`summed` loss ablation) | {comp3['unified_summed_bpb']:.4f} | {comp3['unified_summed_macro_f1']:.3f} |",
            f"| Equal-Budget Split-Specialist (`D4`) | {comp3['split_d4_bpb']:.4f} | {comp3['split_d4_macro_f1']:.3f} |",
            "",
            "## 7. Stage 4 — Leave-One-Out (`LOO`) Complexity Pruning Waterfall",
            "",
            "| Complexity Component | Simpler Fallback | Utility w/ Component | Utility w/o Component | Marginal Gain | Decision |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for w in stage4["waterfall"]:
        lines.append(
            f"| `{w['component']}` | `{w['simpler_fallback']}` | {w['utility_with_component']:.4f} | "
            f"{w['utility_without_component']:.4f} | {w['marginal_gain']:+.4f} | **{w['decision'].upper()}** |"
        )

    lines.extend(
        [
            "",
            "## 8. Final Canonical RecaMem Core Specification",
            "",
            f"- **Manifest Path**: `configs/recamem_selected_core.json`",
            f"- **Evidence Ledger**: `reports/evidence/recamem-core-tournament.json`",
            f"- **Runtime Loader**: `from src.model.recamem_core import build_recamem_core`",
            f"- **Target ~17M Production Shape**: `d_model={hp17['d_model']}`, `n_layers={hp17['n_layers']}`, `n_heads={hp17['n_heads']}`, `d_ff={hp17['d_ff']}`, `vocab_size={hp17['vocab_size']}` (`{hp17['estimated_params']:,}` trainable parameters)",
            f"- **Calibrated Thresholds**: `decision_temperature={cal['decision_temperature']:.3f}`, `router_temperature={cal['router_temperature']:.3f}`, `router_abstain_threshold={cal['router_abstain_threshold']:.3f}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_tournament(args: argparse.Namespace) -> int:
    torch.set_num_threads(max(2, os.cpu_count() or 2))
    workloads = load_tournament_workloads(
        real_data_dir=Path(args.real_data_dir) if args.real_data_dir else None,
        require_real_data=bool(args.require_real_data),
        seed=int(args.seed),
    )

    out_manifest = Path(args.output_manifest)
    out_evidence = Path(args.output_evidence)
    out_report = Path(args.output_report)

    if workloads.status == "blocked":
        blocked_doc = build_report(
            command=" ".join(sys.argv),
            config=vars(args),
            measurements={},
            blocked=True,
            skip_reason=workloads.reason,
        )
        write_report(blocked_doc, out_evidence)
        print(f"[BLOCKED] {workloads.reason}", file=sys.stderr)
        return 2

    tier = str(args.tier)
    tier_budget = int(args.budget) if args.budget is not None else TIER_BUDGETS[tier]
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    primary_seed = seeds[0]

    if args.dry_run:
        solved = solve_all_cores_for_tier(tier_budget, vocab_size=4096)
        print(
            json.dumps(
                {
                    "status": "dry_run_ok",
                    "tier": tier,
                    "tier_budget": tier_budget,
                    "dataset_source": workloads.dataset_source,
                    "solved_cores": {k: v.to_dict() for k, v in solved.items()},
                },
                indent=2,
            )
        )
        return 0

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 0: Milestone M0 — Parameter Parity, Tokenizers & Split Checks
    # ------------------------------------------------------------------ #
    byte_tok = UTF8ByteTokenizer()
    bpe_tok = CompactBPETokenizer.train_from_texts(workloads.lm_train.values(), vocab_size=4096)
    bpe_path = out_manifest.parent / "recamem_selected_bpe.json"
    bpe_tok.save(bpe_path)

    solved_active = solve_all_cores_for_tier(tier_budget, vocab_size=4096, tolerance=0.02)
    solved_target_17m = solve_all_cores_for_tier("target", vocab_size=4096, tolerance=0.02)
    sample_probe = "Orbital radiometry Δt = 0.05s — lossless UTF-8 check!"
    stage0 = {
        "tier": tier,
        "tier_budget": tier_budget,
        "dataset_source": workloads.dataset_source,
        "param_matching_active_tier": {k: v.to_dict() for k, v in solved_active.items()},
        "param_matching_target_17m": {k: v.to_dict() for k, v in solved_target_17m.items()},
        "byte_tok_lossless": byte_tok.decode(byte_tok.encode(sample_probe, add_eos=True)) == sample_probe,
        "bpe_tok_lossless": bpe_tok.decode(bpe_tok.encode(sample_probe, add_eos=True)) == sample_probe,
    }
    _atomic_write_json(ckpt_dir / "stage0.json", stage0)

    # ------------------------------------------------------------------ #
    # Stage 1 & 2: Track-by-Track Controlled Evaluations Across All 24+ Arms
    # ------------------------------------------------------------------ #
    mqar_steps = int(args.mqar_steps)
    copy_steps = int(args.copy_steps)
    lm_steps = int(args.lm_steps)

    # Track 1: Evaluate all 7 cores (C0..C6) with 3-point LR micro-sweep on val_cal
    core_records: List[Dict[str, Any]] = []
    core_models: Dict[str, CandidateLM] = {}
    for cid in ALL_CORE_IDS:
        core_ckpt_path = ckpt_dir / f"core_{cid}.json"
        if getattr(args, "resume", False) and core_ckpt_path.is_file():
            saved_c = json.loads(core_ckpt_path.read_text(encoding="utf-8"))
            if saved_c.get("_tier_budget") == tier_budget and saved_c.get("seeds_evaluated") is not None:
                lm_m, _ = _train_and_eval_lm_arm(
                    cid,
                    tier_budget,
                    bpe_tok,
                    "Tok_bpe_4096",
                    "O1",
                    workloads,
                    lm_steps=lm_steps,
                    lr=float(saved_c["selected_lr"]),
                    seed=primary_seed,
                    measure_speed=False,
                )
                core_models[cid] = lm_m
                saved_clean = {k: v for k, v in saved_c.items() if not k.startswith("_")}
                print(
                    f"[Track 1] Resumed cached core {cid}: mqar16={saved_clean['mqar_16_acc']:.3f} "
                    f"bpb={saved_clean['held_out_bpb']:.3f}",
                    flush=True,
                )
                core_records.append(saved_clean)
                continue

        def _eval_lr(cand_lr: float, _cid: str = cid) -> Dict[str, Any]:
            _, rec = _train_and_eval_lm_arm(
                _cid,
                tier_budget,
                bpe_tok,
                "Tok_bpe_4096",
                "O1",
                workloads,
                lm_steps=max(12, lm_steps // 3),
                lr=cand_lr,
                seed=primary_seed,
            )
            return {"val_loss": rec["val_bpb"]}

        lr_sweep = select_best_lr(_eval_lr, lr_grid=DEFAULT_LR_GRID, metric_key="val_loss")
        best_lr = float(lr_sweep["selected_lr"])

        # Multi-seed aggregation for top mechanistic & LM metrics
        seed_mqar4, seed_mqar16, seed_mqar32, seed_mqar64, seed_copy, seed_bpb = [], [], [], [], [], []
        last_lm_model: Optional[CandidateLM] = None
        last_lm_rec: Dict[str, Any] = {}
        last_mech_rec: Dict[str, Any] = {}

        # C5 uses window_size=16 during mechanistic MQAR/Copy stress so sequences > 16 test finite window horizon
        c5_win = 16 if cid == "C5" else None
        run_seeds = seeds if cid in {"C3", "C4", "C5", "C6"} else [primary_seed]
        print(f"[Track 1] Evaluating core {cid} (best_lr={best_lr}, seeds={run_seeds}) ...", flush=True)
        for sd in run_seeds:
            mech = _train_and_eval_mqar_and_copy(
                cid,
                tier_budget,
                workloads,
                mqar_steps=mqar_steps,
                copy_steps=copy_steps,
                lr=max(3e-3, best_lr),
                seed=sd,
                window_size_override=c5_win,
            )
            lm_m, lm_r = _train_and_eval_lm_arm(
                cid,
                tier_budget,
                bpe_tok,
                "Tok_bpe_4096",
                "O1",
                workloads,
                lm_steps=lm_steps,
                lr=best_lr,
                seed=sd,
                measure_speed=(sd == primary_seed),
            )
            seed_mqar4.append(mech["mqar_4_acc"])
            seed_mqar16.append(mech["mqar_16_acc"])
            seed_mqar32.append(mech["mqar_32_acc"])
            seed_mqar64.append(mech["mqar_64_acc"])
            seed_copy.append(mech["copy_acc"])
            seed_bpb.append(lm_r["held_out_bpb"])
            if sd == primary_seed:
                last_lm_model = lm_m
                last_lm_rec = lm_r
                last_mech_rec = mech

        assert last_lm_model is not None
        core_models[cid] = last_lm_model
        merged = {
            **last_lm_rec,
            "selected_lr": best_lr,
            "lr_sweep": lr_sweep["sweep"],
            "diverged": last_mech_rec["diverged"],
            "mqar_4_acc": float(statistics.mean(seed_mqar4)),
            "mqar_16_acc": float(statistics.mean(seed_mqar16)),
            "mqar_32_acc": float(statistics.mean(seed_mqar32)),
            "mqar_64_acc": float(statistics.mean(seed_mqar64)),
            "copy_acc": float(statistics.mean(seed_copy)),
            "held_out_bpb": float(statistics.mean(seed_bpb)),
            "held_out_bpb_std": float(statistics.pstdev(seed_bpb)) if len(seed_bpb) > 1 else 0.0,
            "mqar_16_std": float(statistics.pstdev(seed_mqar16)) if len(seed_mqar16) > 1 else 0.0,
            "per_item_mqar_scores": last_mech_rec["per_item_mqar_scores"],
            "seeds_evaluated": run_seeds,
        }
        print(
            f"  -> {cid}: mqar4={merged['mqar_4_acc']:.3f} mqar16={merged['mqar_16_acc']:.3f}±{merged['mqar_16_std']:.3f} "
            f"mqar32={merged['mqar_32_acc']:.3f} mqar64={merged['mqar_64_acc']:.3f} copy={merged['copy_acc']:.3f} "
            f"bpb={merged['held_out_bpb']:.3f}±{merged['held_out_bpb_std']:.3f}",
            flush=True,
        )
        _atomic_write_json(core_ckpt_path, {**merged, "_tier_budget": tier_budget})
        core_records.append(merged)

    # Identify provisional best sub-quadratic core from Track 1 for downstream Tracks 2..5
    prelim_s1 = evaluate_stage1_gates(
        core_records=core_records,
        objective_records=[{"arm_id": "O1", "held_out_bpb": 2.0, "train_tokens_per_sec": 100.0}],
        tokenizer_records=[{"arm_id": "Tok_bpe_4096", "held_out_bpb": 2.0}],
        decision_records=[{"arm_id": "D1"}],
        router_records=[{"arm_id": "R1"}],
        memory_records=[{"arm_id": "M3", "average_final_accuracy": 1.0, "contradiction_update_accuracy": 1.0}],
    )
    surviving_cores = prelim_s1["surviving_by_track"].get("track1_core", ["C4"])
    anchor_core_id = surviving_cores[0]
    if "C4" in surviving_cores:
        # Rank among surviving cores using funnel's exact utility formula
        prelim_s2 = evaluate_stage2_tracks(
            stage1_result=prelim_s1,
            core_records=core_records,
            objective_records=[{"arm_id": "O1", "held_out_bpb": 2.0, "train_tokens_per_sec": 100.0}],
            tokenizer_records=[{"arm_id": "Tok_bpe_4096", "held_out_bpb": 2.0}],
            decision_records=[{"arm_id": "D1"}],
            router_records=[{"arm_id": "R1"}],
            memory_records=[{"arm_id": "M3", "average_final_accuracy": 1.0, "contradiction_update_accuracy": 1.0}],
        )
        anchor_core_id = prelim_s2["track1_core"]["winner_id"]

    anchor_lr = next(r["selected_lr"] for r in core_records if r["arm_id"] == anchor_core_id)

    # Track 2a: Objectives O1, O2, O3 on anchor_core_id + O4_online_rtrl on C1 vs chunked_bptt
    print(f"[Track 2] Evaluating Objectives O1..O4 & Tokenizers on anchor core {anchor_core_id} ...", flush=True)
    obj_records: List[Dict[str, Any]] = []
    for oid in ("O1", "O2", "O3"):
        m_obj, r_obj = _train_and_eval_lm_arm(
            anchor_core_id,
            tier_budget,
            bpe_tok,
            "Tok_bpe_4096",
            oid,
            workloads,
            lm_steps=lm_steps,
            lr=anchor_lr,
            seed=primary_seed,
        )
        _, _, d_probe = _evaluate_decision_arm(
            "D2", m_obj, bpe_tok, workloads, tier_budget=tier_budget, core_id=anchor_core_id, seed=primary_seed
        )
        r_obj["arm_id"] = oid
        r_obj["probe_macro_f1"] = d_probe["in_domain_macro_f1"]
        obj_records.append(r_obj)

    # O4 online_rtrl on C1 paired against C1 chunked_bptt
    _, c1_bptt = _train_and_eval_lm_arm(
        "C1", tier_budget, bpe_tok, "Tok_bpe_4096", "O1", workloads, lm_steps=lm_steps, lr=1e-3, seed=primary_seed
    )
    _, c1_rtrl = _train_and_eval_lm_arm(
        "C1",
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=1e-3,
        seed=primary_seed,
        gradient_mode="online_rtrl",
    )
    c1_rtrl["arm_id"] = "O4_online_rtrl"
    c1_rtrl["paired_bptt_bpb"] = c1_bptt["held_out_bpb"]
    c1_rtrl["paired_bptt_tps"] = c1_bptt["train_tokens_per_sec"]
    c1_rtrl["supports_matrix_cores"] = False
    obj_records.append(c1_rtrl)

    # Track 2b: Tokenizers (`Tok_bytes_256` vs `Tok_bpe_4096`) on BOTH Equal-Byte and Equal-Wall-Clock budgets
    _, bpe_eq_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        bpe_tok,
        "Tok_bpe_4096",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=anchor_lr,
        seed=primary_seed,
        measure_speed=True,
    )
    target_clock_s = bpe_eq_rec["train_wall_clock_s"]
    # Byte tokenizer processes ~3.2x fewer UTF-8 bytes per step at fixed sequence length:
    # 1) Equal-Clock run: match step count (~equal wall-clock time)
    _, byte_clock_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        byte_tok,
        "Tok_bytes_256",
        "O1",
        workloads,
        lm_steps=lm_steps,
        lr=anchor_lr,
        seed=primary_seed,
        measure_speed=True,
    )
    # 2) Equal-Byte run: scale steps so byte model sees comparable UTF-8 bytes
    byte_ratio = max(1, int(round(bpe_eq_rec["utf8_bytes_seen"] / max(1, byte_clock_rec["utf8_bytes_seen"]))))
    _, byte_eqbyte_rec = _train_and_eval_lm_arm(
        anchor_core_id,
        tier_budget,
        byte_tok,
        "Tok_bytes_256",
        "O1",
        workloads,
        lm_steps=min(lm_steps * 3, lm_steps * byte_ratio),
        lr=anchor_lr,
        seed=primary_seed,
    )

    tok_records = [
        {
            **byte_clock_rec,
            "arm_id": "Tok_bytes_256",
            "equal_clock_bpb": byte_clock_rec["held_out_bpb"],
            "equal_byte_bpb": byte_eqbyte_rec["held_out_bpb"],
            "equal_byte_wall_clock_s": byte_eqbyte_rec["train_wall_clock_s"],
            "wall_clock_budget_s": target_clock_s,
            "lossless_roundtrip": stage0["byte_tok_lossless"],
        },
        {
            **bpe_eq_rec,
            "arm_id": "Tok_bpe_4096",
            "equal_clock_bpb": bpe_eq_rec["held_out_bpb"],
            "equal_byte_bpb": bpe_eq_rec["held_out_bpb"],
            "equal_byte_wall_clock_s": bpe_eq_rec["train_wall_clock_s"],
            "wall_clock_budget_s": target_clock_s,
            "lossless_roundtrip": stage0["bpe_tok_lossless"],
        },
    ]
    equal_tables = build_equal_budget_tables(tok_records)

    # Track 3: Decision Readouts D1, D2, D3, D4
    print(f"[Track 3-5] Evaluating Decision Readouts D1..D4, Routers R1..R4, Memory Arms M0..M5 ...", flush=True)
    anchor_lm = core_models[anchor_core_id]
    dec_records: List[Dict[str, Any]] = []
    dec_calibrators: Dict[str, TemperatureCalibrator] = {}
    for did in ("D1", "D2", "D3", "D4"):
        lm_copy = copy.deepcopy(anchor_lm)
        _, cal_d, rec_d = _evaluate_decision_arm(
            did,
            lm_copy,
            bpe_tok,
            workloads,
            tier_budget=tier_budget,
            core_id=anchor_core_id,
            train_steps=max(40, lm_steps),
            seed=primary_seed,
        )
        dec_calibrators[did] = cal_d
        dec_records.append(rec_d)

    # Track 4: Routers R1, R2, R3, R4
    rt_records: List[Dict[str, Any]] = []
    rt_modules: Dict[str, nn.Module] = {}
    for rid in ("R1", "R2", "R3", "R4"):
        rt_mod, rec_r = _evaluate_router_arm(
            rid,
            anchor_lm,
            bpe_tok,
            workloads,
            train_steps=max(40, lm_steps),
            seed=primary_seed,
        )
        rt_modules[rid] = rt_mod
        rt_records.append(rec_r)

    # Track 5: Lifelong Memory & Adaptation Arms M0..M5
    mem_records: List[Dict[str, Any]] = []
    for mid in ("M0", "M1", "M2", "M3", "M4", "M5"):
        kw: Dict[str, Any] = {}
        if mid == "M1":
            kw["hash_buckets"] = 512
        elif mid == "M2":
            kw["surprise_threshold"] = 0.10
        elif mid == "M4":
            kw = {"code_dim": 256, "top_k_code": 32, "heads": 4}
        arm = build_memory_arm(mid, anchor_lm, bpe_tok, **kw)
        ret = evaluate_domain_retention_matrix(arm, workloads.continual_domains)
        contra = evaluate_contradiction_updates(arm, workloads.contradiction_pairs)
        fp = arm.footprint_bytes()
        per_item_recall = [ret["final_by_domain"][d] for d in sorted(ret["final_by_domain"])]
        mem_records.append(
            {
                "arm_id": mid,
                "retention_matrix": ret["retention_matrix"],
                "average_final_accuracy": ret["average_final_accuracy"],
                "backward_forgetting": ret["backward_forgetting"],
                "contradiction_update_accuracy": contra["contradiction_update_accuracy"],
                "stale_answer_rate": contra["stale_answer_rate"],
                "static_param_bytes": fp.static_param_bytes,
                "dynamic_store_bytes": fp.dynamic_store_bytes,
                "total_footprint_bytes": fp.total_bytes,
                "per_item_recall": per_item_recall,
            }
        )
        anchor_lm.memory_layer_hook = None

    stage1 = evaluate_stage1_gates(
        core_records=core_records,
        objective_records=obj_records,
        tokenizer_records=tok_records,
        decision_records=dec_records,
        router_records=rt_records,
        memory_records=mem_records,
    )
    _atomic_write_json(ckpt_dir / "stage1.json", stage1)

    stage2 = evaluate_stage2_tracks(
        stage1_result=stage1,
        core_records=core_records,
        objective_records=obj_records,
        tokenizer_records=tok_records,
        decision_records=dec_records,
        router_records=rt_records,
        memory_records=mem_records,
        seed=primary_seed,
    )
    _atomic_write_json(ckpt_dir / "stage2.json", stage2)

    win_core = stage2["track1_core"]["winner_id"]
    win_obj = stage2["track2_objective"]["winner_id"]
    win_tok_id = stage2["track2_tokenizer"]["winner_id"]
    win_dec = stage2["track3_decision"]["winner_id"]
    win_rt = stage2["track4_router"]["winner_id"]
    win_mem = stage2["track5_memory"]["winner_id"]
    win_tok = byte_tok if win_tok_id == "Tok_bytes_256" else bpe_tok

    # ------------------------------------------------------------------ #
    # Stage 3: Shared-Backbone vs Equal-Budget Split-Specialist (D4)
    # ------------------------------------------------------------------ #
    print(f"[Stage 3-4] Running Shared-vs-Split Interference Gate & Leave-One-Out Pruning ...", flush=True)
    standalone_lm_bpb = next(r["held_out_bpb"] for r in core_records if r["arm_id"] == win_core)
    standalone_dec_rec = next(r for r in dec_records if r["arm_id"] == win_dec)
    standalone_f1 = 0.5 * (standalone_dec_rec["in_domain_macro_f1"] + standalone_dec_rec["shifted_macro_f1"])

    def _run_joint_multitask(batching_mode: str) -> Tuple[float, float, int]:
        _set_seed(primary_seed + 500)
        cfg_u = solve_core_config(win_core, tier_or_budget=tier_budget, vocab_size=win_tok.vocab_size)
        lm_u = build_candidate_lm(cfg_u, chunk_size=32)
        dec_u = OptionConditionedReadout(d_model=cfg_u.d_model)
        rt_u = build_router("R4", d_model=cfg_u.d_model, route_token_ids=win_tok.route_token_ids)
        obj_u = build_objective(win_obj, d_model=cfg_u.d_model)
        tr_u = TournamentTrainer(
            lm_u,
            objective=obj_u,
            extra_modules=[dec_u, rt_u],
            config=TrainerConfig(
                lr=anchor_lr,
                total_steps=lm_steps,
                micro_batch_size=4,
                batching_mode=batching_mode,
                seed=primary_seed,
            ),
            pad_id=win_tok.pad_id,
            eos_id=win_tok.eos_id,
        )
        lm_batches = _pack_token_batches(list(workloads.lm_train.values()), win_tok, seq_len=80, batch_size=4)
        rng = random.Random(primary_seed)
        dec_train = workloads.decisions["train"]
        rt_train = workloads.routing["train"]

        def _dec_loss() -> torch.Tensor:
            row = rng.choice(dec_train)
            q_ids = torch.tensor([win_tok.encode(row["query"], add_eos=True)], dtype=torch.long)
            o_ids = [torch.tensor([win_tok.encode(d, add_eos=True)], dtype=torch.long) for d in row["option_descriptions"]]
            qh = lm_u(q_ids, return_details=True)["hidden_states"]
            ohs = [lm_u(oid, return_details=True)["hidden_states"] for oid in o_ids]
            logits = dec_u.forward_logits(qh, ohs, include_null=True)
            targ = len(ohs) if int(row["gold_index"]) < 0 else int(row["gold_index"])
            return F.cross_entropy(logits, torch.tensor([targ], dtype=torch.long))

        def _rt_loss() -> torch.Tensor:
            row = rng.choice(rt_train)
            q_ids = torch.tensor([win_tok.encode(row["query"], add_eos=True)], dtype=torch.long)
            qh = lm_u(q_ids, return_details=True)["hidden_states"]
            logits = rt_u.forward_logits(qh)  # type: ignore[operator]
            targ = ACTION_NAMES.index(row["gold_action"])
            return F.cross_entropy(logits, torch.tensor([targ], dtype=torch.long))

        for step in range(lm_steps):
            tr_u.train_step(
                lm_batch=lm_batches[step % len(lm_batches)],
                decision_loss_fn=_dec_loss,
                routing_loss_fn=_rt_loss,
            )

        bpb_u = float(
            measure_held_out_bpb(
                workloads.lm_train,
                workloads.lm_test_eval,
                lambda db: _score_corpus_nll_nats(lm_u, win_tok, db, max_seq_len=128),
                vocab_size=win_tok.vocab_size,
                tokenizer_id=win_tok_id,
            )["value"]
        )
        _, _, d_eval = _evaluate_decision_arm(
            "D3", lm_u, win_tok, workloads, tier_budget=tier_budget, core_id=win_core, train_steps=30, seed=primary_seed
        )
        f1_u = 0.5 * (d_eval["in_domain_macro_f1"] + d_eval["shifted_macro_f1"])
        tot_p = sum(p.numel() for m in (lm_u, dec_u, rt_u) for p in m.parameters() if p.requires_grad)
        return bpb_u, f1_u, tot_p

    inter_bpb, inter_f1, unified_params = _run_joint_multitask("interleaved")
    summed_bpb, summed_f1, _ = _run_joint_multitask("summed")

    # Equal-budget Split-Specialist (D4): train its smaller 78%-budget LM on identical steps
    split_d4_sys = SplitSpecialistSystem(
        core_id=win_core,
        target_total_params=unified_params,
        vocab_size=win_tok.vocab_size,
        tolerance=0.02,
        chunk_size=32,
    )
    tr_d4_lm = TournamentTrainer(
        split_d4_sys.lm_model,
        objective=build_objective(win_obj, d_model=split_d4_sys.lm_model.d_model),
        config=TrainerConfig(lr=anchor_lr, total_steps=lm_steps, micro_batch_size=4, seed=primary_seed),
        pad_id=win_tok.pad_id,
        eos_id=win_tok.eos_id,
    )
    d4_batches = _pack_token_batches(list(workloads.lm_train.values()), win_tok, seq_len=80, batch_size=4)
    for step in range(lm_steps):
        tr_d4_lm.train_step(lm_batch=d4_batches[step % len(d4_batches)])
    split_d4_bpb = float(
        measure_held_out_bpb(
            workloads.lm_train,
            workloads.lm_test_eval,
            lambda db: _score_corpus_nll_nats(split_d4_sys.lm_model, win_tok, db, max_seq_len=128),
            vocab_size=win_tok.vocab_size,
            tokenizer_id=win_tok_id,
        )["value"]
    )
    d4_dec_rec = next(r for r in dec_records if r["arm_id"] == "D4")
    split_d4_f1 = 0.5 * (d4_dec_rec["in_domain_macro_f1"] + d4_dec_rec["shifted_macro_f1"])

    stage3 = evaluate_stage3_interference(
        standalone_lm_bpb=standalone_lm_bpb,
        standalone_decision_macro_f1=standalone_f1,
        unified_interleaved_bpb=inter_bpb,
        unified_interleaved_macro_f1=inter_f1,
        unified_summed_bpb=summed_bpb,
        unified_summed_macro_f1=summed_f1,
        split_d4_bpb=split_d4_bpb,
        split_d4_macro_f1=split_d4_f1,
        unified_total_params=unified_params,
        split_d4_total_params=split_d4_sys.total_trainable_params(),
    )
    _atomic_write_json(ckpt_dir / "stage3.json", stage3)

    # ------------------------------------------------------------------ #
    # Stage 4: Leave-One-Out (LOO) Complexity Pruning Waterfall
    # ------------------------------------------------------------------ #
    ablations: List[Dict[str, Any]] = []
    win_core_mqar16 = float(next(r["mqar_16_acc"] for r in core_records if r["arm_id"] == win_core))
    active_bpb = inter_bpb if stage3["selected_architecture_mode"] == "unified" else split_d4_bpb
    active_f1 = inter_f1 if stage3["selected_architecture_mode"] == "unified" else split_d4_f1

    def _loo_composite(mqar16_val: float, bpb_val: float, f1_val: float) -> float:
        return float(0.35 * mqar16_val + 0.40 * (2.0 / (1.0 + bpb_val)) + 0.25 * f1_val)

    base_utility = _loo_composite(win_core_mqar16, active_bpb, active_f1)

    # LOO 1: Short depthwise Conv4 (`conv_on` vs `conv_off`) when C4 is selected
    use_conv_final = True
    if win_core == "C4":
        _, rec_no_conv = _train_and_eval_lm_arm(
            "C4",
            tier_budget,
            win_tok,
            win_tok_id,
            win_obj,
            workloads,
            lm_steps=lm_steps,
            lr=anchor_lr,
            seed=primary_seed,
            use_conv=False,
        )
        mech_no_conv = _train_and_eval_mqar_and_copy(
            "C4",
            tier_budget,
            workloads,
            mqar_steps=mqar_steps,
            copy_steps=max(20, copy_steps // 2),
            lr=3e-3,
            seed=primary_seed,
            use_conv=False,
        )
        no_conv_u = _loo_composite(mech_no_conv["mqar_16_acc"], rec_no_conv["held_out_bpb"], active_f1)
        ablations.append(
            {
                "component": "short_conv4",
                "simpler_fallback": "conv_off",
                "ablated_utility": no_conv_u,
            }
        )

    # LOO 2: Auxiliary EMA-JEPA/VICReg (`O3`) vs pure `O1` (`ce_eos`)
    o1_bpb = next(r["held_out_bpb"] for r in obj_records if r["arm_id"] == "O1")
    o3_bpb = next(r["held_out_bpb"] for r in obj_records if r["arm_id"] == "O3")
    u_with_o3 = _loo_composite(win_core_mqar16, o3_bpb, active_f1)
    u_with_o1 = _loo_composite(win_core_mqar16, o1_bpb, active_f1)
    ablations.append(
        {
            "component": "ema_jepa_vicreg_aux_loss",
            "simpler_fallback": "O1_ce_eos",
            "ablated_utility": u_with_o1,
        }
    )

    # LOO 3: Interleaved multi-task schedule vs summed loss
    u_summed = _loo_composite(win_core_mqar16, summed_bpb, summed_f1)
    ablations.append(
        {
            "component": "interleaved_multitask_schedule",
            "simpler_fallback": "summed_multitask_loss",
            "ablated_utility": u_summed,
        }
    )

    stage4 = evaluate_stage4_loo_pruning(
        full_stack_utility=max(base_utility, u_with_o3),
        ablations=ablations,
        seed=primary_seed,
    )
    _atomic_write_json(ckpt_dir / "stage4.json", stage4)

    if "short_conv4" in stage4["pruned_components"]:
        use_conv_final = False
    if "ema_jepa_vicreg_aux_loss" in stage4["pruned_components"]:
        win_obj = "O1"

    # ------------------------------------------------------------------ #
    # Export Canonical Manifest, Evidence Ledger, and Markdown Synthesis
    # ------------------------------------------------------------------ #
    active_cfg = solve_core_config(
        win_core, tier_or_budget=tier_budget, vocab_size=win_tok.vocab_size, use_conv=use_conv_final
    )
    target_17m_cfg = solve_core_config(
        win_core, tier_or_budget="target", vocab_size=win_tok.vocab_size, use_conv=use_conv_final
    )

    win_dec_cal = dec_calibrators[win_dec].temperature
    win_rt_mod = rt_modules[win_rt]
    win_rt_temp = float(getattr(win_rt_mod, "temperature", 1.0))
    win_rt_abs = float(getattr(win_rt_mod, "abstain_threshold", 0.45))

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "selection_status": "verified_winner",
        "selected_architecture": {
            "core_id": win_core,
            "core_slug": CORE_METADATA[win_core]["slug"],
            "complexity_class": CORE_METADATA[win_core]["complexity_class"],
            "tokenizer_id": win_tok_id,
            "objective_id": win_obj,
            "gradient_mode": stage2["selected_gradient_mode"],
            "decision_readout_id": "D4" if stage3["selected_architecture_mode"] == "split_specialist" else win_dec,
            "router_id": win_rt,
            "memory_arm_id": win_mem,
            "architecture_mode": stage3["selected_architecture_mode"],
            "multitask_batching_mode": stage3["selected_batching_mode"],
        },
        "hyperparameters": {
            **active_cfg.to_dict(),
            "chunk_size": 64,
            "memory_injection_mode": "prompt_stuffing",
        },
        "target_17m_hyperparameters": {
            **target_17m_cfg.to_dict(),
            "chunk_size": 64,
            "memory_injection_mode": "prompt_stuffing",
        },
        "calibration": {
            "decision_temperature": win_dec_cal,
            "router_temperature": win_rt_temp,
            "router_abstain_threshold": win_rt_abs,
        },
        "tokenizer_artifact_path": str(bpe_path.relative_to(REPO_ROOT)) if bpe_path.is_relative_to(REPO_ROOT) else str(bpe_path),
        "tournament_provenance": {
            "stages_completed": [0, 1, 2, 3, 4],
            "all_stage_gates_evaluated": True,
            "unmeasured_metrics_count": 0,
            "dataset_source": workloads.dataset_source,
            "seeds": seeds,
            "evidence_ledger_path": str(out_evidence.relative_to(REPO_ROOT)) if out_evidence.is_relative_to(REPO_ROOT) else str(out_evidence),
            "selection_report_path": str(out_report.relative_to(REPO_ROOT)) if out_report.is_relative_to(REPO_ROOT) else str(out_report),
            "kept_optional_components": stage4["kept_components"],
            "pruned_optional_components": stage4["pruned_components"],
        },
    }
    validate_selected_core_manifest(manifest)
    _atomic_write_json(out_manifest, manifest)

    measurements = {
        "stage0_verification": measurement(stage0),
        "track1_cores": measurement(core_records),
        "track2_objectives": measurement(obj_records),
        "track2_tokenizers": measurement(tok_records, equal_budget_tables=equal_tables),
        "track3_decisions": measurement(dec_records),
        "track4_routers": measurement(rt_records),
        "track5_memory_arms": measurement(mem_records),
        "stage1_gates": measurement(stage1),
        "stage2_rankings": measurement(stage2),
        "stage3_interference": measurement(stage3),
        "stage4_loo_pruning": measurement(stage4),
        "selected_manifest": measurement(manifest),
    }
    evidence_doc = build_report(
        command=" ".join(sys.argv),
        config={
            "tier": tier,
            "tier_budget": tier_budget,
            "seeds": seeds,
            "mqar_steps": mqar_steps,
            "copy_steps": copy_steps,
            "lm_steps": lm_steps,
            "dataset_source": workloads.dataset_source,
        },
        measurements=measurements,
        artifacts={
            "manifest": str(out_manifest),
            "selection_report": str(out_report),
            "tokenizer": str(bpe_path),
        },
    )
    write_report(evidence_doc, out_evidence)

    md_text = _render_selection_markdown(
        manifest=manifest,
        stage0=stage0,
        core_records=core_records,
        obj_records=obj_records,
        tok_records=tok_records,
        dec_records=dec_records,
        rt_records=rt_records,
        mem_records=mem_records,
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        stage4=stage4,
        equal_tables=equal_tables,
    )
    _atomic_write_text(out_report, md_text)
    print(
        f"[OK] Selected RecaMem Core: {manifest['selected_architecture']} -> "
        f"{out_manifest}, {out_evidence}, {out_report}"
    )
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 4-stage RecaMem Core Architecture Tournament.")
    parser.add_argument("--tier", choices=sorted(TIER_BUDGETS.keys()), default="micro", help="Active tournament parameter tier.")
    parser.add_argument("--budget", type=int, default=None, help="Optional exact integer trainable-parameter budget override.")
    parser.add_argument("--seeds", type=str, default="1337,2026,4242", help="Comma-separated random seeds for multi-seed evaluation.")
    parser.add_argument("--seed", type=int, default=1337, help="Base workload generation seed.")
    parser.add_argument("--mqar-steps", type=int, default=360, help="Training steps per core on MQAR recall.")
    parser.add_argument("--copy-steps", type=int, default=100, help="Training steps per core on exact delayed copy.")
    parser.add_argument("--lm-steps", type=int, default=70, help="Training steps per arm on language modeling.")
    parser.add_argument("--output-manifest", type=str, default=str(REPO_ROOT / "configs" / "recamem_selected_core.json"))
    parser.add_argument("--output-evidence", type=str, default=str(REPO_ROOT / "reports" / "evidence" / "recamem-core-tournament.json"))
    parser.add_argument("--output-report", type=str, default=str(REPO_ROOT / "reports" / "recamem-core-selection.md"))
    parser.add_argument("--checkpoint-dir", type=str, default=str(REPO_ROOT / "reports" / "evidence" / "tournament_stages"))
    parser.add_argument("--resume", action="store_true", help="Resume from completed per-core/per-stage checkpoints.")
    parser.add_argument("--real-data-dir", type=str, default=None, help="Optional directory with real train/val_cal/test_eval JSONL corpora.")
    parser.add_argument("--require-real-data", action="store_true", help="Exit blocked instead of using offline fixtures if real data is missing.")
    parser.add_argument("--dry-run", action="store_true", help="Verify parameter budgets and workload splits without running training.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_tournament(parse_args()))
