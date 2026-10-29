#!/usr/bin/env python3
"""Standalone LM training for RecaMem cores (Phase 0-3).

Usage:
    PYTHONPATH=. python scripts/train_lm.py --core C4 --d-model 192 --n-layers 6 \\
        --n-heads 4 --d-ff 768 --vocab-size 1024 --seq-len 256 \\
        --lm-tokens 20000000 --data-dir data/tinystories --out-dir runs/lm_c4_1m

Features:
    - Token-based training (--lm-tokens, not steps)
    - AdamW (betas 0.9/0.95, wd 0.1, clip 1.0)
    - Warmup + cosine LR decay
    - Checkpoint/resume every --checkpoint-every steps
    - Generation logging every 1000 steps
    - Model config printing in report
    - G0-G4 gate evaluation
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model.cores.registry import build_candidate_lm


def load_memmap(data_dir: Path):
    meta = json.loads((data_dir / "meta.json").read_text())
    train = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")
    return train, val, meta


def get_batch(data: np.ndarray, seq_len: int, batch_size: int, device: torch.device):
    # Random contiguous windows
    max_start = len(data) - seq_len - 1
    starts = np.random.randint(0, max(max_start, 1), size=batch_size)
    x = np.stack([data[s:s + seq_len] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + seq_len] for s in starts]).astype(np.int64)
    return torch.tensor(x, device=device), torch.tensor(y, device=device)


def compute_val_loss(model, val_data, seq_len, batch_size, device, n_batches=20):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = get_batch(val_data, seq_len, batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


def generate_sample(model, tokenizer, prompt_ids, max_new=64, temperature=0.8, device=None):
    model.eval()
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_new):
            x = torch.tensor([ids[-256:]], dtype=torch.long, device=device)
            logits = model(x)[:, -1, :]
            # temperature sampling
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1).item())
            if hasattr(tokenizer, "eos_id") and nxt == tokenizer.eos_id:
                break
            ids.append(nxt)
    model.train()
    return ids


def save_checkpoint(path: Path, model, optimizer, step, tokens_seen, meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "meta": meta,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, optimizer):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt["step"], ckpt["tokens_seen"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone LM training")
    parser.add_argument("--core", type=str, default="C4")
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=768)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--lm-tokens", type=int, default=20_000_000)
    parser.add_argument("--data-dir", type=str, default="data/tinystories")
    parser.add_argument("--out-dir", type=str, default="runs/lm")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup-frac", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None, dest="resume_from")
    parser.add_argument("--push-to-hub", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--core-id", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)

    data_dir = Path(args.data_dir)
    train_data, val_data, meta = load_memmap(data_dir)
    V = meta["vocab_size"]
    assert V == args.vocab_size, f"V mismatch: data={V} arg={args.vocab_size}"

    # Build model
    cfg = {
        "core_id": args.core,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "vocab_size": V,
        "window_size": args.context,
        "use_conv": True,
        "conv_kernel": 4,
        "state_dim": 16,
    }
    model = build_candidate_lm(cfg, tied_embeddings=True, chunk_size=64)
    model.to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    embed_params = model.embed.weight.numel()
    non_embed = n_params - embed_params
    print(f"Model config: core={args.core} d={args.d_model} layers={args.n_layers} "
          f"heads={args.n_heads} V={V} params={n_params:,} embed={embed_params:,} "
          f"non-embed={non_embed:,} ({100*non_embed/n_params:.1f}%)")

    # Optimizer
    # Separate params for weight decay (no wd on norm/bias)
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 0.1},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # LR schedule
    total_tokens = args.lm_tokens
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    total_steps = max(1, total_tokens // tokens_per_step)
    warmup_steps = max(1, int(total_steps * args.warmup_frac))

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"
    samples_path = out_dir / "samples.txt"
    report_path = out_dir / "report.json"

    step = 0
    tokens_seen = 0
    resume_path = args.resume or args.resume_from
    if resume_path and Path(resume_path).is_file():
        # config hash check
        from recamem import hub as _hub
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("meta", {}).get("config", {})
        req_cfg = {"vocab_size": V, "d_model": args.d_model, "n_layers": args.n_layers,
                   "n_heads": args.n_heads, "d_ff": args.d_ff, "seed": args.seed}
        if ckpt_cfg:
            _hub.check_config_match(ckpt_cfg, req_cfg)
        step, tokens_seen = load_checkpoint(Path(resume_path), model, optimizer)
        print(f"Resumed from {resume_path}: step={step} tokens={tokens_seen}")

    # Overfit single batch test (G0) — optional, triggered by env
    g0_pass = None

    last_val_loss = 0.0
    t0 = time.time()
    print(f"Training started: {total_steps} steps, {total_tokens} tokens, {tokens_per_step} tok/step", flush=True)
    model.train()
    optimizer.zero_grad()

    log_every = 1
    val_every = 50

    while tokens_seen < total_tokens:
        # LR update
        for pg in optimizer.param_groups:
            pg["lr"] = lr_at(step)

        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.seq_len, args.batch_size, device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item() * args.grad_accum
            tokens_seen += args.batch_size * args.seq_len

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        if step % log_every == 0:
            elapsed = time.time() - t0
            toks_per_s = tokens_seen / max(elapsed, 1e-6)
            print(f"step={step} tokens={tokens_seen:,} loss={accum_loss:.4f} "
                  f"lr={lr_at(step):.2e} tok/s={toks_per_s:.0f}", flush=True)
            # live progress (every log step)
            try:
                remaining_tokens = max(0, total_tokens - tokens_seen)
                eta_s = remaining_tokens / max(toks_per_s, 1e-6)
                prog = {
                    "step": step, "tokens_seen": tokens_seen,
                    "val_loss": last_val_loss,
                    "total_steps": total_steps,
                    "elapsed_s": elapsed, "toks_per_s": toks_per_s,
                    "eta_s": eta_s, "remaining_tokens": remaining_tokens,
                    "loss": accum_loss,
                }
                (out_dir / "progress.json").write_text(json.dumps(prog))
            except Exception:
                pass
            # live progress
            try:
                remaining_tokens = max(0, total_tokens - tokens_seen)
                eta_s = remaining_tokens / max(toks_per_s, 1e-6)
                prog = {
                    "step": step, "tokens_seen": tokens_seen, "loss": accum_loss,
                    "val_loss": last_val_loss, "total_steps": total_steps,
                    "elapsed_s": elapsed, "toks_per_s": toks_per_s,
                    "eta_s": eta_s, "remaining_tokens": remaining_tokens,
                    "lr": lr_at(step),
                }
                (out_dir / "progress.json").write_text(json.dumps(prog))
            except Exception:
                pass

        if step % val_every == 0:
            vl = compute_val_loss(model, val_data, args.seq_len, args.batch_size, device)
            last_val_loss = vl
            print(f"  val_loss={vl:.4f} val_ppl={math.exp(min(vl,20)):.2f}", flush=True)
            # write live progress
            try:
                elapsed = time.time() - t0
                toks_per_s = tokens_seen / max(elapsed, 1e-6)
                remaining_tokens = max(0, total_tokens - tokens_seen)
                eta_s = remaining_tokens / max(toks_per_s, 1e-6)
                prog = {
                    "step": step, "tokens_seen": tokens_seen, "val_loss": vl,
                    "val_ppl": math.exp(min(vl, 20)), "total_steps": total_steps,
                    "elapsed_s": elapsed, "toks_per_s": toks_per_s,
                    "eta_s": eta_s, "remaining_tokens": remaining_tokens,
                }
                (out_dir / "progress.json").write_text(json.dumps(prog))
            except Exception:
                pass

        if step % args.checkpoint_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, {"V": V, "config": {"vocab_size": V, "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads, "d_ff": args.d_ff, "seed": args.seed}})
            print(f"  checkpoint saved: step={step}", flush=True)
            # Generation sample
            prompt = train_data[:8].tolist()
            gen_ids = generate_sample(model, None, prompt, max_new=64, temperature=0.8, device=device)
            txt = f"[step {step}] " + " ".join(str(t) for t in gen_ids[:40]) + "\n"
            with samples_path.open("a") as f:
                f.write(txt)

        if step >= total_steps:
            break

    # Compute bytes per token for BPB
    try:
        import json as _j
        # try to load val stories for byte count
        splits_dir = Path("splits")
        # fallback: estimate from data dir meta if present
    except Exception:
        pass
    # Load n-gram baselines if available
    ngram_path = data_dir / "ngram_baselines.json"
    # also check splits dir
    if not ngram_path.is_file():
        cand = Path(".cache/tinystories/splits/ngram_baselines.json")
        if cand.is_file():
            ngram_path = cand
    ngram = {}
    if ngram_path.is_file():
        ngram = json.loads(ngram_path.read_text())
    thresholds = ngram.get("thresholds", {})
    g2_thr = thresholds.get("G2", float("inf"))
    g3_thr = thresholds.get("G3", float("inf"))
    g1_thr = thresholds.get("G1", float("inf"))

    # Estimate bytes per token
    # Compute from val stories if splits available
    bytes_per_token = 4.0
    try:
        cand_splits = Path(".cache/tinystories/splits")
        if (cand_splits / "val_cal.jsonl").is_file():
            vstories = []
            with (cand_splits / "val_cal.jsonl").open() as f:
                for line in f:
                    line=line.strip()
                    if line:
                        vstories.append(json.loads(line).get("text", ""))
            total_bytes = sum(len(s.encode("utf-8")) for s in vstories)
            total_tokens = len(val_data)
            if total_tokens > 0:
                bytes_per_token = total_bytes / total_tokens
    except Exception:
        pass

    # Final val loss
    final_val = compute_val_loss(model, val_data, args.seq_len, args.batch_size, device)
    val_bpb = (final_val / math.log(2)) / max(bytes_per_token, 1e-6)

    # Gate evaluation
    g0 = None  # set externally
    g1 = val_bpb <= g1_thr if g1_thr != float("inf") else None
    g2 = val_bpb <= g2_thr if g2_thr != float("inf") else None
    g3 = (val_bpb <= g3_thr) if g3_thr != float("inf") else None
    # G4 (MQAR) not computed here; placeholder
    g4 = None

    # Final checkpoint
    save_checkpoint(ckpt_path, model, optimizer, step, tokens_seen, {"V": V, "config": {"vocab_size": V, "d_model": args.d_model, "n_layers": args.n_layers, "n_heads": args.n_heads, "d_ff": args.d_ff, "seed": args.seed}})

    # Final report
    elapsed = time.time() - t0
    report = {
        "core": args.core,
        "d_model": args.d_model,
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "vocab_size": V,
        "seq_len": args.seq_len,
        "lm_tokens_target": total_tokens,
        "tokens_seen": tokens_seen,
        "steps": step,
        "final_val_loss": final_val,
        "final_val_ppl": math.exp(min(final_val, 20)),
        "val_bpb": val_bpb,
        "bytes_per_token": bytes_per_token,
        "params": n_params,
        "embed_params": embed_params,
        "non_embed_params": non_embed,
        "non_embed_pct": 100 * non_embed / n_params,
        "elapsed_s": elapsed,
        "tok_per_s": tokens_seen / max(elapsed, 1e-6),
        "seed": args.seed,
        "ngram_thresholds": {"G1": g1_thr, "G2": g2_thr, "G3": g3_thr},
        "gates": {"G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4},
    }
    report_path.write_text(json.dumps(report, indent=2))
    print("\n=== TRAINING COMPLETE ===")
    print(json.dumps(report, indent=2))
    if getattr(args, "push_to_hub", None):
        try:
            from recamem import hub as _hub
            _hub.write_manifest(out_dir)
            _hub.push_checkpoint(args.push_to_hub, out_dir)
            print(f"Pushed checkpoint to {args.push_to_hub}")
        except Exception as e:
            print(f"Push failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
