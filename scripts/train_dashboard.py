#!/usr/bin/env python3
"""Streamlit dashboard for LM training progress.

Run: streamlit run scripts/train_dashboard.py

Launches scripts/train_lm.py as a subprocess and visualizes:
- Loss curve (from log)
- Gate status (G0-G4)
- N-gram baselines
- Generated samples
- Checkpoint info
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

st.set_page_config(page_title="RecaMem Training", page_icon="📈", layout="wide")
st.title("📈 RecaMem LM Training Dashboard")

# Sidebar config
with st.sidebar:
    st.header("Training Config")
    core = st.selectbox("Core", ["C4", "C6", "C3"], index=0)
    d_model = st.number_input("d_model", value=192, step=32)
    n_layers = st.number_input("n_layers", value=6, step=1)
    n_heads = st.number_input("n_heads", value=4, step=1)
    d_ff = st.number_input("d_ff", value=768, step=64)
    vocab_size = st.selectbox("vocab_size", [1024, 2048], index=0)
    seq_len = st.number_input("seq_len", value=256, step=32)
    lm_tokens = st.number_input("lm_tokens", value=20_000_000, step=1_000_000)
    data_dir = st.text_input("data_dir", value="data/tinystories")
    out_dir = st.text_input("out_dir", value="runs/lm1")
    lr = st.number_input("lr", value=0.003, format="%.4f")
    batch_size = st.number_input("batch_size", value=8, step=1)
    grad_accum = st.number_input("grad_accum", value=4, step=1)
    checkpoint_every = st.number_input("checkpoint_every", value=1000, step=100)

if "proc" not in st.session_state:
    st.session_state.proc = None
if "log" not in st.session_state:
    st.session_state.log = []

col1, col2 = st.columns(2)
with col1:
    start = st.button("▶ Start Training", disabled=st.session_state.proc is not None)
with col2:
    stop = st.button("⏹ Stop", disabled=st.session_state.proc is None)

if start:
    cmd = [
        PYTHON, "-u", str(REPO_ROOT / "scripts" / "train_lm.py"),
        "--core", core,
        "--d-model", str(d_model),
        "--n-layers", str(n_layers),
        "--n-heads", str(n_heads),
        "--d-ff", str(d_ff),
        "--vocab-size", str(vocab_size),
        "--seq-len", str(seq_len),
        "--lm-tokens", str(lm_tokens),
        "--data-dir", data_dir,
        "--out-dir", out_dir,
        "--lr", str(lr),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--checkpoint-every", str(checkpoint_every),
    ]
    st.session_state.log = []
    import os as _os
    env = dict(_os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    st.session_state.proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(REPO_ROOT), env=env
    )
    st.session_state.out_dir = out_dir
    st.rerun()

if stop and st.session_state.proc is not None:
    st.session_state.proc.terminate()
    st.session_state.proc = None
    st.rerun()

# Poll process
proc = st.session_state.proc
if proc is not None:
    # Non-blocking read of available output
    import select
    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                break
            st.session_state.log.append(line.rstrip())
    except Exception:
        pass
    if proc.poll() is not None:
        # drain remaining
        try:
            for line in proc.stdout:
                st.session_state.log.append(line.rstrip())
        except Exception:
            pass
        st.session_state.proc = None
        st.success("Training finished.")

# Display
out_dir = st.session_state.get("out_dir", "runs/lm1")
out_path = (REPO_ROOT / out_dir) if not Path(out_dir).is_absolute() else Path(out_dir)

# N-gram baselines
st.subheader("N-gram Baselines")
ng_path = (REPO_ROOT / data_dir / "ngram_baselines.json") if not Path(data_dir).is_absolute() else (Path(data_dir) / "ngram_baselines.json")
if ng_path.is_file():
    ng = json.loads(ng_path.read_text())
    st.json(ng.get("thresholds", ng.get("results", {})))
else:
    st.info("No n-gram baselines found. Run scripts/eval_ngram.py first.")

# Report / gates
st.subheader("Training Report")
progress_path = out_path / "progress.json"
if progress_path.is_file():
    prog = json.loads(progress_path.read_text())
    st.metric("step", f"{prog.get('step',0)} / {prog.get('total_steps',0)}")
    st.metric("loss", f"{prog.get('loss',0):.4f}")
    st.metric("val_loss", f"{prog.get('val_loss',0):.4f}")
    st.metric("tokens_seen", f"{prog.get('tokens_seen',0):,}")
    st.metric("tok/s", f"{prog.get('toks_per_s',0):.0f}")
    elapsed = prog.get('elapsed_s', 0)
    eta = prog.get('eta_s', 0)
    st.metric("elapsed", f"{elapsed/60:.1f} min")
    st.metric("ETA", f"{eta/60:.1f} min" if eta else "—")
    st.progress(min(1.0, prog.get('step',0) / max(1, prog.get('total_steps',1))))

report_path = out_path / "report.json"
if report_path.is_file():
    report = json.loads(report_path.read_text())
    st.subheader("Final Report")
    st.json(report.get("gates", {}))
    st.metric("val_bpb", f"{report.get('val_bpb', 0):.4f}")
    st.metric("tok/s", f"{report.get('tok_per_s', 0):.0f}")
else:
    st.info("No report yet.")

# Loss log
st.subheader("Training Log (tail)")
log_lines = st.session_state.log[-40:]
st.code("\n".join(log_lines) if log_lines else "No output yet.", language="text")

# Samples
st.subheader("Generated Samples")
samples_path = out_path / "samples.txt"
if samples_path.is_file():
    st.text(samples_path.read_text()[-2000:])
else:
    st.info("No samples yet.")

if st.session_state.proc is not None:
    st.info("🔄 Training running...")
    time.sleep(2)
    st.rerun()
else:
    st.info("Idle.")
