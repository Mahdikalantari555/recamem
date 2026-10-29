import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_resume_continues(tmp_path):
    out1 = tmp_path / "run1"
    cmd1 = [sys.executable, str(ROOT / "scripts" / "train_lm.py"),
            "--lm-tokens", "40000",
            "--data-dir", str(ROOT / "data" / "tinystories"),
            "--out-dir", str(out1),
            "--checkpoint-every", "2",
            "--batch-size", "2",
            "--grad-accum", "1",
            "--seq-len", "64"]
    r1 = subprocess.run(cmd1, cwd=str(ROOT))
    assert r1.returncode == 0
    ckpt = out1 / "checkpoint.pt"
    assert ckpt.is_file()

    out2 = tmp_path / "run2"
    cmd2 = [sys.executable, str(ROOT / "scripts" / "train_lm.py"),
            "--lm-tokens", "80000",
            "--data-dir", str(ROOT / "data" / "tinystories"),
            "--out-dir", str(out2),
            "--checkpoint-every", "2",
            "--batch-size", "2",
            "--grad-accum", "1",
            "--seq-len", "64",
            "--resume", str(ckpt)]
    r2 = subprocess.run(cmd2, cwd=str(ROOT))
    assert r2.returncode == 0


def test_resume_rejects_mismatched_config(tmp_path):
    out1 = tmp_path / "run1"
    cmd1 = [sys.executable, str(ROOT / "scripts" / "train_lm.py"),
            "--lm-tokens", "20000",
            "--data-dir", str(ROOT / "data" / "tinystories"),
            "--out-dir", str(out1),
            "--checkpoint-every", "1",
            "--batch-size", "2",
            "--grad-accum", "1",
            "--seq-len", "64",
            "--d-model", "64"]
    r1 = subprocess.run(cmd1, cwd=str(ROOT))
    assert r1.returncode == 0
    ckpt = out1 / "checkpoint.pt"

    out2 = tmp_path / "run2"
    cmd2 = [sys.executable, str(ROOT / "scripts" / "train_lm.py"),
            "--lm-tokens", "40000",
            "--data-dir", str(ROOT / "data" / "tinystories"),
            "--out-dir", str(out2),
            "--checkpoint-every", "1",
            "--batch-size", "2",
            "--grad-accum", "1",
            "--seq-len", "64",
            "--d-model", "128",  # mismatch
            "--resume", str(ckpt)]
    r2 = subprocess.run(cmd2, cwd=str(ROOT))
    assert r2.returncode != 0
