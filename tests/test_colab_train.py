import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_colab_train_small(tmp_path):
    out = tmp_path / "run"
    cmd = [sys.executable, str(ROOT / "scripts" / "colab_train.py"),
           "--lm-tokens", "50000",
           "--data-dir", str(ROOT / "data" / "tinystories"),
           "--out-dir", str(out),
           "--checkpoint-every", "10",
           "--batch-size", "2",
           "--grad-accum", "1"]
    rc = subprocess.run(cmd, cwd=str(ROOT))
    assert rc.returncode == 0
    assert (out / "report.json").is_file()
