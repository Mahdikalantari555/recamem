"""Offline contracts for scripts/fetch_real_model.py.

The real download is a manual real-model tier action; these tests inject the
downloader so they never touch the network. They pin the evidence rules: a failed
or empty download is recorded as `blocked`, never as a pass, and a successful one
records the byte count and SHA-256 of the file actually on disk.
"""
import hashlib
import json

import pytest

from scripts import fetch_real_model as frm


def _read(path):
    return json.loads(path.read_text())


def test_blocked_download_is_recorded_not_passed(tmp_path):
    evidence = tmp_path / "evidence.json"

    def unreachable(repo_id, filename, local_dir):
        raise OSError("TLS/SSL connection has been closed (EOF)")

    record = frm.fetch(
        repo_id="openbmb/MiniCPM5-1B-GGUF",
        filename="MiniCPM5-1B-Q4_K_M.gguf",
        local_dir="models/minicpm5-1b",
        evidence_path=evidence,
        downloader=unreachable,
        root=tmp_path,
    )

    assert record["status"] == "blocked"
    assert record["sha256"] is None and record["bytes"] is None and record["path"] is None
    assert record["error"]["type"] == "OSError"
    assert "TLS/SSL" in record["error"]["message"]
    assert _read(evidence) == record
    assert record["tier"] == "real-model"


def test_empty_or_missing_file_is_blocked(tmp_path):
    def returns_empty(repo_id, filename, local_dir):
        target = tmp_path / local_dir / filename
        target.parent.mkdir(parents=True)
        target.write_bytes(b"")
        return str(target)

    record = frm.fetch(
        evidence_path=tmp_path / "e.json",
        downloader=returns_empty,
        root=tmp_path,
    )
    assert record["status"] == "blocked"
    assert record["error"]["type"] == "FileNotFoundError"


def test_successful_download_records_size_and_sha256(tmp_path):
    payload = b"GGUF" + bytes(range(256)) * 64  # stand-in bytes; no real weights in tests

    def writes_file(repo_id, filename, local_dir):
        target = tmp_path / local_dir / filename
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        return str(target)

    record = frm.fetch(
        filename="model.gguf",
        local_dir="models/x",
        evidence_path=tmp_path / "e.json",
        downloader=writes_file,
        root=tmp_path,
    )
    assert record["status"] == "downloaded"
    assert record["bytes"] == len(payload)
    assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["path"] == "models/x/model.gguf"
    assert record["error"] is None


def test_hub_token_is_never_written_to_evidence(tmp_path, monkeypatch):
    secret = "hf_" + "x" * 30
    monkeypatch.setenv("HF_TOKEN", secret)

    def unreachable(repo_id, filename, local_dir):
        raise OSError("offline")

    frm.fetch(evidence_path=tmp_path / "e.json", downloader=unreachable, root=tmp_path)
    assert secret not in (tmp_path / "e.json").read_text()


def test_main_exit_code_reflects_status(tmp_path, monkeypatch, capsys):
    def unreachable(repo_id, filename, local_dir):
        raise OSError("offline")

    monkeypatch.setattr(frm, "_default_downloader", unreachable)
    code = frm.main(["--evidence", str(tmp_path / "e.json")])
    assert code == 1
    assert "blocked" in capsys.readouterr().err
    assert _read(tmp_path / "e.json")["status"] == "blocked"


@pytest.mark.parametrize("name", ["DEFAULT_REPO", "DEFAULT_FILENAME"])
def test_defaults_point_at_an_official_minicpm_artifact(name):
    value = getattr(frm, name)
    assert value.startswith(("openbmb/MiniCPM", "MiniCPM"))
