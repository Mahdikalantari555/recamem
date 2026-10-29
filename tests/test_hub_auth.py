import os
import pytest

from recamem import hub


def test_get_token_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret123")
    assert hub.get_token() == "secret123"


def test_missing_token_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    # ensure get_token returns None by patching huggingface_hub
    import recamem.hub as h
    monkeypatch.setattr(h, "get_token", lambda: None)
    with pytest.raises(hub.HubAuthError):
        hub.push_checkpoint("user/repo", tmp_path)


def test_write_and_verify_manifest(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01")
    m = hub.write_manifest(tmp_path)
    assert m.is_file()
    assert hub.verify_checkpoint(tmp_path) is True


def test_checksum_mismatch_raises(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    hub.write_manifest(tmp_path)
    (tmp_path / "a.txt").write_text("tampered")
    with pytest.raises(hub.ChecksumError):
        hub.verify_checkpoint(tmp_path)
