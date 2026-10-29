import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from recamem import hub


def test_push_calls_upload_folder(tmp_path):
    (tmp_path / "w.pt").write_bytes(b"weights")
    hub.write_manifest(tmp_path)
    with patch("huggingface_hub.HfApi") as MockApi:
        instance = MockApi.return_value
        instance.create_repo.return_value = None
        instance.upload_folder.return_value = None
        hub.push_checkpoint("user/repo", tmp_path, token="tok")
        instance.create_repo.assert_called_once()
        instance.upload_folder.assert_called_once()


def test_pull_verifies_and_moves(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.pt").write_bytes(b"weights")
    hub.write_manifest(src)
    dest = tmp_path / "dest"
    with patch("huggingface_hub.snapshot_download") as sd:
        def fake_download(repo_id, revision=None, local_dir=None, token=None):
            # copy src contents to local_dir
            import shutil
            for p in src.iterdir():
                shutil.copy2(p, Path(local_dir) / p.name)
            # also manifest
        sd.side_effect = fake_download
        hub.pull_checkpoint("user/repo", dest, token=None)
        assert (dest / "w.pt").is_file()
        assert (dest / "manifest.json").is_file()
        hub.verify_checkpoint(dest)


def test_pull_aborts_on_checksum_mismatch(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "w.pt").write_bytes(b"weights")
    hub.write_manifest(src)
    # tamper manifest expectation by changing file after manifest
    (src / "w.pt").write_bytes(b"tampered")
    dest = tmp_path / "dest"
    with patch("huggingface_hub.snapshot_download") as sd:
        def fake_download(repo_id, revision=None, local_dir=None, token=None):
            import shutil
            for p in src.iterdir():
                shutil.copy2(p, Path(local_dir) / p.name)
        sd.side_effect = fake_download
        with pytest.raises(hub.ChecksumError):
            hub.pull_checkpoint("user/repo", dest, token=None)
        # dest should not contain partial? We check no w.pt verified; may exist but verify fails
