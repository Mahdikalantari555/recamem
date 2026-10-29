"""HuggingFace Hub sync for RecaMem checkpoints.

Local-first: push/pull checkpoints to/from a HF repo with sha256 verification.
Auth via HF_TOKEN env var or prior `huggingface-cli login`.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


class HubAuthError(RuntimeError):
    pass


class ChecksumError(RuntimeError):
    pass


class ConfigMismatchError(ValueError):
    pass


MANIFEST_NAME = "manifest.json"


def get_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token as _gt

        return _gt()
    except Exception:
        return None


def _require_token() -> str:
    t = get_token()
    if not t:
        raise HubAuthError(
            "No HuggingFace token found. Set HF_TOKEN env var or run "
            "`huggingface-cli login`."
        )
    return t


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(local_dir: Path) -> Path:
    local_dir = Path(local_dir)
    files = []
    for p in sorted(local_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            rel = p.relative_to(local_dir).as_posix()
            files.append({"path": rel, "sha256": sha256_file(p)})
    manifest = {"files": files}
    out = local_dir / MANIFEST_NAME
    out.write_text(json.dumps(manifest, indent=2))
    return out


def verify_checkpoint(local_dir: Path) -> bool:
    local_dir = Path(local_dir)
    manifest_path = local_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {MANIFEST_NAME} in {local_dir}")
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest.get("files", []):
        rel = entry["path"]
        p = local_dir / rel
        if not p.is_file():
            raise ChecksumError(f"Missing file: {rel}")
        actual = sha256_file(p)
        if actual != entry["sha256"]:
            raise ChecksumError(
                f"Checksum mismatch for {rel}: expected {entry['sha256']}, got {actual}"
            )
    return True


def push_checkpoint(repo_id: str, local_dir: Path, token: Optional[str] = None,
                   private: bool = False, commit_message: str = "Upload checkpoint") -> str:
    token = token or _require_token()
    local_dir = Path(local_dir)
    if not (local_dir / MANIFEST_NAME).is_file():
        write_manifest(local_dir)
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    except Exception:
        pass
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return repo_id


def pull_checkpoint(repo_id: str, local_dir: Path, revision: Optional[str] = None,
                   token: Optional[str] = None) -> Path:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    # download to temp then verify, then move
    import tempfile
    import shutil

    tmp = Path(tempfile.mkdtemp(prefix="recamem_pull_"))
    try:
        from huggingface_hub import snapshot_download

        tok = token  # None ok for public
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(tmp),
            token=tok,
        )
        # verify
        verify_checkpoint(tmp)
        # move contents
        for child in tmp.iterdir():
            dest = local_dir / child.name
            if child.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(child, dest)
            else:
                shutil.copy2(child, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return local_dir


def config_hash(cfg: dict) -> str:
    import hashlib as _h

    keys = ["vocab_size", "d_model", "n_layers", "n_heads", "d_ff", "seed"]
    payload = {k: cfg.get(k) for k in keys}
    return _h.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def check_config_match(ckpt_cfg: dict, requested_cfg: dict) -> None:
    if config_hash(ckpt_cfg) != config_hash(requested_cfg):
        raise ConfigMismatchError(
            f"Checkpoint config {ckpt_cfg} does not match requested {requested_cfg}"
        )
