# Design

## Context

RecaMem already has local training (`scripts/train_lm.py`), checkpoint/resume,
and n-gram gates. This change adds a cloud path: train on Colab GPU, then
sync checkpoints via HuggingFace Hub. See proposal.md for motivation.

Constraints:
- No external API keys beyond HuggingFace token (user-provided).
- Local-first: pulling must work offline after download.
- Fail loud: auth/checksum/config mismatches raise, no silent fallback.
- Incremental only: resume from checkpoint, never retrain from scratch.

## Goals / Non-Goals

**Goals**
- Push/pull checkpoints to/from HuggingFace Hub with sha256 verification.
- Colab entry point that trains for a token budget and pushes on completion.
- Incremental resume rejecting mismatched vocab/arch.
- CLI + notebook for easy use.

**Non-Goals**
- Training orchestration/scheduling (user runs Colab manually).
- Multi-GPU or distributed training.
- Model architecture changes.
- Automatic hyperparameter search.

## Decisions

### D1: HuggingFace Hub as transport
**Choice**: Use `huggingface_hub` library (upload_folder/download_snapshot).
**Rationale**: Standard, handles auth, resumable uploads, large files.
**Alternative**: Custom HTTP — rejected, reinvents auth/resume.

### D2: Manifest with sha256
**Choice**: Each checkpoint dir contains `manifest.json` listing files +
sha256. Push uploads whole dir; pull verifies against manifest.
**Rationale**: Detects corruption; matches portable-brain pattern.
**Alternative**: Hub's built-in LFS checksums — insufficient for our
atomic abort requirement.

### D3: Incremental resume via existing checkpoint format
**Choice**: Reuse `train_lm.py` checkpoint (weights+optimizer+step+corpus
position). Add config-hash check (vocab size, d_model, n_layers, seed).
**Rationale**: No new format; consistent with local training.
**Alternative**: New cloud-specific format — rejected, duplication.

### D4: Colab entry point wraps train_lm
**Choice**: `scripts/colab_train.py` imports train_lm logic (or shells to
it) with env-var config and auto-push at end.
**Rationale**: Reuses tested code path.
**Alternative**: Separate trainer — rejected, drift risk.

### D5: Auth via env or cached login
**Choice**: `HF_TOKEN` env var first, then `huggingface_hub.get_token()`.
**Rationale**: Works in Colab (secret) and local.

### D6: Repo layout
**Choice**: One HF repo per model family; checkpoints as subfolders
`checkpoints/step_NNNN/`. Manifest at repo root lists available checkpoints.
**Rationale**: Versioning + rollback; matches "incremental" need.

## Risks / Trade-offs

- [Large uploads slow on Colab] → Use upload_folder with resume; document.
- [Token leakage in notebooks] → Use Colab secrets, never hardcode.
- [Architecture drift between local/cloud] → Shared train_lm code path.
- [Hub rate limits] → Retry with backoff in push/pull.

## Migration Plan

1. Add `recamem/hub.py` + CLI commands.
2. Add `scripts/colab_train.py`, `push_to_hub.py`, `pull_from_hub.py`.
3. Add notebook template.
4. Add tests for hub auth/push/pull/checksum/resume (mock hub).
5. Document in README + plans.

## Open Questions

- None blocking. Checkpoint retention policy (keep last N) deferred to
  implementation default (keep all, user prunes).