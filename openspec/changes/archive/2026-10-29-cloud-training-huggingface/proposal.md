# Proposal

## Why

RecaMem models and plastic memory are trained locally on CPU, which limits
iteration speed and model scale. There is no way to train on a GPU cloud
(Colab) and bring the resulting weights back into a local profile, and no
way to share or version checkpoints via HuggingFace Hub. This change
introduces a cloud-training toolkit with HuggingFace push/pull so models
can be trained incrementally on Colab and downloaded locally.

## What Changes

- Add a `training/cloud-huggingface` capability defining the contract for
  cloud training, checkpoint sync, and HuggingFace integration.
- New scripts: `scripts/colab_train.py` (Colab entry point with resume),
  `scripts/push_to_hub.py`, `scripts/pull_from_hub.py`.
- New module `recamem/hub.py` wrapping HuggingFace Hub upload/download with
  auth via token, sha256 verification, and incremental checkpointing.
- CLI commands `recamem push` / `recamem pull` / `recamem train-cloud`.
- Notebook template `notebooks/recamem_colab.ipynb` for one-click Colab runs.
- Support incremental training: resume from last checkpoint, append training
  steps without retraining from scratch, never change vocab/arch after start.

## Capabilities

### New Capabilities
- `training/cloud-huggingface`: Cloud training + HuggingFace push/pull
  toolkit contract (auth, push, pull, resume, verification, incremental).

### Modified Capabilities
- (none)

## Impact

- Affected: `recamem/cli.py`, new `recamem/hub.py`, new scripts, new
  notebook. Dependencies: `huggingface_hub` (optional extra). Auth via
  `HF_TOKEN` env var or `huggingface-cli login`. No breaking changes to
  existing local training.