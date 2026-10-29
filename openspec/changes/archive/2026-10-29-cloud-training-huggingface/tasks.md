# Tasks

## 1. Setup & Dependencies

- [x] 1.1 Add `huggingface_hub` to optional extras in pyproject.toml and verify `pip install -e ".[cloud]"` succeeds
- [x] 1.2 Create `recamem/hub.py` skeleton (auth, push, pull, verify) and verify import succeeds
- [x] 1.3 Add `recamem push` / `recamem pull` / `recamem train-cloud` CLI commands and verify `recamem --help` lists them

## 2. Hub Module (recamem/hub.py)

- [x] 2.1 Implement `get_token()` (env HF_TOKEN then cached login) and verify missing token raises clear auth error
- [x] 2.2 Implement `push_checkpoint(repo_id, local_dir, token=None)` with repo creation and atomic commit; verify upload via mocked hub
- [x] 2.3 Implement `pull_checkpoint(repo_id, local_dir, revision=None, token=None)` with sha256 verification; verify mismatch aborts
- [x] 2.4 Implement `write_manifest(local_dir)` generating sha256 for all files; verify manifest matches files
- [x] 2.5 Implement `verify_checkpoint(local_dir)` comparing files to manifest; verify returns True on match, raises on mismatch

## 3. Scripts

- [x] 3.1 Create `scripts/push_to_hub.py` CLI wrapper calling recamem.hub.push_checkpoint; verify `--help` and dry run
- [x] 3.2 Create `scripts/pull_from_hub.py` CLI wrapper calling recamem.hub.pull_checkpoint; verify `--help` and dry run
- [x] 3.3 Create `scripts/colab_train.py` wrapping train_lm with env-var config and auto-push; verify runs 100 steps locally
- [x] 3.4 Add config-hash check to checkpoint load (vocab size, d_model, n_layers, seed); verify mismatch raises

## 4. Incremental Training Integration

- [x] 4.1 Add `--resume-from <dir>` to train_lm.py loading checkpoint + verifying config hash; verify resume continues step counter
- [x] 4.2 Add `--push-to-hub <repo>` to train_lm.py triggering push on completion; verify push called after training
- [x] 4.3 Add `--checkpoint-every` already present; ensure pushed checkpoints include manifest; verify manifest present

## 5. Notebook Template

- [x] 5.1 Create `notebooks/recamem_colab.ipynb` with install, clone, config, train, push cells; verify opens in Jupyter/Colab
- [x] 5.2 Document Colab secrets usage (HF_TOKEN) in notebook; verify no hardcoded tokens

## 6. Tests

- [x] 6.1 Add `tests/test_hub_auth.py` for token resolution and missing-token error
- [x] 6.2 Add `tests/test_hub_push_pull.py` with mocked huggingface_hub verifying push/pull/checksum
- [x] 6.3 Add `tests/test_colab_train.py` running 100 steps locally without push
- [x] 6.4 Add `tests/test_incremental_resume.py` verifying resume continues and rejects mismatched config

## 7. Documentation

- [x] 7.1 Update README with cloud training section and Colab badge
- [x] 7.2 Add `plans/cloud-training.md` with usage examples
- [x] 7.3 Document HF token setup in `docs/cloud-training.md`

## 8. Integration Check

- [x] 8.1 Run full test suite and verify no new failures
- [x] 8.2 Run offline gate and verify passes