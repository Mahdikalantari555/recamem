# RecaMem migration & rollback runbook

## Migration (`recamem migrate`)

Copy legacy `droids/` profiles into the recamem home:

```sh
PYTHONPATH=. python -m recamem.cli migrate droids/ --dest ~/.recamem/agents
```

- Profiles already present at the destination are reported as `already_migrated`
  and left untouched.
- A per-profile writer lock prevents concurrent migration of the same profile.
- Verify recall parity with the pre-change profile before declaring success.

## Rollback

1. Stop any running `recamem serve` / `recamem ui` process.
2. Restore the previous `droids/` directory from backup or git.
3. Re-run `recamem doctor` to confirm anchor + storage readiness.
4. If a profile was partially migrated, remove the destination directory and
   re-run migrate from a clean backup.

## Blocked items (no llama.cpp / network)

The following tasks require `llama-cpp-python` and/or real model artifacts and
are reported as `blocked`, not passed:

- post-merge 3.4 (KV prefill), 3.7 (GGUF streaming/cache), 8.1 (GGUF/dual-mode
  integration), 9.2 (real ONNX/GGUF/RTU grounding)
- post-merge 7.10 (matched-budget 1B+TTC vs 8B evaluation)

Install `llama-cpp-python` and a local GGUF artifact to unblock; do not weaken
the offline gate.
