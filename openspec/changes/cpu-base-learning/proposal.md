# Proposal: CPU base learning and incremental teaching

## Why
The user wants a small, locally trainable base knowledge model that can be taught incrementally, without GGUF or llama.cpp. Tiny datasets can demonstrate narrow language learning but do not guarantee general understanding or emergent capabilities.

## What Changes
Extend the existing `recamem-language-learning` training work with a reproducible CPU-only data and evaluation pipeline, a native PyTorch checkpoint and generation path, and an incremental teaching workflow using RecaMem memory. Provide a small openly licensed or self-authored fixture for offline smoke tests and an explicit, provenance-verified procedure for acquiring a larger language corpus. Enable optional Muir when the separate `cpu-muir-core` change is ready, with C4 as baseline.

## Capabilities
- `training/cpu-base-learning`: data, native CPU training, evaluation, incremental teaching and evidence.

## Non-goals
No promise of general understanding, emergence, useful open-domain knowledge, or specified training time on unknown hardware. No new GGUF dependency or required pretrained model. Do not remove existing optional GGUF integrations as part of this change.

## Impact
Reuse `scripts/train_lm.py`, corpus preparation, tokenizer, evaluation and existing memory APIs rather than duplicate the existing language-learning proposal. Data, checkpoints and reports require provenance; large corpora and weights stay out of Git.

## Continuous learning alignment

The target is automatic model, tokenizer and bounded training-plan selection, with ongoing background incremental learning; see `autonomous-incremental-learning`. The manual smoke run is not continuous learning.
