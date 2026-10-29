# Spec Delta

## Purpose

Defines the methodology, gates, and acceptance criteria for teaching the RecaMem
core (C4 gated-delta) to generate coherent English text on CPU-only hardware,
including a debug ladder, n-gram baselines, token budgets, and scale-up protocol.

## ADDED Requirements

### Requirement: Debug ladder before long training

The system SHALL execute a six-step debug ladder before any training run exceeding
1,000 steps. Each step SHALL be completed and recorded before proceeding to the
next.

#### Scenario: Overfit single batch

- **WHEN** a single fixed batch of 8 sequences x 128 tokens is trained for 300 steps
- **THEN** the loss SHALL reach approximately zero
- **THEN** if the loss does not reach zero, the run SHALL be blocked and the failure SHALL be recorded as a code, loss, label-shift, or LR bug

#### Scenario: N-gram baselines computed

- **WHEN** the debug ladder runs
- **THEN** unigram, bigram, trigram, and 4-gram byte-level BPB SHALL be computed on the held-out val set with simple smoothing
- **THEN** all four values SHALL be recorded as the gate thresholds

#### Scenario: Reference transformer beats trigram

- **WHEN** the reference transformer (C6, linear_kv) is trained on 5M tokens with the same data and tokenizer
- **THEN** it SHALL achieve BPB below the trigram baseline
- **THEN** if it does not, the run SHALL be blocked and the failure SHALL be recorded as a data or evaluation bug, not an architecture bug

#### Scenario: Token ladder is monotonic

- **WHEN** the model is trained on 1M, 5M, 20M, and 50M tokens (from scratch or continued with correct schedule)
- **THEN** BPB SHALL be monotonically decreasing across the token counts
- **THEN** BPB vs tokens (log scale) SHALL be approximately linear
- **THEN** if BPB flattens, the run SHALL be blocked and the failure SHALL be recorded as an LR or batch-size problem

#### Scenario: Generation test passes

- **WHEN** 20 stories are generated at temperature 0.8
- **THEN** at least 15 of 20 SHALL be readable (correct grammar, no repetition loops, sensible endings)
- **THEN** distinct-2 and 4-gram repetition rate SHALL be computed and recorded

#### Scenario: Memory capability preserved

- **WHEN** MQAR-4 is run on the LM-trained model
- **THEN** accuracy SHALL remain above 0.75 (G4)
- **THEN** if it drops below 0.75, the run SHALL be blocked

### Requirement: N-gram baselines as language gates

The system SHALL compute and record unigram, bigram, trigram, and 4-gram byte-level BPB on the held-out validation set. These values SHALL serve as the G1, G2, and G3 gate thresholds.

#### Scenario: Baselines are recorded

- **WHEN** a training run produces its report
- **THEN** the report SHALL contain unigram, bigram, trigram, and 4-gram BPB values
- **THEN** the report SHALL state which gate (G1/G2/G3) the model's BPB falls below

### Requirement: Success gates for language learning

The system SHALL define and enforce five success gates. A gate SHALL be considered passed only when its condition is met.

#### Scenario: G0 code correctness

- **WHEN** a single batch is overfit to approximately zero loss
- **THEN** G0 SHALL be marked passed
- **THEN** if G0 fails, no further training SHALL proceed until the bug is fixed

#### Scenario: G1 common patterns

- **WHEN** the model's BPB is at least 25% below the unigram baseline
- **THEN** G1 SHALL be marked passed
- **THEN** the model has learned common characters and words

#### Scenario: G2 language begins

- **WHEN** the model's BPB is below the trigram baseline
- **THEN** G2 SHALL be marked passed
- **THEN** the model has moved beyond simple local statistics
- **THEN** only after G2 passes SHALL architecture, tokenizer, and interference comparisons be taken seriously

#### Scenario: G3 basic grammar

- **WHEN** the model's BPB is below the 4-gram baseline AND at least 15 of 20 generated stories are readable
- **THEN** G3 SHALL be marked passed
- **THEN** the model has basic grammar

#### Scenario: G4 recall preserved

- **WHEN** MQAR-16 accuracy is above 0.75 on the LM-trained model
- **THEN** G4 SHALL be marked passed
- **THEN** the model's recall capability is preserved

### Requirement: Token budget and time targets

The system SHALL define token-budget targets and estimated CPU training times for each model scale.

#### Scenario: Token budget is documented

- **WHEN** a training run is configured
- **THEN** the token target SHALL be at least 20 tokens per parameter (Chinchilla minimum)
- **THEN** the run report SHALL state the token target, estimated time, and actual time

#### Scenario: Scale targets

- **WHEN** training a 1M-parameter model
- **THEN** the token target SHALL be 20M tokens
- **THEN** the estimated CPU time SHALL be 0.5 to 1.7 hours

- **WHEN** training a 3M-parameter model
- **THEN** the token target SHALL be 60M tokens
- **THEN** the estimated CPU time SHALL be 5 to 15 hours

- **WHEN** training a 5M-parameter model
- **THEN** the token target SHALL be 100M tokens
- **THEN** the estimated CPU time SHALL be 14 to 42 hours

- **WHEN** training a 17M-parameter model
- **THEN** the token target SHALL be 340M tokens
- **THEN** GPU training SHALL be recommended (estimated 7 to 20 days on CPU)

### Requirement: Model configuration for sub-5M learning

The system SHALL use a model configuration optimized for learning language at sub-5M parameter scale.

#### Scenario: Embedding budget

- **WHEN** a model has fewer than 5M parameters
- **THEN** the vocabulary size SHALL be 1024 or 2048 (not 4096)
- **THEN** embeddings SHALL be tied (same matrix for input and output)
- **THEN** the non-embedding parameter count SHALL be at least 50% of total parameters

#### Scenario: Architecture parameters

- **WHEN** a 1M-parameter model is configured
- **THEN** d_model SHALL be approximately 192
- **THEN** n_layers SHALL be 6 to 8
- **THEN** n_heads SHALL be 4
- **THEN** context length SHALL be 256

#### Scenario: Initialization

- **WHEN** model weights are initialized
- **THEN** normal init with std approximately 0.02 SHALL be used
- **THEN** output projections SHALL be scaled by 1/sqrt(2L)
- **THEN** the forgetting gate (alpha) SHALL be initialized so sigmoid is approximately 0.9 to 0.99

### Requirement: Training hyperparameters

The system SHALL use training hyperparameters optimized for small-model language learning.

#### Scenario: Optimizer configuration

- **WHEN** training begins
- **THEN** AdamW SHALL be used with betas (0.9, 0.95)
- **THEN** weight decay SHALL be 0.1 (not applied to norm or bias)
- **THEN** gradient clipping SHALL be 1.0

#### Scenario: Learning rate schedule

- **WHEN** training begins
- **THEN** LR peak SHALL be 1e-3 to 3e-3
- **THEN** warmup SHALL be 2 to 5 percent of total steps
- **THEN** decay SHALL be cosine or linear to 10 percent or zero
- **THEN** the schedule SHALL be tied to total token count, not step count

#### Scenario: Batch and precision

- **WHEN** training begins
- **THEN** effective batch size SHALL be 16k to 64k tokens per step (via gradient accumulation)
- **THEN** dropout SHALL be 0 (none)
- **THEN** bf16 autocast SHALL be used if CPU supports AMX or AVX512-BF16, else fp32
- **THEN** torch.set_num_threads SHALL be set to physical core count

### Requirement: Scale-up protocol

The system SHALL follow a scale-up protocol after G2 passes.

#### Scenario: Scale-up order

- **WHEN** G2 passes on the 1M model
- **THEN** training SHALL proceed to 3M model (60M tokens)
- **THEN** after 3M succeeds, training SHALL proceed to 5M model (100M tokens)
- **THEN** after 5M succeeds, 17M model SHALL be trained (340M tokens, GPU recommended)

#### Scenario: Decision heads after LM

- **WHEN** the LM reaches G2
- **THEN** decision readout and router heads MAY be added
- **THEN** they SHALL be trained with curriculum sampling
- **THEN** JEPA objective SHALL remain off until LM is solid

### Requirement: Code changes for language learning

The system SHALL implement specific code changes to support language learning.

#### Scenario: Token-based training

- **WHEN** the tournament or training script is invoked
- **THEN** `--lm-tokens` SHALL be available (replacing `--lm-steps`)
- **THEN** every log entry SHALL include tokens seen, tok/s, and current LR

#### Scenario: Checkpoint and resume

- **WHEN** training exceeds 1,000 steps
- **THEN** checkpoints SHALL be written every 1,000 steps
- **THEN** each checkpoint SHALL contain model weights, optimizer state, step count, and corpus position
- **THEN** resume SHALL continue from the last checkpoint without reprocessing earlier bytes

#### Scenario: Config reporting

- **WHEN** a training run produces its report
- **THEN** the report SHALL print d_model, n_layers, V, embedding param count, and non-embedding param count

#### Scenario: N-gram baselines in Stage 0

- **WHEN** Stage 0 of the tournament runs
- **THEN** unigram, bigram, trigram, and 4-gram byte-level BPB SHALL be computed and recorded