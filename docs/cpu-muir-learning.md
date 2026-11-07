# Experimental CPU Muir and native LM

C7 (`muir`) is an opt-in sequential PyTorch implementation of the update in
[Muir](https://arxiv.org/html/2611.08376v1), equation 8. The gates use a fused
down-projection and individual up-projections; ranks scale to small models.
It uses per-head normalization, causal convolution and explicit streaming state.
It is not the TPU parallel kernel or an exact reproduction of its initialization,
and no speed or quality improvement is established. C4 remains the default.
State reporting assumes FP32. Python token loops can make C7 slow on CPU.

To test the plumbing with the self-authored MIT smoke fixture (not language
training evidence):

```sh
pip install -e .
PYTHONPATH=. python scripts/prepare_cpu_smoke.py --out-dir runs/cpu-smoke-data
PYTHONPATH=. python scripts/train_lm.py --core C7 --d-model 16 --n-layers 1 \
  --n-heads 2 --d-ff 32 --vocab-size 256 --seq-len 16 --batch-size 1 \
  --grad-accum 1 --lm-tokens 64 --threads 1 --data-dir runs/cpu-smoke-data \
  --out-dir runs/cpu-smoke-run
```

Load the resulting **trusted local** checkpoint with
`recamem.native_lm.NativeLMBackend('runs/cpu-smoke-run/checkpoint.pt')` and
call `generate(prompt)`. The loader currently accepts byte-tokenized native
training checkpoints only; older checkpoints lacking architecture/tokenizer
metadata are rejected. Only load trusted checkpoints, since other PyTorch
versions and checkpoint formats may use pickle. Tiny smoke training produces
an essentially untrained model, not a useful conversational assistant.

For a meaningful experiment, the existing TinyStories preparation/training
work remains available, but rights verification, a pinned corpus, data-split
checks, fair CPU comparisons and native BPE loading have not been completed
for this workflow. No substantive corpus or trained model is shipped here.
Reports have missing baseline gates when no baseline is supplied. New checkpoints restore the CPU and NumPy random generator states on resume.
Older checkpoints lack that information and cannot resume bit-exactly; changing
the total training budget also changes the learning-rate schedule. The proposed incremental teaching, independent recall
and retention tests, and Svelte workspace remain work in progress. Facts in
RecaMem memory must not be described as updates to the model's weights.

## Learning Studio

Build the companion with `npm --prefix companion ci && npm --prefix companion
run build`, then run `PYTHONPATH=. python -m recamem.cli companion --no-browser`.
The CPU workspace can start and cancel an explicit self-authored smoke run,
show its real job metrics, and try its native checkpoint in Evaluation. Memory
and companion actions use the existing engine and require a configured real
anchor (or a mock explicitly selected for testing). The default local server
is loopback-only. Selecting a checkpoint in the legacy advanced view records
provenance only; it does not attach it to chat.

## Preparing your own local text

For locally held text you are authorized to use, provide separate training and
validation files with one document per line. This does not download or verify a
corpus or license. It rejects identical normalized lines across splits, but does
not detect near-duplicates or semantic leakage. It limits each input to 64 MiB
and refuses to overwrite an existing output directory.

```sh
PYTHONPATH=. python scripts/prepare_cpu_corpus.py \
  --train /path/to/train.txt --val /path/to/val.txt \
  --source 'description and version of your data' --license 'your verified rights' \
  --out-dir runs/my-corpus
```

The resulting `train.bin`, `val.bin` and `meta.json` use the existing byte
training format. Train explicitly with `scripts/train_lm.py --data-dir
runs/my-corpus --vocab-size 256` and appropriate CPU settings; this is not yet
connected to automatic background learning or the Studio. Independently verify
rights, representativeness and near-duplicate leakage before interpreting any
results; the metadata records rights as user-asserted, not independently verified.

## Reproducible held-out comparison

For a trusted native byte checkpoint and a prepared dataset with matching hashes:

```sh
PYTHONPATH=. python scripts/evaluate_cpu_lm.py \
  --checkpoint runs/my-run/checkpoint.pt --data-dir runs/my-corpus \
  --context 64 --max-predictions 4096 --threads 1 --output runs/heldout-eval.json
```

The report measures the same contiguous validation prefix with a model, an
add-one-smoothed byte unigram and an add-one-smoothed byte bigram. Model context
is reset at each disjoint window; change the window size only when recording the
comparison. It verifies token counts and hashes against the supplied metadata,
but cannot prove the checkpoint never trained on validation data or that rights
are correct. Run only on trusted local checkpoints. This is not a retention,
composition or checkpoint-promotion test. The output path must not already exist.

As a pipeline check only, a 64-token C4 run on the self-authored smoke fixture
(seed 1337, width 8, one layer, two heads, context 16, 1 CPU thread, 123 held-out
bytes) measured 8.47 model bits/byte, 4.75 unigram and 6.14 bigram. This is a
tiny fixture and a barely trained model, not a representative quality result.
Full local JSON provenance is `/home/user/recamem-heldout-eval.json` in this
sandbox. Reproduce with your own checkpoint and dataset; sandbox RAM was 3.85 GiB,
not the 16 GB target.

### Experimental memory-backed sampling

In Evaluation, a completed smoke job offers **Try with memory**, in addition to
the existing memory-free sample. This explicit experimental path loads the job's
checkpoint and passes fitting recalled text to it; it abstains rather than
sampling if no evidence fits. It reports the evidence and whether memory was
used. A working real memory anchor is required, and a smoke model is still not a
useful assistant. Retrieved text is untrusted, prompt formatting is not a prompt
injection defense, and a generated answer is not verified by its evidence.
Semantic evidence may contain linked fact IDs rather than a resolved source.
This does not change model weights or replace normal chat, CLI or MCP paths.
