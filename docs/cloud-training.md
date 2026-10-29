# Cloud Training & HuggingFace Sync

## Overview
Train RecaMem models on a GPU (Colab) and sync checkpoints via HuggingFace Hub.

## Setup
```bash
pip install -e ".[cloud]"
export HF_TOKEN=hf_...   # or: huggingface-cli login
```

## Train on Colab
Open `notebooks/recamem_colab.ipynb` in Colab. Or run directly:
```bash
python scripts/colab_train.py --lm-tokens 20000000 --push-to-hub yourname/recamem-ckpt
```

## Push / Pull manually
```bash
recamem push --repo yourname/recamem-ckpt --dir runs/lm1
recamem pull --repo yourname/recamem-ckpt --dir runs/lm1_restored
```

## Incremental resume
```bash
recamem train-cloud --lm-tokens 40000000 --resume-from yourname/recamem-ckpt --push-to-hub yourname/recamem-ckpt
```
Resume rejects mismatched vocab size / architecture / seed.

## Verification
Each checkpoint dir contains `manifest.json` with sha256 per file. Pull aborts on mismatch.