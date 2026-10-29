# Cloud Training Plan

See docs/cloud-training.md and openspec/changes/cloud-training-huggingface.

Usage examples:
- Fresh run: `python scripts/colab_train.py --lm-tokens 20000000 --data-dir data/tinystories --out-dir runs/cloud1 --push-to-hub user/recamem`
- Resume: `python scripts/colab_train.py --lm-tokens 40000000 --resume-from user/recamem --push-to-hub user/recamem`