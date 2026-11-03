"""Test overfitting a single batch with candidate LM core (Task 4.1)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.model.cores.registry import build_candidate_lm


def test_overfit_single_batch():
    """A small C4 LM model overfits a single batch to near-zero loss in ~150 steps."""
    torch.manual_seed(1337)
    cfg = {
        "core_id": "C4",
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "vocab_size": 128,
        "seq_len": 16,
    }
    model = build_candidate_lm(cfg, tied_embeddings=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    x = torch.randint(0, 128, (2, 16))
    y = torch.randint(0, 128, (2, 16))

    for step in range(150):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits.view(-1, 128), y.view(-1))
        loss.backward()
        optimizer.step()

    assert loss.item() < 0.05, f"Expected loss < 0.05 on single batch overfit, got {loss.item()}"
