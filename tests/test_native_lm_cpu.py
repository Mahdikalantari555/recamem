import pytest
import torch
from recamem.native_lm import NativeLMBackend
from src.model.cores.registry import build_candidate_lm


def test_native_checkpoint_and_generation(tmp_path):
    cfg = dict(core_id='C7', vocab_size=256, d_model=8, n_layers=1, n_heads=2, d_ff=16, tokenizer='utf8_bytes_256')
    model = build_candidate_lm(cfg)
    path = tmp_path / 'model.pt'
    torch.save({'model_state': model.state_dict(), 'meta': {'config': cfg}}, path)
    backend = NativeLMBackend(path, context_window=32, max_output_bytes=3)
    assert isinstance(backend.generate('hello', max_tokens=2), str)
    assert backend.count_tokens('é') == 2
    with pytest.raises(FileNotFoundError): NativeLMBackend(tmp_path / 'missing.pt')
    torch.save({'model_state': model.state_dict(), 'meta': {'config': {}}}, path)
    with pytest.raises(ValueError): NativeLMBackend(path)
