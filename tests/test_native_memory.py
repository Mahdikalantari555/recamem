import pytest
import torch

from recamem.backend_contracts import QueryBudgetError
from recamem.engine import NO_INFORMATION_MESSAGE
from recamem.native_lm import NativeLMBackend
from recamem.native_memory import NativeMemorySession
from src.model.cores.registry import build_candidate_lm


class Backend:
    tokenizer_kind = 'bytes'
    def __init__(self, context=256):
        self.context = context
        self.calls = []
    def context_window(self): return self.context
    def output_reserve(self): return 16
    def count_tokens(self, text): return len(text.encode('utf-8'))
    def generate(self, prompt, max_tokens=16):
        self.calls.append((prompt, max_tokens))
        return 'experimental answer'


class Memory:
    def __init__(self, hits): self.hits = hits
    def recall(self, query, top_k=3): return self.hits[:top_k]


def test_native_memory_tracks_only_included_evidence():
    hits = [{'id': 3, 'text': 'A fact.', 'source': 'user_text', 'source_url': 'https://example.org'},
            {'id': 4, 'text': 'X' * 500, 'source': 'user_text'}]
    backend = Backend()
    answer = NativeMemorySession(Memory(hits), backend).answer('Question?', max_tokens=8)
    assert answer['memory_used'] and answer['status'] == 'generated'
    assert not answer['model_weights_updated']
    assert answer['evidence'] == [hits[0]]
    assert answer['prompt_budget']['trimmed_facts'] == 1
    assert 'A fact.' in backend.calls[0][0] and backend.calls[0][0].endswith('Question?')


def test_no_evidence_or_no_space_never_generates():
    backend = Backend(context=48)
    session = NativeMemorySession(Memory([]), backend)
    result = session.answer('Unknown?', max_tokens=8)
    assert result['response'] == NO_INFORMATION_MESSAGE and not backend.calls
    session = NativeMemorySession(Memory([{'text': 'long ' * 100, 'source': 'user'}]), backend)
    result = session.answer('Unknown?', max_tokens=8)
    assert result['status'] == 'no_information' and not backend.calls
    with pytest.raises(QueryBudgetError):
        session.answer('Q' * 200, max_tokens=8)
    with pytest.raises(ValueError): session.answer('x', max_tokens=True)


def test_actual_native_model_receives_real_recalled_memory(tmp_path, monkeypatch):
    from recamem.manager import DroidManager
    monkeypatch.setenv('RECAMEM_ANCHOR', 'mock')
    engine = DroidManager(base_dir=str(tmp_path / 'droids')).get_droid('demo')
    engine.teach('The calibration constant is 0.98.', source='user_text')
    cfg = dict(core_id='C4', vocab_size=256, d_model=8, n_layers=1, n_heads=2,
               d_ff=16, tokenizer='utf8_bytes_256')
    model = build_candidate_lm(cfg)
    path = tmp_path / 'checkpoint.pt'
    torch.save({'model_state': model.state_dict(), 'meta': {'config': cfg}}, path)
    backend = NativeLMBackend(path, context_window=256, max_output_bytes=4)
    result = NativeMemorySession(engine, backend).answer('What is the calibration constant?', max_tokens=4)
    assert result['status'] == 'generated'
    assert result['evidence'][0]['text'] == 'The calibration constant is 0.98.'
    assert result['evidence'][0]['source'] == 'user_text'
    assert isinstance(result['response'], str)
    assert not result['model_weights_updated']


def test_memory_correction_evidence_survives_profile_reload(tmp_path, monkeypatch):
    from recamem.engine import DroidEngine
    monkeypatch.setenv('RECAMEM_ANCHOR', 'mock')
    engine = DroidEngine(name='memory-test')
    engine.teach('The calibration constant is 0.98.', source='user_text')
    hits = engine.recall('What is the calibration constant?')
    assert hits
    correction = engine.correct(hits[0]['id'], 'The calibration constant is 0.99.')
    assert correction['status'] == 'corrected'
    profile = tmp_path / 'profile'
    engine.save_profile(str(profile))
    restored = DroidEngine(name='memory-test')
    assert restored.load_profile(str(profile))
    backend = Backend()
    result = NativeMemorySession(restored, backend).answer('What is the calibration constant?', max_tokens=8)
    assert result['status'] == 'generated'
    assert result['evidence'][0]['text'] == 'The calibration constant is 0.99.'
    assert result['evidence'][0]['retrieval_store'] == 'semantic'
    assert correction['new_id'] in result['evidence'][0]['source_cluster_ids']
    assert restored.knowledge.get_fact(correction['new_id'])['source'] == 'correction'
    assert '0.99' in backend.calls[0][0]
    assert not result['model_weights_updated']
