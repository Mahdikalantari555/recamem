import json
import math

import pytest
import torch

from scripts.evaluate_cpu_lm import evaluate, main
from scripts.prepare_cpu_corpus import prepare
from src.model.cores.registry import build_candidate_lm


def setup_fixture(tmp_path):
    train = tmp_path / 'train.txt'; train.write_text('abcabcabc\n')
    val = tmp_path / 'val.txt'; val.write_text('abcdxyz\n')
    data = tmp_path / 'data'
    prepare(train, val, data, source='test fixture', license_name='test')
    cfg = dict(core_id='C4', vocab_size=256, d_model=8, n_layers=1, n_heads=2,
               d_ff=16, tokenizer='utf8_bytes_256')
    torch.manual_seed(7)
    model = build_candidate_lm(cfg)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model_state': model.state_dict(), 'step': 1, 'tokens_seen': 10,
                'meta': {'config': cfg}}, checkpoint)
    return data, checkpoint


def test_evaluate_deterministic_cpu_and_baselines(tmp_path):
    data, checkpoint = setup_fixture(tmp_path)
    report = evaluate(checkpoint, data, context=3, max_predictions=5)
    again = evaluate(checkpoint, data, context=3, max_predictions=5)
    assert report['model'] == again['model']
    assert report['baselines'] == again['baselines']
    assert report['predicted_bytes'] == 5 and report['device'] == 'cpu'
    assert report['checkpoint_training_progress']['tokens_seen'] == 10
    # Train bytes contain a, b and c three times each and one newline.
    # Validation targets for n=5 are b, c, d, x, y; Laplace alpha=1.
    expected = -(2 * math.log(4 / 266) + 3 * math.log(1 / 266)) / 5 / math.log(2)
    assert report['baselines']['unigram_add_one']['bits_per_byte'] == pytest.approx(expected)
    for metric in [report['model'], *report['baselines'].values()]:
        assert math.isfinite(metric['bits_per_byte']) and metric['bits_per_byte'] > 0
    assert 'not independently verified' in report['warning']
    output = tmp_path / 'report.json'
    assert main(['--checkpoint', str(checkpoint), '--data-dir', str(data), '--output', str(output)]) == 0
    assert json.loads(output.read_text())['predicted_bytes'] == 7
    with pytest.raises(SystemExit):
        main(['--checkpoint', str(checkpoint), '--data-dir', str(data), '--output', str(output)])


def test_evaluate_rejects_corrupt_data_and_bad_limits(tmp_path):
    data, checkpoint = setup_fixture(tmp_path)
    with pytest.raises(ValueError, match='context'):
        evaluate(checkpoint, data, context=0)
    (data / 'val.bin').write_bytes(b'\x01\x00\x02\x00')
    with pytest.raises(ValueError, match='does not match'):
        evaluate(checkpoint, data)
