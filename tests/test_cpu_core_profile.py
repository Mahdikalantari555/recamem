import json
from scripts.profile_cpu_cores import main


def test_profile_reports_parity_and_real_provenance(tmp_path):
    path = tmp_path / 'profile.json'
    assert main(['--cores', 'C4', 'C7', '--d-model', '8', '--n-heads', '2',
                 '--d-ff', '16', '--seq-len', '4', '--warmup', '0',
                 '--iterations', '1', '--threads', '1', '--output', str(path)]) == 0
    report = json.loads(path.read_text())
    assert 'synthetic' in report['dataset']
    assert report['threads'] == 1 and report['seq_len'] == 4
    for core in ('C4', 'C7'):
        row = report['cores'][core]
        assert row['parameters'] > 0 and row['finite_gradients']
        assert row['logit_max_abs_error'] < 1e-4
        assert all(value < 1e-4 for value in row['state_max_abs_error'].values())
        assert row['forward']['samples'] == 1
        assert row['profiler_top_ops']
