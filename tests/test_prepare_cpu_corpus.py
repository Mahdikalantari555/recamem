import hashlib
import json
import numpy as np
import pytest
from scripts.prepare_cpu_corpus import prepare
from scripts.train_lm import load_memmap


def test_user_corpus_preparation_is_reproducible(tmp_path):
    train = tmp_path / 'train.txt'; train.write_text('Alpha lives here.\nA different item.\n')
    val = tmp_path / 'val.txt'; val.write_text('Another held out document.\n')
    meta = prepare(train, val, tmp_path / 'out', source='local example', license_name='user-owned')
    training, validation, disk_meta = load_memmap(tmp_path / 'out')
    assert training.tolist() == list(train.read_bytes())
    assert validation.tolist() == list(val.read_bytes())
    assert meta == disk_meta
    assert meta['train_sha256'] == hashlib.sha256(train.read_bytes()).hexdigest()
    assert meta['rights_verification'] == 'user_asserted_not_independently_verified'
    second = prepare(train, val, tmp_path / 'out2', source='local example', license_name='user-owned')
    assert meta == second
    with pytest.raises(ValueError, match='already exists'):
        prepare(train, val, tmp_path / 'out', source='x', license_name='x')


def test_leakage_and_invalid_input_fail_without_output(tmp_path):
    train = tmp_path / 'train'; train.write_text(' One   line\nOther\n')
    val = tmp_path / 'val'; val.write_text('one line\n')
    with pytest.raises(ValueError, match='overlaps'):
        prepare(train, val, tmp_path / 'out', source='x', license_name='x')
    assert not (tmp_path / 'out').exists()
    val.write_bytes(b'\xff')
    with pytest.raises(ValueError, match='UTF-8'):
        prepare(train, val, tmp_path / 'out', source='x', license_name='x')
    assert not (tmp_path / 'out').exists()
    with pytest.raises(ValueError, match='source and license'):
        prepare(train, val, tmp_path / 'out', source='', license_name='x')
