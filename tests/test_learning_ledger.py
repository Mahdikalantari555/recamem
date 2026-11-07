import hashlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from recamem.learning import LearningLedger, LearningPolicyError


def test_consent_idempotency_revocation_and_restart(tmp_path):
    ledger = LearningLedger(tmp_path, 'first')
    digest = hashlib.sha256(b'evidence').hexdigest()
    with pytest.raises(LearningPolicyError, match='not approved'):
        ledger.enqueue('event-1', 'explicit-teach', 'fact:1', digest)
    ledger.approve_source('explicit-teach', True)
    assert ledger.enqueue('event-1', 'explicit-teach', 'fact:1', digest)
    assert not ledger.enqueue('event-1', 'explicit-teach', 'fact:1', digest)
    with pytest.raises(LearningPolicyError, match='different evidence'):
        ledger.enqueue('event-1', 'explicit-teach', 'fact:2', digest)
    ledger.approve_source('explicit-teach', False)
    ledger.approve_source('explicit-teach', True)
    assert ledger.status()['excluded'] == 1
    assert ledger.status()['pending'] == 0
    assert LearningLedger(tmp_path, 'first').status()['excluded'] == 1
    assert LearningLedger(tmp_path, 'other').status()['excluded'] == 0
    assert ledger.status()['training_active'] is False


def test_concurrent_idempotent_enqueue(tmp_path):
    ledger = LearningLedger(tmp_path, 'first')
    ledger.approve_source('source', True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ledger.enqueue('id', 'source', 'fact:1', 'a' * 64), range(12)))
    assert sum(results) == 1
    assert ledger.status()['pending'] == 1


def test_configuration_is_persisted_and_validated(tmp_path):
    ledger = LearningLedger(tmp_path, 'first')
    assert ledger.status()['status'] == 'blocked'
    assert ledger.configure(enabled=False, threads=2)['status'] == 'paused'
    assert LearningLedger(tmp_path, 'first').status()['threads'] == 2
    for enabled, threads in [(1, 1), (True, True), (True, 0), (True, 33)]:
        with pytest.raises(LearningPolicyError):
            ledger.configure(enabled=enabled, threads=threads)
