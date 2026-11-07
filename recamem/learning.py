"""Durable control plane for future incremental learning.

This module records consent and references to existing evidence, not copies of
facts. It does not schedule work or train model weights. A queue entry is not a
claim that learning has occurred.
"""
from __future__ import annotations

import re
from contextlib import contextmanager
import sqlite3
import time
from pathlib import Path


class LearningPolicyError(ValueError):
    """An intake or configuration request violates the learning policy."""


class LearningLedger:
    def __init__(self, base_dir: str | Path, profile: str):
        self.path = Path(base_dir) / "learning-control.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.profile = profile
        with self._connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS learning_config (
                    profile TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1,
                    threads INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS learning_sources (
                    profile TEXT NOT NULL, source TEXT NOT NULL,
                    approved INTEGER NOT NULL, updated_at REAL NOT NULL,
                    PRIMARY KEY(profile, source)
                );
                CREATE TABLE IF NOT EXISTS learning_intake (
                    profile TEXT NOT NULL, event_id TEXT NOT NULL,
                    source TEXT NOT NULL, evidence_ref TEXT NOT NULL,
                    sha256 TEXT NOT NULL, status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY(profile, event_id)
                );
            ''')
            db.execute('INSERT OR IGNORE INTO learning_config(profile) VALUES (?)', (profile,))

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def configure(self, *, enabled: bool, threads: int) -> dict:
        if not isinstance(enabled, bool) or isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 32:
            raise LearningPolicyError('enabled must be boolean and threads must be an integer from 1 to 32')
        with self._connect() as db:
            db.execute('UPDATE learning_config SET enabled=?, threads=? WHERE profile=?',
                       (int(enabled), threads, self.profile))
        return self.status()

    def approve_source(self, source: str, approved: bool) -> None:
        self._validate_identifier(source, 'source')
        if not isinstance(approved, bool):
            raise LearningPolicyError('approved must be boolean')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('''INSERT INTO learning_sources(profile,source,approved,updated_at)
                          VALUES(?,?,?,?) ON CONFLICT(profile,source) DO UPDATE SET
                          approved=excluded.approved, updated_at=excluded.updated_at''',
                       (self.profile, source, int(approved), time.time()))
            if not approved:
                db.execute("UPDATE learning_intake SET status='excluded' WHERE profile=? AND source=? AND status='pending'",
                           (self.profile, source))

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        if not isinstance(value, str) or not value.strip() or len(value) > 256 or any(ord(c) < 32 for c in value):
            raise LearningPolicyError(f'{label} must be a non-empty identifier of at most 256 characters')

    def enqueue(self, event_id: str, source: str, evidence_ref: str, sha256: str) -> bool:
        """Queue an authorized immutable reference. Return False for an exact retry.

        Caller is responsible for the reference pointing to durable, authorized
        evidence; no consumer is implemented yet. Conflicting retries fail.
        """
        for label, value in [('event_id', event_id), ('source', source), ('evidence_ref', evidence_ref)]:
            self._validate_identifier(value, label)
        if not isinstance(sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', sha256):
            raise LearningPolicyError('sha256 must be a lowercase SHA-256 digest')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute('SELECT source,evidence_ref,sha256 FROM learning_intake WHERE profile=? AND event_id=?',
                                  (self.profile, event_id)).fetchone()
            if existing:
                if tuple(existing) != (source, evidence_ref, sha256):
                    raise LearningPolicyError('event_id already exists with different evidence')
                return False
            source_row = db.execute('SELECT approved FROM learning_sources WHERE profile=? AND source=?',
                                    (self.profile, source)).fetchone()
            if not source_row or not source_row['approved']:
                raise LearningPolicyError('source is not approved for training')
            db.execute('INSERT INTO learning_intake VALUES (?,?,?,?,?,?,?)',
                       (self.profile, event_id, source, evidence_ref, sha256, 'pending', time.time()))
            return True

    def status(self) -> dict:
        with self._connect() as db:
            config = db.execute('SELECT enabled,threads FROM learning_config WHERE profile=?', (self.profile,)).fetchone()
            counts = {row['status']: row['n'] for row in db.execute(
                'SELECT status,COUNT(*) AS n FROM learning_intake WHERE profile=? GROUP BY status', (self.profile,))}
            sources = [dict(row) for row in db.execute(
                'SELECT source,approved,updated_at FROM learning_sources WHERE profile=? ORDER BY source', (self.profile,))]
        enabled = bool(config['enabled'])
        return {'status': 'blocked' if enabled else 'paused', 'enabled': enabled,
                'reason': 'Background training, evaluation and checkpoint promotion are not implemented.',
                'threads': config['threads'], 'pending': counts.get('pending', 0),
                'excluded': counts.get('excluded', 0), 'sources': sources,
                'training_active': False, 'resource_limits_enforced': False}
