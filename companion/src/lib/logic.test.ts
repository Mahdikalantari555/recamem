import { describe, expect, it } from 'vitest';
import { blockedReason, derivePet, formatGateStatus, formatMetric, petLabel, pickBest } from './logic';
import type { Checkpoint, CompanionState } from './types';

const baseState = (): CompanionState => ({
  anchor: 'mock', backend: 'memory-only', profile: 'demo',
  facts_total: 10, facts_active: 8, facts_retired: 2,
  feedback_counts: { approve: 4, reject: 1, total: 5 },
  memory_health: {}, ttt_status: {},
  retrieval_health: { available: false }, retrieval_hit_rate: null,
  last_sleep_at: null, selected_checkpoint: null,
  measured_at: 0, signals: []
});

const checkpoint = (name: string, bpb: number, eligible = true): Checkpoint => ({
  path: `/tmp/${name}`, name, eligible, reason: '', held_out_bpb: bpb, gates: { G2: 'passed' }
});

describe('derivePet', () => {
  it('derives knowledge and bond from measured signals only', () => {
    const pet = derivePet(baseState());
    expect(pet.knowledge.value).toBeCloseTo(0.4);
    expect(pet.bond.value).toBeCloseTo(0.4);
    expect(pet.energy.value).toBeNull();
    expect(pet.mood.value).toBeCloseTo(0.4);
  });

  it('never substitutes a decorative value for a missing signal', () => {
    const state = baseState();
    state.facts_active = null;
    const pet = derivePet(state);
    expect(pet.knowledge.value).toBeNull();
    expect(petLabel(null)).toBe('no signal');
  });

  it('uses labelled retrieval quality when available', () => {
    const state = baseState();
    state.retrieval_hit_rate = 0.8;
    expect(derivePet(state).mood.value).toBeCloseTo(0.6);
  });
});

describe('pickBest', () => {
  it('returns the eligible checkpoint with the lowest measured BPB', () => {
    expect(pickBest([checkpoint('a', 2.0), checkpoint('b', 1.5)])?.name).toBe('b');
  });

  it('refuses ineligible and unmeasured checkpoints', () => {
    expect(pickBest([checkpoint('a', 1.0, false)])).toBeNull();
    expect(pickBest([])).toBeNull();
  });
});

describe('honest display', () => {
  it('formats unavailable metrics without inventing a number', () => {
    expect(formatMetric(null)).toBe('unavailable');
    expect(formatMetric(1.23456)).toBe('1.2346');
  });

  it('surfaces blocked reasons instead of rendering a success state', () => {
    expect(blockedReason({ status: 'none', reason: 'no eligible checkpoint' })).toBe('no eligible checkpoint');
    expect(blockedReason({ status: 'blocked' })).toBe('blocked');
    expect(blockedReason({ status: 'ok' })).toBeNull();
  });

  it('formats gate statuses honestly without fabricating passes', () => {
    expect(formatGateStatus(true)).toBe('passed');
    expect(formatGateStatus('passed')).toBe('passed');
    expect(formatGateStatus(false)).toBe('failed');
    expect(formatGateStatus('failed')).toBe('failed');
    expect(formatGateStatus(null)).toBe('not evaluated');
    expect(formatGateStatus(undefined)).toBe('not evaluated');
  });
});
