import { describe, expect, it } from 'vitest';
import {
  buildObservatoryModel,
  describeFreshness,
  describeLearning,
  placeOnShell,
  shellRadius,
  type ObservatoryInput,
} from './observatory';
import type { CompanionEvent, CompanionState, Job } from './types';
import type { LearningStatus } from './api';

const NOW = 1_700_000_000_000; // ms

const health = {
  status: 'ok', profile: 'demo', anchor: 'mock', version: '0.2.0',
  device: 'cpu', loaded: true, active_op: null, error: null,
};

const state = (overrides: Partial<CompanionState> = {}): CompanionState => ({
  anchor: 'mock', backend: 'memory-only', profile: 'demo',
  facts_total: 10, facts_active: 6, facts_retired: 4,
  feedback_counts: { approve: 3, reject: 1, total: 4 },
  memory_health: {}, ttt_status: {}, retrieval_health: {},
  retrieval_hit_rate: null, last_sleep_at: null, selected_checkpoint: null,
  measured_at: NOW / 1000 - 2, signals: [],
  ...overrides,
});

const blocked: LearningStatus = {
  status: 'blocked', enabled: false, threads: 1,
  reason: 'No learning supervisor is configured.',
  pending: 0, training_active: false, resource_limits_enforced: true,
};
const paused: LearningStatus = { ...blocked, status: 'paused', enabled: true };

const job = (id: string, statusValue: string, extra: Partial<Job> = {}): Job => ({
  id, kind: 'train', status: statusValue, progress: 0.5, tokens_seen: 256,
  result: null, error: null, created_at: 1, updated_at: 2, ...extra,
});

const event = (id: string, action: string): CompanionEvent => ({
  id, action, engine_call: 'teach(text=…)', target: null, timestamp: NOW / 1000 - 5,
  before: {}, after: {},
});

const input = (overrides: Partial<ObservatoryInput> = {}): ObservatoryInput => ({
  health, learning: blocked, state: state(), jobs: [], events: [], ...overrides,
});

describe('observatory model', () => {
  it('draws nothing data-driven when the backend has no records', () => {
    const model = buildObservatoryModel(input({ state: null, learning: null }), NOW);
    expect(model.empty).toBe(true);
    expect(model.nodes).toHaveLength(0);
    expect(model.notices.join(' ')).toMatch(/no measured memory signals/i);
    expect(model.status.label).toBe('status unavailable');
  });

  it('maps each job and event to exactly one measured node', () => {
    const model = buildObservatoryModel(input({
      jobs: [job('abc12345', 'running'), job('def67890', 'done')],
      events: [event('e1', 'feed'), event('e2', 'sleep')],
    }), NOW);
    const jobs = model.nodes.filter((node) => node.kind === 'job');
    const events = model.nodes.filter((node) => node.kind === 'event');
    expect(jobs).toHaveLength(2);
    expect(events).toHaveLength(2);
    expect(jobs.every((node) => node.measured && node.provenance === 'GET /api/v1/jobs')).toBe(true);
    expect(events.every((node) => node.provenance.startsWith('GET /api/v1/companion/state'))).toBe(true);
    expect(jobs.find((node) => node.id === 'job-abc12345')?.tone).toBe('green');
    expect(jobs.find((node) => node.id === 'job-def67890')?.tone).toBe('lavender');
  });

  it('treats memory markers as a count projection, never as specific facts', () => {
    const model = buildObservatoryModel(input({ state: state({ facts_active: 6 }) }), NOW);
    const memory = model.nodes.filter((node) => node.kind === 'memory');
    expect(memory).toHaveLength(6);
    expect(memory[0].detail).toMatch(/count projection/i);
    expect(memory[0].provenance).toContain('facts_active');
    expect(model.counts.factsActive).toBe(6);
  });

  it('caps the marker budget so a large store cannot flood the scene', () => {
    const model = buildObservatoryModel(input({
      state: state({ facts_active: 5000 }),
      events: Array.from({ length: 40 }, (_, i) => event(`e${i}`, 'feed')),
      jobs: Array.from({ length: 20 }, (_, i) => job(`j${i}`, 'done')),
    }), NOW);
    expect(model.nodes.filter((node) => node.kind === 'memory')).toHaveLength(24);
    expect(model.nodes.filter((node) => node.kind === 'event')).toHaveLength(12);
    expect(model.nodes.filter((node) => node.kind === 'job')).toHaveLength(8);
    expect(model.counts.jobs).toBe(20); // the count stays truthful
  });

  it('reports blocking honestly and never implies training', () => {
    const model = buildObservatoryModel(input({ learning: blocked }), NOW);
    expect(model.status).toEqual({
      label: 'blocked', tone: 'amber', reason: blocked.reason,
    });
    expect(model.notices.join(' ')).toMatch(/motion does not mean training/i);
  });

  it('marks stale measurements instead of presenting them as live', () => {
    const model = buildObservatoryModel(
      input({ state: state({ measured_at: NOW / 1000 - 600 }) }), NOW,
    );
    expect(model.freshness.stale).toBe(true);
    expect(model.freshness.label).toMatch(/stale/);
    expect(model.notices.join(' ')).toMatch(/stale/i);
  });

  it('surfaces a backend error without inventing data', () => {
    const model = buildObservatoryModel(input({ state: null, learning: null, error: 'HTTP 503' }), NOW);
    expect(model.notices[0]).toContain('HTTP 503');
    expect(model.nodes).toHaveLength(0);
  });

  it('is deterministic: identical records give identical nodes', () => {
    const build = () => buildObservatoryModel(input({
      jobs: [job('a1', 'done')], events: [event('e1', 'feed')], state: state({ facts_active: 3 }),
    }), NOW);
    expect(build()).toEqual(build());
  });

  it('places nodes deterministically inside their shell', () => {
    const first = placeOnShell(0, 8, 1);
    const second = placeOnShell(3, 8, 1);
    expect(first).not.toEqual(second);
    for (const point of [first, second]) {
      const distance = Math.hypot(point.x, point.y, point.z);
      expect(distance).toBeCloseTo(shellRadius(1), 5);
      expect(Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z)).toBe(true);
    }
  });

  it('keeps a single node on the shell instead of dividing by zero', () => {
    const point = placeOnShell(0, 1, 2);
    expect(Math.hypot(point.x, point.y, point.z)).toBeCloseTo(shellRadius(2), 5);
    expect(Number.isFinite(point.y)).toBe(true);
  });
});

describe('status helpers', () => {
  it('distinguishes paused from blocked', () => {
    expect(describeLearning(paused).label).toBe('paused');
    expect(describeLearning(paused).tone).toBe('lavender');
    expect(describeLearning(null).label).toBe('status unavailable');
  });

  it('reports freshness in seconds and never negative age', () => {
    const fresh = describeFreshness(NOW / 1000 - 2, NOW);
    expect(fresh.stale).toBe(false);
    expect(fresh.ageSeconds).toBeCloseTo(2, 5);
    const future = describeFreshness(NOW / 1000 + 30, NOW);
    expect(future.ageSeconds).toBe(0);
    expect(describeFreshness(null, NOW).label).toBe('never measured');
  });
});
