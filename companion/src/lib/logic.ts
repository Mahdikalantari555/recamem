import type { Checkpoint, CompanionState, Pet, PetAttribute } from './types';

/** Every displayed attribute is derived from measured signals, or null. */
export function derivePet(state: CompanionState): Pet {
  const attr = (value: number | null, sources: string[]): PetAttribute => ({ value, sources });
  const active = state.facts_active;
  const approvals = state.feedback_counts?.approve ?? 0;
  const hitRate = state.retrieval_hit_rate;
  const knowledge = typeof active === 'number' ? Math.min(1, active / 20) : null;
  const bond = typeof approvals === 'number' ? Math.min(1, approvals / 10) : null;
  const energy = typeof state.last_sleep_at === 'number'
    ? 1 / (1 + Math.max(0, Date.now() / 1000 - state.last_sleep_at) / 86400)
    : null;
  const mood = typeof hitRate === 'number' && bond !== null
    ? Math.min(1, 0.5 * hitRate + 0.5 * bond)
    : bond;
  return {
    mood: attr(mood, ['retrieval_health.precision_at_k', 'feedback_counts.approve']),
    energy: attr(energy, ['last_sleep_at']),
    bond: attr(bond, ['feedback_counts.approve']),
    knowledge: attr(knowledge, ['facts_active']),
    note: 'A null value means its source signal was unavailable.'
  };
}

/** "Best" is the eligible checkpoint with the lowest measured held-out BPB. */
export function pickBest(rows: Checkpoint[]): Checkpoint | null {
  const eligible = rows.filter((row) => row.eligible && typeof row.held_out_bpb === 'number');
  if (eligible.length === 0) return null;
  return eligible.reduce((best, row) =>
    (row.held_out_bpb as number) < (best.held_out_bpb as number) ? row : best);
}

export function formatMetric(value: unknown): string {
  if (value === null || value === undefined) return 'unavailable';
  if (typeof value === 'number' && Number.isFinite(value)) return value.toFixed(4);
  return String(value);
}

export function petLabel(value: number | null): string {
  if (value === null) return 'no signal';
  if (value >= 0.75) return 'thriving';
  if (value >= 0.4) return 'content';
  return 'sleepy';
}

export function blockedReason(payload: unknown): string | null {
  if (!payload || typeof payload !== 'object') return null;
  const row = payload as Record<string, unknown>;
  if (row.status === 'blocked' || row.status === 'none') {
    return typeof row.reason === 'string' ? row.reason : 'blocked';
  }
  return null;
}

export function formatGateStatus(status: unknown): 'passed' | 'failed' | 'not evaluated' {
  if (status === true || status === 'passed') return 'passed';
  if (status === false || status === 'failed') return 'failed';
  return 'not evaluated';
}
