import type { CompanionEvent, CompanionState, Health, Job } from './types';
import type { LearningStatus } from './api';

/**
 * The observatory is a *view* of records the backend already exposes. Nothing
 * here invents model internals, embeddings or training activity:
 *
 * - `job` nodes are training/eval jobs from GET /api/v1/jobs;
 * - `event` nodes are recorded companion actions from GET /api/v1/companion/state;
 * - `memory` nodes are a *count projection* of `facts_active` (the API reports
 *   how many active facts exist, not which facts they are);
 * - the centre core is decorative identity only, and is flagged as such.
 *
 * Layout is deterministic so the same records always appear in the same place,
 * and so a test can assert it without rendering.
 */

export type NodeKind = 'job' | 'event' | 'memory';
export type Tone = 'green' | 'lavender' | 'amber' | 'grey';

export interface ObservatoryInput {
  health: Health | null;
  learning: LearningStatus | null;
  state: CompanionState | null;
  jobs: Job[];
  events: CompanionEvent[];
  error?: string;
}

export interface ObservatoryNode {
  id: string;
  kind: NodeKind;
  title: string;
  detail: string;
  provenance: string;
  /** True when the node stands for a specific backend record. */
  measured: boolean;
  tone: Tone;
  /** 0 = inner memory shell, 1 = events, 2 = jobs. */
  shell: number;
  /** Relative marker size; placement is computed per shell by the scene. */
  size: number;
  selectable: boolean;
}

export interface ObservatoryModel {
  status: { label: string; tone: Tone; reason: string };
  freshness: {
    measuredAt: number | null;
    ageSeconds: number | null;
    stale: boolean;
    label: string;
  };
  counts: {
    jobs: number;
    events: number;
    factsActive: number | null;
    factsRetired: number | null;
    feedback: number;
  };
  identity: { profile: string; anchor: string; device: string };
  nodes: ObservatoryNode[];
  notices: string[];
  empty: boolean;
  decorative: string[];
}

const STALE_AFTER_SECONDS = 30;
const MAX_MEMORY_NODES = 24;
const MAX_EVENT_NODES = 12;
const MAX_JOB_NODES = 8;

const SHELL_RADIUS = [2.0, 3.05, 4.15];

export function shellRadius(shell: number): number {
  return SHELL_RADIUS[Math.min(SHELL_RADIUS.length - 1, Math.max(0, shell))];
}

/** Golden-angle spiral: stable, evenly spaced, no randomness. */
export function placeOnShell(index: number, total: number, shell: number) {
  const golden = Math.PI * (3 - Math.sqrt(5));
  const y = total <= 1 ? 0 : 1 - (2 * index) / (total - 1);
  const radius = Math.sqrt(Math.max(0, 1 - y * y));
  const angle = golden * index;
  return {
    x: Math.cos(angle) * radius * shellRadius(shell),
    y: y * shellRadius(shell),
    z: Math.sin(angle) * radius * shellRadius(shell),
  };
}

export function describeLearning(learning: LearningStatus | null): {
  label: string;
  tone: Tone;
  reason: string;
} {
  if (!learning) {
    return { label: 'status unavailable', tone: 'grey', reason: 'The learning status endpoint did not answer.' };
  }
  const reason = learning.reason || 'No reason was reported.';
  if (learning.status === 'blocked') return { label: 'blocked', tone: 'amber', reason };
  return { label: 'paused', tone: 'lavender', reason };
}

export function describeFreshness(measuredAt: number | null, now: number): {
  measuredAt: number | null;
  ageSeconds: number | null;
  stale: boolean;
  label: string;
} {
  if (measuredAt === null || !Number.isFinite(measuredAt)) {
    return { measuredAt: null, ageSeconds: null, stale: true, label: 'never measured' };
  }
  // The companion state reports `measured_at` in epoch seconds.
  const ageSeconds = Math.max(0, now / 1000 - measuredAt);
  const stale = ageSeconds > STALE_AFTER_SECONDS;
  const rounded = ageSeconds < 1 ? 'just now' : `${Math.round(ageSeconds)} s ago`;
  return {
    measuredAt,
    ageSeconds,
    stale,
    label: stale ? `${rounded} (stale)` : rounded,
  };
}

function jobTone(job: Job): Tone {
  if (job.status === 'running') return 'green';
  if (job.status === 'done') return 'lavender';
  if (job.status === 'failed' || job.error) return 'amber';
  return 'grey';
}

function jobDetail(job: Job): string {
  const parts = [`status ${job.status}`];
  if (typeof job.tokens_seen === 'number') parts.push(`${job.tokens_seen.toLocaleString()} tokens seen`);
  if (typeof job.progress === 'number' && job.status === 'running') {
    parts.push(`${Math.round(Math.min(1, Math.max(0, job.progress)) * 100)}% of the budget`);
  }
  if (job.checkpoint_path) parts.push('checkpoint saved');
  if (job.error) parts.push(`error: ${job.error}`);
  return parts.join(' · ');
}

export function buildObservatoryModel(input: ObservatoryInput, now = Date.now()): ObservatoryModel {
  const { health, learning, state, jobs, events } = input;
  const notices: string[] = [];
  const nodes: ObservatoryNode[] = [];

  if (input.error) notices.push(`Backend unavailable: ${input.error}`);

  const factsActive = typeof state?.facts_active === 'number' ? state.facts_active : null;
  const factsRetired = typeof state?.facts_retired === 'number' ? state.facts_retired : null;
  const feedback = state?.feedback_counts?.total ?? 0;

  const status = describeLearning(learning);
  const freshness = describeFreshness(state?.measured_at ?? null, now);

  if (learning && learning.status === 'blocked') {
    notices.push('Background learning is blocked. The scene is decoration; motion does not mean training.');
  }
  if (freshness.stale && state) {
    notices.push('Memory signals are stale; the last measurement did not come from this session.');
  }
  if (input.error || !state) {
    notices.push('No measured memory signals. Nothing data-driven is drawn until the backend answers.');
  }

  // -- memory: count projection only ------------------------------------- #
  const memoryCount = factsActive === null ? 0 : Math.min(factsActive, MAX_MEMORY_NODES);
  for (let index = 0; index < memoryCount; index++) {
    nodes.push({
      id: `memory-${index}`,
      kind: 'memory',
      title: `Active fact #${index + 1}`,
      detail:
        `Count projection: one of ${factsActive} active facts` +
        (factsRetired ? ` (${factsRetired} retired)` : '') +
        '. This API reports the count, not the individual record, so this marker is not a specific fact.',
      provenance: 'GET /api/v1/companion/state → state.facts_active',
      measured: true,
      tone: 'green',
      shell: 0,
      size: 1,
      selectable: true,
    });
  }

  // -- events: real recorded companion actions --------------------------- #
  const recentEvents = events.slice(-MAX_EVENT_NODES);
  recentEvents.forEach((event, index) => {
    nodes.push({
      id: `event-${event.id ?? index}`,
      kind: 'event',
      title: event.action || 'companion action',
      detail: `${event.engine_call || 'engine call unavailable'} · ${
        event.timestamp ? new Date(event.timestamp * 1000).toLocaleTimeString() : 'time unavailable'
      }`,
      provenance: 'GET /api/v1/companion/state → events[]',
      measured: true,
      tone: 'green',
      shell: 1,
      size: 1,
      selectable: true,
    });
  });

  // -- jobs: real training / eval runs ----------------------------------- #
  const recentJobs = jobs.slice().sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0)).slice(0, MAX_JOB_NODES);
  recentJobs.forEach((job) => {
    nodes.push({
      id: `job-${job.id}`,
      kind: 'job',
      title: `${job.kind} run ${String(job.id).slice(0, 8)}`,
      detail: jobDetail(job),
      provenance: 'GET /api/v1/jobs',
      measured: true,
      tone: jobTone(job),
      shell: 2,
      size: job.status === 'running' ? 1.45 : 1.15,
      selectable: true,
    });
  });

  const empty = nodes.length === 0;

  return {
    status,
    freshness,
    counts: {
      jobs: jobs.length,
      events: events.length,
      factsActive,
      factsRetired,
      feedback,
    },
    identity: {
      profile: health?.profile ?? state?.profile ?? 'unknown profile',
      anchor: health?.anchor ?? state?.anchor ?? 'unknown anchor',
      device: health?.device ?? 'unknown device',
    },
    nodes,
    notices,
    empty,
    decorative: [
      'The central core is identity decoration, not a depiction of weights.',
      'Shell rings are layout guides. Node position is a deterministic layout, not embedding geometry.',
    ],
  };
}
