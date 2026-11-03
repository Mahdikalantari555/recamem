import type { Checkpoint, CompanionEvent, CompanionState, Health, Job, Pet, Turn } from './types';

const PREFIX = '/api/v1';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${PREFIX}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) }
  });
  const text = await response.text();
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    body = { error: text };
  }
  if (!response.ok) {
    const message = (body as { error?: string }).error || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return body as T;
}

const post = <T>(path: string, payload: unknown) =>
  request<T>(path, { method: 'POST', body: JSON.stringify(payload) });

export const getHealth = () => request<Health>('/health');
export const getState = () =>
  request<{ state: CompanionState; pet: Pet; events: CompanionEvent[]; last_fact_id: number | null }>(
    '/companion/state'
  );
export const act = (action: string, payload: Record<string, unknown> = {}) =>
  post<{ event: CompanionEvent; result: unknown; state: CompanionState }>('/companion/act', { action, ...payload });
export const inference = (prompt: string) =>
  post<{ prompt: string; response: string; memory: Turn['memory']; anchor: string; backend: string }>(
    '/inference', { prompt, mode: 'chat' });
export const feedback = (action: string, factId: number, text?: string) =>
  post<{ result: unknown }>('/feedback', { action, fact_id: factId, ...(text ? { text } : {}) });
export const listCheckpoints = () => request<{ checkpoints: Checkpoint[] }>('/checkpoints');
export const bestCheckpoint = () =>
  request<{ status: string; reason?: string; checkpoint?: Checkpoint }>('/checkpoints/best');
export const selectCheckpoint = (name: string) =>
  post<{ selected: string; checkpoint: Checkpoint; applied_to_generation: boolean; reason: string }>(
    '/checkpoints/select', { name }
  );
export const createJob = (params: Record<string, unknown>) => post<Job>('/jobs', { kind: 'train', ...params });
export const getJob = (id: string) => request<Job>(`/jobs/${encodeURIComponent(id)}`);
export const cancelJob = (id: string) =>
  post<Job>(`/jobs/${encodeURIComponent(id)}/cancel`, {});
export const listJobs = () => request<{ jobs: Job[] }>('/jobs');

/** SSE for real training progress. Callers fall back to polling getJob(). */
export function jobEvents(id: string, onEvent: (event: Record<string, unknown>) => void): () => void {
  const source = new EventSource(`${PREFIX}/jobs/${encodeURIComponent(id)}/events`);
  source.onmessage = (message: MessageEvent<string>) => {
    try {
      onEvent(JSON.parse(message.data) as Record<string, unknown>);
    } catch {
      onEvent({ type: 'log', line: message.data });
    }
  };
  return () => source.close();
}
