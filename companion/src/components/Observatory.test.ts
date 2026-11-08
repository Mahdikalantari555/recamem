// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/svelte';
import Observatory from '../components/Observatory.svelte';

/**
 * Component contract for the observatory, in jsdom:
 * - the semantic records list is always present, even when WebGL is missing;
 * - a failing graphics context never blocks the controls or the records;
 * - reduced motion starts the scene paused;
 * - the "never starts training" promise holds: only GET requests are issued.
 */

const health = {
  status: 'ok', profile: 'demo', anchor: 'mock', version: '0.2.0',
  device: 'cpu', loaded: true, active_op: null, error: null,
};
const learning = {
  status: 'blocked', enabled: false, threads: 1,
  reason: 'Background training is not implemented.', pending: 0,
  training_active: false, resource_limits_enforced: true,
};
const snapshot = {
  state: {
    anchor: 'mock', backend: 'memory-only', profile: 'demo',
    facts_total: 3, facts_active: 2, facts_retired: 1,
    feedback_counts: { approve: 1, reject: 0, total: 1 },
    memory_health: {}, ttt_status: {}, retrieval_health: {},
    retrieval_hit_rate: null, last_sleep_at: null, selected_checkpoint: null,
    measured_at: Math.floor(Date.now() / 1000) - 3, signals: [],
  },
  pet: {},
  last_fact_id: null,
  events: [
    { id: 'e1', action: 'feed', engine_call: "teach(text, source='companion')",
      target: null, timestamp: Math.floor(Date.now() / 1000) - 40, before: {}, after: {} },
  ],
};
const jobList = {
  jobs: [{
    id: 'job-123456', kind: 'train', status: 'done', progress: 1, tokens_seen: 1024,
    result: null, error: null, created_at: 1, updated_at: 2,
  }],
};

function stubFetch() {
  const calls: Array<{ url: string; method: string }> = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push({ url, method: (init?.method ?? 'GET').toUpperCase() });
    const body =
      url.endsWith('/health') ? health :
      url.endsWith('/learning') ? learning :
      url.endsWith('/companion/state') ? snapshot :
      url.endsWith('/jobs') ? jobList :
      { error: 'not found' };
    return { ok: url.endsWith('/jobs') || url.endsWith('/health') || url.endsWith('/learning') || url.endsWith('/companion/state'), status: 200, text: async () => JSON.stringify(body) } as Response;
  });
  vi.stubGlobal('fetch', fetchMock);
  return calls;
}

function stubNoWebGL() {
  // jsdom has no WebGL: make the renderer constructor fail the way Chrome does
  // when hardware acceleration is disabled.
  const original = HTMLCanvasElement.prototype.getContext;
  HTMLCanvasElement.prototype.getContext = function getContext() {
    return null;
  };
  return () => { HTMLCanvasElement.prototype.getContext = original; };
}

function stubMatchMedia(reduced: boolean) {
  vi.stubGlobal('matchMedia', (query: string) => ({
    matches: reduced && query.includes('prefers-reduced-motion'),
    media: query,
    addEventListener: () => {},
    removeEventListener: () => {},
    addListener: () => {},
    removeListener: () => {},
    dispatchEvent: () => false,
    onchange: null,
  }));
}

beforeEach(() => {
  stubMatchMedia(false);
  window.localStorage.clear();
  if (!globalThis.ResizeObserver) {
    vi.stubGlobal('ResizeObserver', class {
      observe() {} unobserve() {} disconnect() {}
    });
  }
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Observatory component', () => {
  it('keeps every record available when WebGL is unavailable', async () => {
    stubNoWebGL();
    stubFetch();
    render(Observatory);
    await waitFor(() => expect(screen.getByText(/Background training is not implemented/i)).toBeTruthy());
    // records list, not the scene, carries the data
    expect(await screen.findByRole('button', { name: /Active fact #1/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /train run job-1234/i })).toBeTruthy();
    // the failure is stated, and the 2D view is offered
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Use the 2D view/i })).toBeTruthy(),
    );
  });

  it('offers the accessible 2D table with sources', async () => {
    stubNoWebGL();
    stubFetch();
    render(Observatory);
    await screen.findByRole('button', { name: /Active fact #1/i });
    await screen.getByRole('button', { name: /Accessible 2D/i }).click();
    const table = await screen.findByTestId('observatory-2d');
    expect(table.querySelectorAll('tbody tr')).toHaveLength(4); // 2 facts + 1 event + 1 job
    expect(table.textContent).toMatch(/GET \/api\/v1\/jobs/);
    expect(table.textContent).toMatch(/count projection/i);
  });

  it('never writes, never trains, never leaves the origin', async () => {
    stubNoWebGL();
    const calls = stubFetch();
    render(Observatory);
    await screen.findByRole('button', { name: /Active fact #1/i });
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.every((call) => call.method === 'GET')).toBe(true);
    expect(calls.every((call) => call.url.startsWith('/api/v1/'))).toBe(true);
  });

  it('starts paused when the system asks for reduced motion', async () => {
    stubNoWebGL();
    stubMatchMedia(true);
    stubFetch();
    render(Observatory);
    const button = await screen.findByRole('button', { name: /Pause motion|Resume motion/i });
    expect(button.textContent).toMatch(/Resume motion/);
  });

  it('states that the backend is blocked and that motion is not evidence', async () => {
    stubNoWebGL();
    stubFetch();
    render(Observatory);
    expect(await screen.findByText(/background learning blocked/i)).toBeTruthy();
    expect(screen.getByText(/Motion is not evidence/i)).toBeTruthy();
    expect(screen.getByText(/never starts training, research or memory writes/i)).toBeTruthy();
  });

  it('surfaces a backend error without inventing records', async () => {
    stubNoWebGL();
    vi.stubGlobal('fetch', vi.fn(async () => { throw new Error('offline'); }));
    render(Observatory);
    await waitFor(() =>
      expect(screen.getByText(/Backend unavailable/i)).toBeTruthy(),
    );
    expect(screen.getByText(/No measured memory signals/i)).toBeTruthy();
    expect(screen.getByText(/offline/)).toBeTruthy();
  });
});
