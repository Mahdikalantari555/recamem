<script lang="ts">
  import { untrack } from 'svelte';
  import { getHealth, getLearning, getState, listJobs, type LearningStatus } from '../lib/api';
  import type { CompanionEvent, CompanionState, Health, Job } from '../lib/types';
  import { buildObservatoryModel, type ObservatoryNode } from '../lib/observatory';
  import { createObservatoryScene, type SceneHandle } from '../lib/observatoryScene';

  let { compact = false }: { compact?: boolean } = $props();

  type Mode = '3d' | '2d';
  const STORE_KEY = 'recamem.observatory.3d';

  const prefersReducedMotion = () =>
    typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

  const readMode = (): Mode => {
    try {
      return window.localStorage.getItem(STORE_KEY) === 'off' ? '2d' : '3d';
    } catch {
      return '3d';
    }
  };

  let mode = $state<Mode>('3d');
  let motion = $state(true);
  let userChoseMotion = $state(false);

  let health = $state<Health | null>(null);
  let learning = $state<LearningStatus | null>(null);
  let engineState = $state<CompanionState | null>(null);
  let jobs = $state<Job[]>([]);
  let events = $state<CompanionEvent[]>([]);
  let error = $state('');
  let loading = $state(true);
  let clock = $state(Date.now());
  let lastRefresh = $state<number | null>(null);
  const failures: string[] = [];

  let canvasHost = $state<HTMLDivElement | null>(null);
  let scene = $state<SceneHandle | null>(null);
  let sceneEpoch = $state(0);
  let sceneIssue = $state('');
  let contextLost = $state(false);
  let selectedId = $state<string | null>(null);
  let activeTab = $state<'signals' | 'provenance'>('signals');

  const model = $derived(
    buildObservatoryModel({ health, learning, state: engineState, jobs, events, error }, clock),
  );
  const selected = $derived(model.nodes.find((node) => node.id === selectedId) ?? null);
  const nodeSignature = $derived(
    model.nodes.map((node) => `${node.id}:${node.tone}:${node.size}`).join('|'),
  );
  const shellLabels = ['Memory', 'Companion actions', 'Training runs'] as const;

  const byShell = $derived(
    [0, 1, 2].map((shell) => ({
      shell,
      label: shellLabels[shell],
      nodes: model.nodes.filter((node) => node.shell === shell),
    })),
  );

  async function settle<T>(work: Promise<T>, fallback: T, label: string): Promise<T> {
    try {
      return await work;
    } catch (cause) {
      failures.push(`${label}: ${cause instanceof Error ? cause.message : String(cause)}`);
      return fallback;
    }
  }

  async function refresh() {
    if (typeof document !== 'undefined' && document.hidden) return;
    loading = true;
    failures.length = 0;
    const [nextHealth, nextLearning, snapshot, jobList] = await Promise.all([
      settle(getHealth(), null, 'health'),
      settle(getLearning(), null, 'learning status'),
      settle(getState(), null, 'companion state'),
      settle(listJobs(), null, 'jobs'),
    ]);
    health = nextHealth;
    learning = nextLearning;
    if (snapshot) {
      engineState = snapshot.state;
      events = snapshot.events;
    }
    jobs = jobList?.jobs ?? [];
    error = failures.join(' · ');
    lastRefresh = Date.now();
    clock = Date.now();
    loading = false;
  }

  function chooseMode(next: Mode) {
    mode = next;
    try {
      window.localStorage.setItem(STORE_KEY, next === '3d' ? 'on' : 'off');
    } catch {
      /* storage may be blocked; the choice still applies to this view */
    }
  }

  function toggleMotion() {
    userChoseMotion = true;
    motion = !motion;
  }

  function selectNode(id: string | null) {
    selectedId = id;
    if (id) activeTab = 'signals';
  }

  $effect(() => {
    motion = !prefersReducedMotion();
    const query = window.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (!query) return;
    const onChange = () => {
      if (!userChoseMotion) motion = !query.matches;
    };
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  });

  $effect(() => {
    mode = readMode();
    void refresh();
    const timer = window.setInterval(() => {
      clock = Date.now();
      void refresh();
    }, 20000);
    const onVisible = () => {
      if (!document.hidden) void refresh();
    };
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisible);
    };
  });

  // Scene lifetime follows the 3D mode only.
  $effect(() => {
    if (mode !== '3d' || !canvasHost) return;
    let disposed = false;
    let handle: SceneHandle | null = null;
    sceneIssue = '';
    contextLost = false;

    const startMotion = untrack(() => motion);
    void createObservatoryScene(canvasHost, {
      motion: startMotion,
      onSelect: (id) => selectNode(id),
      onContextLost: () => {
        contextLost = true;
        sceneIssue = 'The graphics context was lost. The scene is stopped; the list and controls still work.';
      },
      onContextRestored: () => {
        contextLost = false;
        sceneIssue = 'The graphics context was restored.';
      },
    }).then((created) => {
      if (disposed) {
        created.dispose();
        return;
      }
      handle = created;
      scene = created;
      if (!created.supported) {
        sceneIssue = created.reason ?? '3D is unavailable in this browser.';
      }
      sceneEpoch += 1;
    });

    return () => {
      disposed = true;
      scene = null;
      handle?.dispose();
    };
  });

  $effect(() => {
    void sceneEpoch;
    void nodeSignature;
    const nodes = untrack(() => model.nodes);
    if (scene?.supported) scene.setNodes(nodes);
  });

  $effect(() => {
    void sceneEpoch;
    if (scene?.supported) scene.setSelected(selectedId);
  });

  $effect(() => {
    void sceneEpoch;
    if (scene?.supported) scene.setMotion(motion && !contextLost);
  });

  const relativeRefresh = (value: number | null) => {
    if (!value) return 'not loaded yet';
    const seconds = Math.max(0, Math.round((clock - value) / 1000));
    return seconds < 2 ? 'just now' : `${seconds} s ago`;
  };

  const toneLabel: Record<ObservatoryNode['tone'], string> = {
    green: 'active',
    lavender: 'recorded',
    amber: 'attention',
    grey: 'inactive',
  };
</script>

<section class="observatory" aria-labelledby="observatory-title">
  <header class="obs-head">
    <div>
      <div class="obs-eyebrow"><span>✳</span> THREE-DIMENSIONAL LEARNING STUDIO</div>
      <h1 id="observatory-title">The learning observatory</h1>
      <p class="obs-lede">
        A scene of the records this machine actually holds: active memory markers, companion
        actions and training runs. The glowing core is decoration, not a picture of model weights.
      </p>
    </div>
    <div class="obs-freshness">
      <span class:stale={model.freshness.stale} class="dot" aria-hidden="true"></span>
      <div>
        <b>Measured {model.freshness.label}</b>
        <small>Refreshed {relativeRefresh(lastRefresh)} · read-only</small>
      </div>
    </div>
  </header>

  <div class="obs-controls" role="group" aria-label="Observatory display controls">
    <div class="segmented">
      <button
        type="button"
        class:active={mode === '3d'}
        aria-pressed={mode === '3d'}
        onclick={() => chooseMode('3d')}
      >3D scene</button>
      <button
        type="button"
        class:active={mode === '2d'}
        aria-pressed={mode === '2d'}
        onclick={() => chooseMode('2d')}
      >Accessible 2D</button>
    </div>
    <button
      type="button"
      class="ctrl"
      aria-pressed={motion}
      disabled={mode !== '3d' || contextLost}
      onclick={toggleMotion}
    >{motion ? 'Pause motion' : 'Resume motion'}</button>
    <button type="button" class="ctrl" onclick={() => { selectedId = null; }} disabled={!selectedId}>
      Clear selection
    </button>
    <button type="button" class="ctrl" onclick={() => void refresh()} disabled={loading}>
      {loading ? 'Reading…' : 'Refresh records'}
    </button>
    <span class="obs-hint">Opening this view never starts training, research or memory writes.</span>
  </div>

  <div class="obs-grid" class:compact>
    <div class="obs-stage">
      {#if mode === '3d'}
        <div
          class="scene"
          class:dim={contextLost}
          bind:this={canvasHost}
          data-testid="observatory-canvas"
        ></div>
        <p class="sr-only" id="scene-summary">
          Decorative 3D scene plus {model.nodes.length} record markers
          ({model.counts.factsActive ?? 0} active memory facts, {model.counts.events} recorded actions,
          {model.counts.jobs} training runs). The same records and controls are listed beside the
          scene and in the accessible 2D view.
        </p>
        <div class="scene-legend">
          <span><i class="swatch memory"></i> Memory markers</span>
          <span><i class="swatch event"></i> Actions</span>
          <span><i class="swatch job"></i> Runs</span>
          <span><i class="swatch core"></i> Core (decorative)</span>
        </div>
        {#if sceneIssue}
          <p class="scene-note" role="status">
            {sceneIssue}
            {#if !scene?.supported}
              <button type="button" class="link" onclick={() => chooseMode('2d')}>Use the 2D view</button>
            {/if}
          </p>
        {/if}
      {:else}
        <div class="obs-2d" data-testid="observatory-2d">
          <h2>Accessible 2D view</h2>
          <p>
            Same records as the scene, no graphics context required. Every row names its source,
            so nothing here depends on WebGL, motion or pointer input.
          </p>
          {#if model.empty}
            <p class="obs-empty">
              No records yet. Nothing data-driven is drawn — teach a fact or finish a run and the
              markers appear here.
            </p>
          {:else}
            <table>
              <caption class="sr-only">Measured records available to the observatory</caption>
              <thead>
                <tr><th scope="col">Record</th><th scope="col">Kind</th><th scope="col">Signal</th></tr>
              </thead>
              <tbody>
                {#each model.nodes as node (node.id)}
                  <tr class:selected={node.id === selectedId}>
                    <th scope="row">
                      <button type="button" class="row-button" onclick={() => selectNode(node.id)}>
                        {node.title}
                      </button>
                    </th>
                    <td>{node.kind}</td>
                    <td>{node.detail}<br /><small class="row-source">{node.provenance}</small></td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      {/if}

      {#if model.notices.length}
        <ul class="obs-notices">
          {#each model.notices as notice (notice)}
            <li>{notice}</li>
          {/each}
        </ul>
      {/if}
    </div>

    <aside class="obs-panel">
      <div class="panel-tabs" role="tablist" aria-label="Observatory details">
        <button
          type="button"
          role="tab"
          id="tab-signals"
          aria-selected={activeTab === 'signals'}
          aria-controls="panel-signals"
          class:active={activeTab === 'signals'}
          onclick={() => activeTab = 'signals'}
        >Records</button>
        <button
          type="button"
          role="tab"
          id="tab-provenance"
          aria-selected={activeTab === 'provenance'}
          aria-controls="panel-provenance"
          class:active={activeTab === 'provenance'}
          onclick={() => activeTab = 'provenance'}
        >Provenance</button>
      </div>

      {#if activeTab === 'signals'}
        <div id="panel-signals" role="tabpanel" aria-labelledby="tab-signals">
          <div class="status-card">
            <span class="badge tone-{model.status.tone}">background learning {model.status.label}</span>
            <p>{model.status.reason}</p>
            <small>The scene animates whether or not learning is on. Motion is not evidence.</small>
          </div>

          <dl class="counts">
            <div><dt>Active facts</dt><dd>{model.counts.factsActive ?? '—'}</dd></div>
            <div><dt>Retired</dt><dd>{model.counts.factsRetired ?? '—'}</dd></div>
            <div><dt>Feedback</dt><dd>{model.counts.feedback}</dd></div>
            <div><dt>Runs</dt><dd>{model.counts.jobs}</dd></div>
          </dl>

          <h3 class="panel-heading">Select a record</h3>
          {#if model.empty}
            <p class="obs-empty">
              Nothing measured to show. The scene still renders a decorative core, clearly labelled.
            </p>
          {:else}
            {#each byShell as group (group.shell)}
              {#if group.nodes.length}
                <div class="shell-group">
                  <h4>{group.label}<span>{group.nodes.length}</span></h4>
                  <ul>
                    {#each group.nodes as node (node.id)}
                      <li>
                        <button
                          type="button"
                          class:selected={node.id === selectedId}
                          aria-pressed={node.id === selectedId}
                          onclick={() => selectNode(node.id)}
                        >
                          <i class="dot tone-{node.tone}" aria-hidden="true"></i>
                          <span>{node.title}</span>
                          <small>{toneLabel[node.tone]}</small>
                        </button>
                      </li>
                    {/each}
                  </ul>
                </div>
              {/if}
            {/each}
          {/if}

          <div class="detail" aria-live="polite">
            {#if selected}
              <h3>{selected.title}</h3>
              <p>{selected.detail}</p>
              <p class="provenance"><b>Source</b> {selected.provenance}</p>
              <p class="measured-flag">
                {selected.measured ? 'Measured record' : 'Illustrative only'} ·
                position is layout, not embedding geometry
              </p>
            {:else}
              <p class="obs-empty">
                Pick a marker or a record to read its source. Selection only reads; it never runs a job.
              </p>
            {/if}
          </div>
        </div>
      {:else}
        <div id="panel-provenance" role="tabpanel" aria-labelledby="tab-provenance">
          <dl class="meta">
            <div><dt>Profile</dt><dd>{model.identity.profile}</dd></div>
            <div><dt>Anchor</dt><dd>{model.identity.anchor}</dd></div>
            <div><dt>Device</dt><dd>{model.identity.device}</dd></div>
            <div><dt>State measured</dt><dd>{model.freshness.label}</dd></div>
          </dl>
          <h3 class="panel-heading">What is measured</h3>
          <ul class="notes">
            <li>Jobs: one marker per run, from <code>GET /api/v1/jobs</code>.</li>
            <li>Actions: one marker per recorded companion action, from <code>GET /api/v1/companion/state</code>.</li>
            <li>Memory: one marker per active fact, capped at 24, from the reported <code>facts_active</code> count.</li>
          </ul>
          <h3 class="panel-heading">What is decoration</h3>
          <ul class="notes">
            {#each model.decorative as note (note)}<li>{note}</li>{/each}
            <li>Orbit speed and shell rotation carry no measurement.</li>
          </ul>
          <h3 class="panel-heading">What is not here</h3>
          <ul class="notes">
            <li>No per-fact text, embeddings or attention data: the API does not expose them.</li>
            <li>No claim that retrieval verified an answer or that weights changed.</li>
            <li>No network calls beyond this app's own same-origin API.</li>
          </ul>
          {#if error}<p class="obs-error" role="alert">{error}</p>{/if}
        </div>
      {/if}
    </aside>
  </div>
</section>

<style>
  .observatory { color: #263a32; }
  .sr-only {
    position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
    overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; border: 0;
  }
  .obs-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-bottom: 22px; }
  .obs-eyebrow { font-size: 9px; letter-spacing: 1.6px; font-weight: 800; color: #8ca78e; }
  .obs-eyebrow span { font-size: 13px; margin-right: 5px; }
  .obs-head h1 {
    font-family: Manrope, 'DM Sans', sans-serif; font-size: 30px; letter-spacing: -1.3px;
    margin: 9px 0 8px; color: #243a2d;
  }
  .obs-lede { max-width: 620px; margin: 0; font-size: 13px; line-height: 1.65; color: #7d937f; }
  .obs-freshness {
    display: flex; gap: 9px; align-items: flex-start; border: 1px solid #e2ece0; background: #f6faf3;
    border-radius: 10px; padding: 11px 13px; min-width: 210px;
  }
  .obs-freshness b { display: block; font-size: 11px; color: #4e6b53; }
  .obs-freshness small { font-size: 10px; color: #9fae9f; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #7fae7d; margin-top: 4px; flex: none; }
  .dot.stale { background: #d6b465; }
  .obs-controls {
    display: flex; flex-wrap: wrap; align-items: center; gap: 9px; padding: 11px 12px;
    border: 1px solid #e8ede6; background: #fdfefc; border-radius: 11px; margin-bottom: 14px;
  }
  .segmented { display: inline-flex; border: 1px solid #e2ebdf; border-radius: 8px; overflow: hidden; }
  .segmented button {
    border: 0; background: #fff; color: #7b927e; font-size: 11px; font-weight: 700;
    padding: 8px 13px;
  }
  .segmented button.active { background: #eaf2e9; color: #345c40; }
  .ctrl {
    border: 1px solid #e4ebe2; background: #fff; color: #54705a; border-radius: 8px;
    padding: 8px 11px; font-size: 11px; font-weight: 700;
  }
  .ctrl:disabled { opacity: 0.45; }
  .obs-hint { font-size: 10px; color: #a9b5a9; margin-left: auto; }
  .obs-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(300px, 1fr); gap: 14px; align-items: start; }
  .obs-grid.compact { grid-template-columns: minmax(0, 1.1fr) minmax(280px, 1fr); }
  .obs-stage {
    border: 1px solid #e8ede6; border-radius: 13px; background:
      radial-gradient(120% 100% at 50% 0%, #f3f9ef 0%, #eef5e8 42%, #e7f0e0 100%);
    padding: 12px; min-height: 380px; position: relative; overflow: hidden;
  }
  .scene { width: 100%; height: 420px; border-radius: 9px; }
  .scene.dim { opacity: 0.35; }
  .scene-legend {
    display: flex; flex-wrap: wrap; gap: 14px; padding: 10px 4px 0;
    font-size: 10px; color: #7f957f; letter-spacing: 0.2px;
  }
  .scene-legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .swatch.memory { background: #7fb583; }
  .swatch.event { background: #9a8eb9; }
  .swatch.job { background: #d0a45c; }
  .swatch.core { background: #cfe6c6; border: 1px solid #a9c8a1; }
  .scene-note {
    margin: 10px 4px 0; font-size: 11px; line-height: 1.55; color: #8a6a33;
    background: #fbf5e6; border: 1px solid #efe2c4; border-radius: 8px; padding: 9px 11px;
  }
  .link { border: 0; background: none; color: #6f7f3f; text-decoration: underline; font-size: 11px; padding: 0 0 0 4px; }
  .obs-notices { list-style: none; margin: 12px 0 0; padding: 0; display: grid; gap: 6px; }
  .obs-notices li {
    font-size: 10.5px; line-height: 1.55; color: #7c6a44; background: rgba(255, 253, 246, 0.86);
    border: 1px solid #ecdfc6; border-radius: 7px; padding: 8px 10px;
  }
  .obs-2d { padding: 6px 6px 0; }
  .obs-2d h2 { font-family: Manrope, sans-serif; font-size: 17px; letter-spacing: -0.5px; margin: 4px 0 8px; }
  .obs-2d > p { font-size: 12px; color: #7f9282; line-height: 1.6; margin: 0 0 14px; max-width: 560px; }
  .obs-2d table { width: 100%; border-collapse: collapse; font-size: 11.5px; }
  .obs-2d th, .obs-2d td { text-align: left; padding: 9px 10px; border-bottom: 1px solid #e6eee2; vertical-align: top; }
  .obs-2d thead th { font-size: 9px; letter-spacing: 0.8px; color: #a3b1a2; text-transform: uppercase; }
  .obs-2d tr.selected { background: #eff6ea; }
  .row-button {
    border: 0; background: none; padding: 0; text-align: left; color: #3f6147;
    font-weight: 700; font-size: 11.5px;
  }
  .obs-2d td { color: #7e917f; }
  .row-source { color: #9fae9f; font-size: 10px; }
  .obs-panel { border: 1px solid #e8ede6; border-radius: 13px; background: #fff; padding: 14px 16px 18px; }
  .panel-tabs { display: flex; gap: 4px; border-bottom: 1px solid #eef2ec; margin: -2px 0 14px; }
  .panel-tabs button {
    border: 0; background: none; padding: 9px 11px; font-size: 11px; font-weight: 700;
    color: #98a799; border-bottom: 2px solid transparent;
  }
  .panel-tabs button.active { color: #345c40; border-bottom-color: #6f9a72; }
  .status-card { background: #f8fbf5; border: 1px solid #e9f0e5; border-radius: 9px; padding: 11px 12px; }
  .badge { display: inline-block; font-size: 9px; font-weight: 800; letter-spacing: 0.5px; padding: 5px 8px; border-radius: 20px; }
  .badge.tone-amber { background: #fbf3df; color: #927335; }
  .badge.tone-lavender { background: #f0eef8; color: #7d71a1; }
  .badge.tone-green { background: #edf6ea; color: #5c8a5f; }
  .badge.tone-grey { background: #f2f4f1; color: #8b978b; }
  .status-card p { font-size: 11.5px; color: #627a66; line-height: 1.55; margin: 9px 0 6px; }
  .status-card small { font-size: 10px; color: #a6b3a6; line-height: 1.5; display: block; }
  .counts { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 13px 0 4px; }
  .counts div { background: #fafcf8; border: 1px solid #eef3ea; border-radius: 8px; padding: 9px 10px; }
  .counts dt { font-size: 9px; letter-spacing: 0.7px; color: #a4b1a4; text-transform: uppercase; }
  .counts dd { margin: 4px 0 0; font-size: 17px; font-family: Manrope, sans-serif; color: #45664c; }
  .panel-heading { font-size: 11px; letter-spacing: 0.4px; color: #4c6652; margin: 16px 0 8px; }
  .shell-group { margin-bottom: 12px; }
  .shell-group h4 {
    font-size: 9px; letter-spacing: 0.9px; text-transform: uppercase; color: #9dac9d;
    margin: 0 0 6px; display: flex; justify-content: space-between;
  }
  .shell-group h4 span { color: #bcc7ba; }
  .shell-group ul { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; max-height: 168px; overflow: auto; }
  .shell-group button {
    width: 100%; display: flex; align-items: center; gap: 8px; text-align: left;
    border: 1px solid transparent; background: #fbfdfa; border-radius: 7px; padding: 7px 8px;
    font-size: 11px; color: #5c7360;
  }
  .shell-group button.selected { background: #eff7ea; border-color: #cfe3c8; color: #37543d; }
  .shell-group button small { margin-left: auto; font-size: 9px; color: #a8b5a7; letter-spacing: 0.4px; }
  .shell-group .dot { margin-top: 0; }
  .dot.tone-lavender { background: #9a8eb9; }
  .dot.tone-amber { background: #d0a45c; }
  .dot.tone-grey { background: #b3bdb2; }
  .detail { margin-top: 14px; border-top: 1px solid #eff3ed; padding-top: 12px; }
  .detail h3 { font-size: 12.5px; margin: 0 0 7px; color: #37553e; }
  .detail p { font-size: 11px; line-height: 1.6; color: #718674; margin: 0 0 8px; }
  .detail .provenance { color: #7f917f; }
  .detail .provenance b { color: #4f6a55; margin-right: 5px; }
  .measured-flag { font-size: 10px !important; color: #9aa89a !important; }
  .obs-empty { font-size: 11.5px; line-height: 1.65; color: #97a697; margin: 8px 0; }
  .obs-error { font-size: 11px; color: #b5554d; line-height: 1.55; }
  .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 0 0 6px; }
  .meta div { background: #fafcf8; border: 1px solid #eef3ea; border-radius: 8px; padding: 9px 10px; }
  .meta dt { font-size: 9px; letter-spacing: 0.7px; color: #a4b1a4; text-transform: uppercase; }
  .meta dd { margin: 4px 0 0; font-size: 11.5px; color: #45664c; word-break: break-word; }
  .notes { margin: 0 0 4px; padding-left: 16px; display: grid; gap: 6px; }
  .notes li { font-size: 10.5px; line-height: 1.6; color: #748776; }
  .notes code { font-size: 9.5px; background: #f4f8f1; padding: 1px 4px; border-radius: 3px; color: #5d7a62; }
  @media (max-width: 960px) {
    .obs-grid, .obs-grid.compact { grid-template-columns: 1fr; }
    .obs-head { flex-direction: column; align-items: flex-start; }
    .scene { height: 320px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .obs-stage, .obs-panel, .scene { scroll-behavior: auto; }
  }
  @media (forced-colors: active) {
    .obs-stage, .obs-panel, .scene-note, .obs-notices li { border: 1px solid CanvasText; }
    .dot, .swatch { forced-color-adjust: none; }
  }
</style>
