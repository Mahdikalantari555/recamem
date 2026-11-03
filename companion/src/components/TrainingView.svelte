<script lang="ts">
  import {
    bestCheckpoint, cancelJob, createJob, getJob, hubPull, hubPush,
    jobEvents, listCheckpoints, selectCheckpoint
  } from '../lib/api';
  import { blockedReason, formatGateStatus, formatMetric } from '../lib/logic';
  import type { Checkpoint, Job } from '../lib/types';

  let { onChanged }: { onChanged: () => Promise<void> } = $props();

  let form = $state({
    core: 'C4',
    lm_tokens: 2000,
    d_model: 64,
    n_layers: 2,
    n_heads: 4,
    d_ff: 128,
    vocab_size: 1024,
    seq_len: 64,
    lr: 0.003,
    batch_size: 2,
    grad_accum: 1,
    checkpoint_every: 2,
    seed: 1337,
    resume: ''
  });

  let job = $state<Job | null>(null);
  let events = $state<Array<Record<string, unknown>>>([]);
  let metricsHistory = $state<Array<{ step: number; tokens: number; loss?: number; bpb?: number }>>([]);
  let checkpoints = $state<Checkpoint[]>([]);
  let best = $state<Checkpoint | null>(null);
  let blocked = $state<string | null>(null);
  let error = $state('');
  let selectionNotice = $state('');
  let stopEvents: (() => void) | null = null;
  let pollTimer: number | undefined;

  // Hub push/pull state
  let hubRepoId = $state('');
  let hubCheckpointName = $state('');
  let hubNotice = $state('');
  let hubError = $state('');
  let hubBusy = $state(false);

  async function refresh(): Promise<void> {
    try {
      const list = await listCheckpoints();
      checkpoints = list.checkpoints;
      const answer = await bestCheckpoint();
      best = answer.checkpoint ?? null;
      blocked = blockedReason(answer);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function recordMetric(event: Record<string, unknown>): void {
    if (event.type === 'metrics') {
      const tokens = typeof event.tokens_seen === 'number'
        ? event.tokens_seen
        : (metricsHistory.length + 1) * 100;
      const loss = typeof event.loss === 'number' ? event.loss : undefined;
      const bpb = typeof event.held_out_bpb === 'number' ? event.held_out_bpb : undefined;
      if (loss !== undefined || bpb !== undefined) {
        metricsHistory = [...metricsHistory, { step: metricsHistory.length + 1, tokens, loss, bpb }];
      }
    }
  }

  function watch(id: string): void {
    if (stopEvents) stopEvents();
    metricsHistory = [];
    stopEvents = jobEvents(id, (event) => {
      events = [...events.slice(-200), event];
      recordMetric(event);
      if (event.type === 'finished' || event.type === 'cancelled') void refresh();
    });
    window.clearInterval(pollTimer);
    pollTimer = window.setInterval(async () => {
      try {
        job = await getJob(id);
        if (['done', 'failed', 'cancelled'].includes(job.status)) {
          window.clearInterval(pollTimer);
          stopEvents?.();
          stopEvents = null;
          void refresh();
          await onChanged();
        }
      } catch (cause) {
        error = cause instanceof Error ? cause.message : String(cause);
      }
    }, 1500);
  }

  async function start(isResume = false): Promise<void> {
    error = '';
    events = [];
    metricsHistory = [];
    try {
      const params: Record<string, unknown> = { ...form };
      if (!isResume || !params.resume) {
        delete params.resume;
      }
      job = await createJob(params);
      watch(job.id);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function cancel(): Promise<void> {
    if (!job) return;
    try {
      await cancelJob(job.id);
      job = await getJob(job.id);
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function select(name: string): Promise<void> {
    error = '';
    try {
      const outcome = await selectCheckpoint(name);
      selectionNotice = outcome.applied_to_generation
        ? `Checkpoint ${outcome.selected} is active for generation.`
        : `Recorded ${outcome.selected} for provenance only. ${outcome.reason}`;
      await refresh();
      await onChanged();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  function setResumeCheckpoint(name: string): void {
    const ckpt = checkpoints.find((c) => c.name === name);
    if (ckpt) {
      form.resume = ckpt.path;
    }
  }

  async function onHubPush(): Promise<void> {
    hubError = '';
    hubNotice = '';
    const repo = hubRepoId.trim();
    if (!repo) {
      hubError = 'Please specify a repository ID (e.g. username/model).';
      return;
    }
    const ckpt = hubCheckpointName || (best ? best.name : (checkpoints[0]?.name ?? ''));
    if (!ckpt) {
      hubError = 'No checkpoint selected to push.';
      return;
    }
    const confirmed = window.confirm(
      `Confirm push: upload checkpoint "${ckpt}" to Hugging Face repo "${repo}"?\n\nAuthentication uses server-side HF_TOKEN.`
    );
    if (!confirmed) return;
    hubBusy = true;
    try {
      const res = await hubPush(repo, ckpt, true);
      hubNotice = `Successfully pushed ${res.checkpoint} to ${res.repo_id}${res.url ? ` (${res.url})` : ''}.`;
    } catch (err) {
      hubError = err instanceof Error ? err.message : String(err);
    } finally {
      hubBusy = false;
    }
  }

  async function onHubPull(): Promise<void> {
    hubError = '';
    hubNotice = '';
    const repo = hubRepoId.trim();
    if (!repo) {
      hubError = 'Please specify a repository ID (e.g. username/model).';
      return;
    }
    const confirmed = window.confirm(
      `Confirm pull: download model from Hugging Face repo "${repo}" to local checkpoints?\n\nAuthentication uses server-side HF_TOKEN.`
    );
    if (!confirmed) return;
    hubBusy = true;
    try {
      const res = await hubPull(repo, undefined, true);
      hubNotice = `Successfully pulled ${res.repo_id} to ${res.checkpoint}.`;
      await refresh();
      await onChanged();
    } catch (err) {
      hubError = err instanceof Error ? err.message : String(err);
    } finally {
      hubBusy = false;
    }
  }

  // Derived chart metrics
  const chartData = $derived.by(() => {
    if (metricsHistory.length < 2) return null;
    const tokens = metricsHistory.map((m) => m.tokens);
    const minTokens = Math.min(...tokens);
    const maxTokens = Math.max(...tokens);
    const tokenRange = maxTokens - minTokens || 1;

    const values = metricsHistory
      .flatMap((m) => [m.loss, m.bpb])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    if (values.length === 0) return null;

    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const valRange = maxVal - minVal || 1;

    const width = 480;
    const height = 150;
    const padX = 45;
    const padY = 20;
    const innerW = width - padX - 15;
    const innerH = height - padY - 25;

    const lossPoints = metricsHistory
      .filter((m): m is typeof m & { loss: number } => typeof m.loss === 'number')
      .map((m) => {
        const x = padX + ((m.tokens - minTokens) / tokenRange) * innerW;
        const y = padY + (1 - (m.loss - minVal) / valRange) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    const bpbPoints = metricsHistory
      .filter((m): m is typeof m & { bpb: number } => typeof m.bpb === 'number')
      .map((m) => {
        const x = padX + ((m.tokens - minTokens) / tokenRange) * innerW;
        const y = padY + (1 - (m.bpb - minVal) / valRange) * innerH;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');

    return {
      width, height, padX, padY, innerW, innerH,
      minTokens, maxTokens, minVal, maxVal,
      lossPoints, bpbPoints
    };
  });

  // Report & baselines from job result
  const reportData = $derived.by(() => {
    const res = job?.result as Record<string, unknown> | undefined;
    if (!res) return null;
    const rep = (res.report ?? {}) as Record<string, unknown>;
    const thresholds = (rep.ngram_thresholds ?? (res.manifest as Record<string, unknown> | undefined)?.ngram_thresholds ?? {}) as Record<string, number>;
    const gates = (rep.gates ?? (res.manifest as Record<string, unknown> | undefined)?.gates ?? {}) as Record<string, unknown>;
    const samples = (Array.isArray(res.samples) ? res.samples : rep.samples ?? []) as string[];
    return { thresholds, gates, samples };
  });

  $effect(() => {
    void refresh();
    return () => {
      window.clearInterval(pollTimer);
      if (stopEvents) stopEvents();
    };
  });
</script>

<div class="panel">
  <h2>Train a model</h2>
  <p class="muted">
    A training job launches <code>scripts/train_lm.py</code> as a real subprocess.
    Progress comes from parsed tokens/log output — never from elapsed time. With no
    real corpus, the server uses a deterministic fixture corpus and says so; that
    run is a pipeline smoke test, not a language-learning result.
  </p>
  <div class="row">
    {#each Object.entries(form) as [key, value] (key)}
      <div class="field">
        <label for={`field-${key}`}>{key}</label>
        {#if key === 'core'}
          <input id={`field-${key}`} type="text" bind:value={form.core} />
        {:else if key === 'resume'}
          <input id={`field-${key}`} type="text" placeholder="checkpoint path or name" bind:value={form.resume} />
        {:else}
          <input id={`field-${key}`} type="number" step={key === 'lr' ? '0.0001' : '1'} bind:value={form[key as keyof typeof form]} />
        {/if}
      </div>
    {/each}
  </div>
  <div class="row" style="margin-top:12px">
    <button class="btn primary" onclick={() => void start(false)}>Start training</button>
    <button class="btn" onclick={() => void start(true)} disabled={!form.resume}>Resume training</button>
    <button class="btn" onclick={() => void cancel()} disabled={!job || ['done', 'failed', 'cancelled'].includes(job.status)}>Cancel</button>
    <button class="btn" onclick={() => void refresh()}>Refresh checkpoints</button>
  </div>
  {#if selectionNotice}<p class="muted" role="status">{selectionNotice}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<div class="panel">
  <h2>Live job & Metrics</h2>
  {#if !job}
    <p class="empty">No job started.</p>
  {:else}
    <p class="muted">
      id={job.id} · status={job.status} · progress={formatMetric(job.progress)}
      {#if typeof job.elapsed === 'number'}· elapsed={job.elapsed}s{/if}
      {#if typeof job.tokens_seen === 'number'}· tokens={job.tokens_seen.toLocaleString()}{/if}
      {#if job.checkpoint_path}· checkpoint={job.checkpoint_path}{/if}
    </p>
    <div class="meter"><div style={`width:${Math.round((job.progress || 0) * 100)}%`}></div></div>

    <!-- Live Loss / BPB Chart -->
    <div style="margin: 14px 0;">
      <h3 style="font-size:12px;color:var(--muted);text-transform:uppercase;margin:0 0 6px">Live Loss & BPB Curve</h3>
      {#if chartData}
        <div style="background:#0b0e14;border:1px solid var(--line);border-radius:8px;padding:8px 12px;overflow-x:auto;">
          <svg width={chartData.width} height={chartData.height} style="display:block;max-width:100%;">
            <!-- Axis lines -->
            <line x1={chartData.padX} y1={chartData.padY} x2={chartData.padX} y2={chartData.height - 25} stroke="#2b3442" stroke-width="1" />
            <line x1={chartData.padX} y1={chartData.height - 25} x2={chartData.width - 15} y2={chartData.height - 25} stroke="#2b3442" stroke-width="1" />

            <!-- Y min/max labels -->
            <text x={chartData.padX - 6} y={chartData.padY + 8} fill="#9aa7b8" font-size="10" text-anchor="end">{chartData.maxVal.toFixed(2)}</text>
            <text x={chartData.padX - 6} y={chartData.height - 25} fill="#9aa7b8" font-size="10" text-anchor="end">{chartData.minVal.toFixed(2)}</text>

            <!-- X tokens labels -->
            <text x={chartData.padX} y={chartData.height - 8} fill="#9aa7b8" font-size="10">{chartData.minTokens}</text>
            <text x={chartData.width - 15} y={chartData.height - 8} fill="#9aa7b8" font-size="10" text-anchor="end">{chartData.maxTokens} tok</text>

            <!-- Loss line (blue) -->
            {#if chartData.lossPoints}
              <polyline fill="none" stroke="#7cc4ff" stroke-width="2" points={chartData.lossPoints} />
            {/if}

            <!-- BPB line (green) -->
            {#if chartData.bpbPoints}
              <polyline fill="none" stroke="#b5f36d" stroke-width="2" points={chartData.bpbPoints} />
            {/if}
          </svg>
          <div class="row" style="margin-top:6px;font-size:11px;color:var(--muted);gap:14px">
            <span style="color:#7cc4ff">■ Loss</span>
            <span style="color:#b5f36d">■ Held-out BPB</span>
          </div>
        </div>
      {:else}
        <p class="empty" style="font-size:12px;margin:4px 0">Waiting for training metric steps to render chart...</p>
      {/if}
    </div>

    <!-- N-gram baselines & G0-G4 Badges -->
    {#if reportData}
      <div style="margin: 12px 0;">
        <h3 style="font-size:12px;color:var(--muted);text-transform:uppercase;margin:0 0 6px">Gate Outcomes (G0–G4)</h3>
        <div class="row" style="margin-bottom:8px">
          {#each ['G0', 'G1', 'G2', 'G3', 'G4'] as gateKey}
            {@const outcome = formatGateStatus(reportData.gates[gateKey])}
            <span class="gate" class:pass={outcome === 'passed'} class:none={outcome === 'not evaluated'}>
              {gateKey}: {outcome}
            </span>
          {/each}
        </div>

        {#if Object.keys(reportData.thresholds).length > 0}
          <h3 style="font-size:12px;color:var(--muted);text-transform:uppercase;margin:10px 0 6px">N-Gram Baselines / Thresholds</h3>
          <div class="row" style="gap:12px;font-size:12px;color:var(--muted)">
            {#each Object.entries(reportData.thresholds) as [k, thr]}
              <span><strong>{k} threshold:</strong> {formatMetric(thr)}</span>
            {/each}
          </div>
        {/if}

        {#if reportData.samples.length > 0}
          <h3 style="font-size:12px;color:var(--muted);text-transform:uppercase;margin:12px 0 6px">Generated Samples</h3>
          <div style="display:flex;flex-direction:column;gap:6px;max-height:180px;overflow-y:auto;">
            {#each reportData.samples as sample}
              <pre style="margin:0;padding:6px 10px;font-size:11px;background:#0d1117;"><code>{sample}</code></pre>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  {/if}

  <h3 style="font-size:12px;color:var(--muted);text-transform:uppercase;margin:12px 0 6px">Events & Output Tail</h3>
  <pre aria-live="polite">{events.slice(-25).map((event) =>
    event.type === 'log' || event.type === 'notice'
      ? String(event.line)
      : `${event.type}: ${JSON.stringify(event.metrics ?? event.manifest ?? {})}`
  ).join('\n') || (job?.log_tail && job.log_tail.length > 0 ? job.log_tail.slice(-25).join('\n') : 'No events yet.')}</pre>
</div>

<!-- Hugging Face Hub push/pull -->
<div class="panel">
  <h2>Hugging Face Hub</h2>
  <p class="muted">
    Push or pull checkpoints to/from the Hugging Face Hub. Hub authentication is handled
    server-side only via <code>HF_TOKEN</code> or a prior <code>huggingface-cli login</code>;
    no token is ever sent from or stored in the browser.
  </p>
  <div class="row" style="margin-bottom:10px">
    <div class="field" style="flex:2">
      <label for="hub-repo-id">Hub Repository ID (e.g. username/repo-name)</label>
      <input id="hub-repo-id" type="text" placeholder="username/recamem-model" bind:value={hubRepoId} />
    </div>
    <div class="field" style="flex:1">
      <label for="hub-ckpt-select">Checkpoint to Push</label>
      <select id="hub-ckpt-select" bind:value={hubCheckpointName}>
        {#if checkpoints.length === 0}
          <option value="">No local checkpoints</option>
        {:else}
          {#each checkpoints as ckpt}
            <option value={ckpt.name}>{ckpt.name} {ckpt.eligible ? '(eligible)' : ''}</option>
          {/each}
        {/if}
      </select>
    </div>
  </div>
  <div class="row">
    <button class="btn primary" onclick={() => void onHubPush()} disabled={!hubRepoId.trim() || hubBusy}>
      {hubBusy ? 'Processing...' : 'Push to Hub'}
    </button>
    <button class="btn" onclick={() => void onHubPull()} disabled={!hubRepoId.trim() || hubBusy}>
      {hubBusy ? 'Processing...' : 'Pull from Hub'}
    </button>
  </div>
  {#if hubNotice}<p class="muted" role="status" style="margin-top:8px">{hubNotice}</p>{/if}
  {#if hubError}<p class="error" role="alert" style="margin-top:8px">{hubError}</p>{/if}
</div>

<div class="panel">
  <h2>Best checkpoint</h2>
  {#if best}
    <p>
      <strong>{best.name}</strong> — BPB {formatMetric(best.held_out_bpb)} ·
      dataset {best.dataset ?? 'unknown'} · seed {String(best.seed ?? 'unknown')}
    </p>
    <div class="row">
      {#each Object.entries(best.gates ?? {}) as [gate, outcome] (gate)}
        {@const status = formatGateStatus(outcome)}
        <span class="gate" class:pass={status === 'passed'} class:none={status !== 'passed'}>
          {gate}: {status}
        </span>
      {/each}
    </div>
    <div class="row" style="margin-top:10px">
      <button class="btn primary" onclick={() => void select(best!.name)}>Use this model</button>
      <button class="btn" onclick={() => setResumeCheckpoint(best!.name)}>Resume from this</button>
    </div>
  {:else}
    <p class="badge warn">demo / recall-only — {blocked ?? 'no eligible checkpoint'}</p>
    <p class="muted">
      An eligible checkpoint needs a sha256 manifest, config, seed, dataset,
      commit, command, a measured held-out BPB and a G2 result. Nothing is
      silently loaded from random-init weights.
    </p>
  {/if}
  <h3 style="font-size:13px;margin-top:16px">All checkpoints</h3>
  {#if checkpoints.length === 0}
    <p class="empty">No checkpoint directories yet.</p>
  {:else}
    <ul class="muted">
      {#each checkpoints as row (row.name)}
        <li style="margin-bottom:6px">
          <code>{row.name}</code> — {row.eligible ? 'eligible' : `ineligible: ${row.reason}`}
          {#if row.eligible}
            <button class="btn" style="padding:2px 8px;margin-left:8px" onclick={() => void select(row.name)}>Select</button>
          {/if}
          <button class="btn" style="padding:2px 8px;margin-left:4px" onclick={() => setResumeCheckpoint(row.name)}>Resume</button>
        </li>
      {/each}
    </ul>
  {/if}
</div>
