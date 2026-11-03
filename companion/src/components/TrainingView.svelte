<script lang="ts">
  import {
    bestCheckpoint, cancelJob, createJob, getJob, jobEvents,
    listCheckpoints, selectCheckpoint
  } from '../lib/api';
  import { blockedReason, formatMetric } from '../lib/logic';
  import type { Checkpoint, Job } from '../lib/types';

  let { onChanged }: { onChanged: () => Promise<void> } = $props();

  let form = $state({
    lm_tokens: 2000, d_model: 64, n_layers: 2, d_ff: 128,
    vocab_size: 1024, seq_len: 64, batch_size: 2, grad_accum: 1,
    checkpoint_every: 2, seed: 1337
  });
  let job = $state<Job | null>(null);
  let events = $state<Array<Record<string, unknown>>>([]);
  let checkpoints = $state<Checkpoint[]>([]);
  let best = $state<Checkpoint | null>(null);
  let blocked = $state<string | null>(null);
  let error = $state('');
  let selectionNotice = $state('');
  let stopEvents: (() => void) | null = null;
  let pollTimer: number | undefined;

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

  function watch(id: string): void {
    if (stopEvents) stopEvents();
    stopEvents = jobEvents(id, (event) => {
      events = [...events.slice(-200), event];
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

  async function start(): Promise<void> {
    error = '';
    events = [];
    try {
      job = await createJob({ ...form });
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
        <input id={`field-${key}`} type="number" bind:value={form[key as keyof typeof form]} />
      </div>
    {/each}
  </div>
  <div class="row" style="margin-top:12px">
    <button class="btn primary" onclick={() => void start()}>Start training</button>
    <button class="btn" onclick={() => void cancel()} disabled={!job || ['done', 'failed', 'cancelled'].includes(job.status)}>Cancel</button>
    <button class="btn" onclick={() => void refresh()}>Refresh checkpoints</button>
  </div>
  {#if selectionNotice}<p class="muted" role="status">{selectionNotice}</p>{/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}
</div>

<div class="panel">
  <h2>Live job</h2>
  {#if !job}
    <p class="empty">No job started.</p>
  {:else}
    <p class="muted">
      id={job.id} · status={job.status} · progress={formatMetric(job.progress)}
      {#if job.result?.out_dir}· out_dir={String(job.result.out_dir)}{/if}
    </p>
    <div class="meter"><div style={`width:${Math.round((job.progress || 0) * 100)}%`}></div></div>
    {#if job.result?.manifest}
      {@const manifest = job.result.manifest as Record<string, unknown>}
      <p class="muted">
        checkpoint manifest written: {String(manifest.written)} ·
        held-out BPB: {formatMetric(manifest.held_out_bpb)}
      </p>
    {/if}
  {/if}
  <pre aria-live="polite">{events.slice(-25).map((event) =>
    event.type === 'log' || event.type === 'notice'
      ? String(event.line)
      : `${event.type}: ${JSON.stringify(event.metrics ?? event.manifest ?? {})}`
  ).join('\n') || 'No events yet.'}</pre>
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
        <span class="gate" class:pass={outcome === 'passed'} class:none={outcome !== 'passed'}>{gate}: {outcome}</span>
      {/each}
    </div>
    <button class="btn primary" onclick={() => void select(best!.name)}>Use this model</button>
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
        <li>
          <code>{row.name}</code> — {row.eligible ? 'eligible' : `ineligible: ${row.reason}`}
          {#if row.eligible}
            <button class="btn" style="padding:2px 8px;margin-left:8px" onclick={() => void select(row.name)}>Select</button>
          {/if}
        </li>
      {/each}
    </ul>
  {/if}
</div>
