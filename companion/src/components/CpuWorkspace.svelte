<script lang="ts">
  import { createJob, getJob, listJobs, cancelJob, getHealth, nativeGenerate, getLearning, configureLearning, nativeMemoryGenerate } from '../lib/api';
  import type { Job, Health } from '../lib/types';
  import LearningConstellation from './LearningConstellation.svelte';
  import type { LearningStatus } from '../lib/api';
  let { openCompanion, openAdvanced }: { openCompanion: () => void; openAdvanced: () => void } = $props();
  let threads = $state(1);
  let job = $state<Job | null>(null);
  let jobs = $state<Job[]>([]);
  let health = $state<Health | null>(null);
  let learning = $state<LearningStatus | null>(null);
  let error = $state('');
  let busy = $state(false);
  let prompt = $state('Once upon a time');
  let generated = $state<string | null>(null);
  let generating = $state(false);
  let evidence = $state<{ id?: number; text: string; source?: string; source_url?: string; retrieval_store?: string; source_cluster_ids?: number[] }[]>([]);
  let sampleNote = $state('');
  let section = $state<'overview' | 'training' | 'datasets' | 'evaluation'>('overview');
  let report = $derived((job?.result?.report ?? {}) as Record<string, unknown>);
  let logs = $derived(job?.log_tail ?? []);
  let pct = $derived(Math.round(Math.min(1, Math.max(0, job?.progress ?? 0)) * 100));
  let recentLoss = $derived.by(() => {
    for (let i = logs.length - 1; i >= 0; i--) {
      const found = logs[i].match(/\bloss=([\d.]+)/);
      if (found) return Number(found[1]).toFixed(3);
    }
    return typeof job?.result?.loss === 'number' ? job.result.loss.toFixed(3) : '—';
  });
  async function refresh() {
    try {
      const [h, list, learningStatus] = await Promise.all([getHealth(), listJobs(), getLearning()]);
      health = h; learning = learningStatus; jobs = list.jobs.filter(j => j.kind === 'train').slice().reverse();
      if (job) job = await getJob(job.id);
      else if (jobs.length) job = await getJob(jobs[0].id);
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
  }
  async function setLearning(enabled: boolean) {
    error = '';
    try { learning = await configureLearning(enabled, learning?.threads ?? 1); }
    catch (e) { error = e instanceof Error ? e.message : String(e); }
  }
  async function start() {
    error = ''; busy = true;
    try {
      job = await createJob({dataset:'cpu_smoke', core:'C4', lm_tokens:512, threads,
        d_model:16, n_layers:1, n_heads:2, d_ff:32, vocab_size:256,
        seq_len:16, batch_size:1, grad_accum:1, checkpoint_every:100, seed:1337});
      await refresh();
    } catch(e) { error = e instanceof Error ? e.message : String(e); }
    finally { busy = false; }
  }
  async function generate() {
    if (!job) return; generating = true; error = ''; generated = null; evidence = []; sampleNote = 'No memory was used.';
    try { const result = await nativeGenerate(job.id, prompt); generated = result.response; }
    catch(e) { error = e instanceof Error ? e.message : String(e); }
    finally { generating = false; }
  }
  async function generateWithMemory() {
    if (!job) return; generating = true; error = ''; generated = null; evidence = []; sampleNote = '';
    try {
      const result = await nativeMemoryGenerate(job.id, prompt);
      generated = result.response; evidence = result.evidence;
      sampleNote = result.memory_used ? 'Retrieved evidence was included; model weights were not updated. ' + result.warning : 'No fitting evidence was found, so generation was skipped.';
    } catch (e) { error = e instanceof Error ? e.message : String(e); }
    finally { generating = false; }
  }
  async function cancel() {
    if (!job) return;
    try { await cancelJob(job.id); await refresh(); }
    catch(e) { error = e instanceof Error ? e.message : String(e); }
  }
  $effect(() => { void refresh(); const timer = setInterval(() => void refresh(), 2000); return () => clearInterval(timer); });
</script>

<div class="workspace">
  <aside class="sidebar">
    <div class="identity"><div class="logo-mark">✳</div><div><b>recamem</b><small>LEARNING STUDIO</small></div></div>
    <div class="side-label">WORKSPACE</div>
    <div class="side-nav">
      <button class:active={section === 'overview'} onclick={() => section = 'overview'}><span>◫</span> Overview</button>
      <button class:active={section === 'training'} onclick={() => section = 'training'}><span>◉</span> Training lab</button>
      <button class:active={section === 'datasets'} onclick={() => section = 'datasets'}><span>▤</span> Datasets</button>
      <button class:active={section === 'evaluation'} onclick={() => section = 'evaluation'}><span>◈</span> Evaluation</button>
      <button onclick={openCompanion}><span>✧</span> Memory & companion <span class="arrow">↗</span></button>
      <button onclick={openAdvanced}><span>⚙</span> Advanced training <span class="arrow">↗</span></button>
    </div>
    <div class="side-bottom"><div class="side-cpu"><span class="cpu-icon">▦</span><div><b>Local workspace</b><small>CPU execution</small></div><span class="online-dot"></span></div><p>Built for learning, one<br />step at a time.</p></div>
  </aside>
  <div class="workspace-main">
    <header class="workspace-header"><div class="breadcrumb">Workspace <span>/</span> <strong>{section === 'overview' ? 'Overview' : section === 'training' ? 'Training lab' : section === 'datasets' ? 'Datasets' : 'Evaluation'}</strong></div><div class="header-right"><span class="local-pill"><i></i> CPU ONLY</span><span class="header-divider"></span><span class="header-note">Your local environment</span><span class="avatar">R</span></div></header>
    <main class="workspace-content">
      {#if section === 'overview' || section === 'training'}
        <div class="page-title"><div><div class="eyebrow"><span>✳</span> YOUR LEARNING WORKSPACE</div><h1>{section === 'training' ? 'Training lab' : 'A little smarter, step by step.'}</h1><p>Train a small model, follow its progress, and build on what it learns.</p></div><button class="outline-button" onclick={openCompanion}>Open companion <span>↗</span></button></div>
        <div class="hero"><div class="hero-copy"><div class="hero-kicker"><span>✦</span> BUILT FOR CURIOSITY</div><h2>Big ideas start<br />with small models<span>.</span></h2><p>Start with an automatic CPU smoke-test configuration. Continuous background learning is not available yet.</p><button onclick={() => document.getElementById('configure')?.scrollIntoView({behavior:'smooth'})}>Configure a run <span>↗</span></button></div><div class="hero-art" aria-hidden="true"><LearningConstellation /><div class="art-tag">AN ILLUSTRATION <span>✳</span></div></div></div>
        <div class="section-heading"><div><span class="section-index">01 / YOUR EXPERIMENT</span><h2>Let's get learning</h2><p>Everything you need to start a local training run.</p></div><span class="subtle-label"><span class="tiny-dot"></span> RUNS LOCALLY</span></div>
        <section class="surface learning-status" aria-labelledby="learning-title">
          <span class="badge-soft">{learning ? learning.status.toUpperCase() : 'STATUS UNAVAILABLE'}</span>
          <div class="learning-status-heading"><div><span class="section-index">CONTINUOUS LEARNING</span><h3 id="learning-title">Background learning</h3></div></div>
          <p>{learning?.reason ?? 'Could not load background learning status.'} No automatic model training is running.</p>
          <p>The saved preference does not start a trainer yet. Source authorization, evaluation and safe model promotion are still being implemented.</p>
          {#if learning}<button class="learning-control" onclick={() => setLearning(!learning?.enabled)}>{learning.enabled ? 'Pause future learning' : 'Allow future learning'}</button>{/if}
          {#if error}<p class="workspace-error" role="alert">{error}</p>{/if}
        </section>
        <div class="work-grid">
          <div class="main-column">
            <section class="surface configure" id="configure"><div class="card-heading"><div class="card-icon green">◉</div><div><h3>Configure training</h3><p>Use automatic defaults for this temporary smoke test.</p></div><span class="card-number">01</span></div><div class="divider"></div><div class="config-row"><div class="field-block"><label for="threads">CPU THREADS</label><input id="threads" type="number" min="1" max="32" bind:value={threads} /><span>Resource limit for this manual run.</span></div></div><div class="config-footer"><div class="mini-info"><span>✳</span><div><b>Ready to experiment</b><small>Uses C4, byte tokens and a fixed 512-token smoke budget automatically; this is not continuous learning.</small></div></div><button class="start-button" onclick={start} disabled={busy || (job?.status === 'running') || threads < 1 || threads > 32}>{busy ? 'Starting…' : 'Start training'} <span>↗</span></button></div>{#if error}<p class="workspace-error" role="alert">{error}</p>{/if}</section>
            <section class="surface dataset-card"><div class="card-heading"><div class="card-icon lavender">▤</div><div><h3>Training data</h3><p>Know exactly what your model learns from.</p></div><span class="badge-soft">SMOKE TEST</span></div><div class="dataset-details"><div class="dataset-file"><div class="file-icon">▤</div><div><b>CPU smoke corpus</b><small>Self-authored · MIT license · byte tokenizer</small></div></div><div class="dataset-stat"><b>515</b><small>TRAIN BYTES</small></div><div class="dataset-stat"><b>124</b><small>VALIDATION BYTES</small></div><span class="dataset-status">● &nbsp; Available</span></div><p class="data-note"><b>Just a starting point.</b> This tiny dataset tests the pipeline only; it cannot establish language understanding. A substantive, verified corpus is still needed.</p></section>
          </div>
          <section class="surface monitor"><div class="monitor-heading"><div><span class="section-index">LIVE MONITOR</span><h3>Training activity</h3></div><span class="activity-dot">●</span></div><p class="monitor-description">Your run, in real time.</p><div class="monitor-line"></div>{#if job}<div class="run-state"><div><span class="running-indicator"></span><b>{job.status === 'running' ? 'Training in progress' : `Run ${job.status}`}</b></div><span class="run-chip">{job.status.toUpperCase()}</span></div><p class="job-id">RUN {job.id.slice(0, 12)}</p><div class="progress-meta"><span>Token progress</span><b>{pct}%</b></div><div class="progress-track"><div style={`width:${pct}%`}></div></div><div class="monitor-numbers"><div><small>TOKENS SEEN</small><b>{(job.tokens_seen ?? 0).toLocaleString()}</b></div><div><small>LATEST LOSS</small><b>{recentLoss}</b></div></div>{#if job.status === 'running'}<button class="cancel-button" onclick={cancel}>Stop run</button>{/if}{#if report.final_val_loss !== undefined}<div class="result-line"><span>Held-out loss</span><b>{Number(report.final_val_loss).toFixed(3)}</b></div>{/if}{#if job.error}<p class="workspace-error">{job.error}</p>{/if}{:else}<div class="empty-monitor"><div class="empty-graphic"><span>◌</span><i>✳</i></div><b>A quiet place to start</b><p>Start a training run and your live progress will appear here.</p></div>{/if}<div class="monitor-foot"><span>◷</span> Updates from your local training process</div></section>
        </div>
        <div class="section-heading recent-heading"><div><span class="section-index">02 / YOUR PROGRESS</span><h2>Recent runs</h2><p>Every experiment is a step forward.</p></div></div><section class="surface recent-card"><div class="recent-head"><span>EXPERIMENT</span><span>STATUS</span><span>TOKENS</span><span>CHECKPOINT</span></div>{#if jobs.length}{#each jobs.slice(0,5) as row}<button class="recent-row" onclick={async () => {job = await getJob(row.id)}}><span><span class="run-icon">◉</span> <b>{String(row.params?.core ?? 'Training run')}</b><small>{row.id.slice(0, 10)}</small></span><span><span class="status-label">{row.status}</span></span><span>{(row.tokens_seen ?? 0).toLocaleString()}</span><span>{row.checkpoint_path ? 'Saved' : '—'} &nbsp; ↗</span></button>{/each}{:else}<div class="recent-empty"><span>◷</span><div><b>No runs yet</b><p>Your training history will show up here after your first experiment.</p></div></div>{/if}</section>
      {:else if section === 'datasets'}
        <div class="page-title simple"><div><div class="eyebrow"><span>✳</span> DATA LIBRARY</div><h1>Start with good data.</h1><p>See what is available and where it comes from.</p></div></div><section class="surface large-info"><div class="card-icon lavender">▤</div><span class="badge-soft">SMOKE TEST ONLY</span><h2>CPU smoke corpus</h2><p>A small, self-authored collection of short sentences. It is included with RecaMem under the MIT license and is prepared automatically when you start a smoke run.</p><div class="info-grid"><div><small>TRAIN</small><b>515 bytes</b></div><div><small>VALIDATION</small><b>124 bytes</b></div><div><small>TOKENIZER</small><b>UTF-8 bytes</b></div></div><div class="notice-box">This fixture tests the training pipeline. It is far too small to demonstrate general language understanding. A substantive dataset is not configured.</div><button class="start-button" onclick={() => section = 'training'}>Go to training <span>↗</span></button></section>
      {:else}
        <div class="page-title simple"><div><div class="eyebrow"><span>✳</span> EVALUATION</div><h1>Measure what matters.</h1><p>Evidence, not assumptions, about your model.</p></div></div><section class="surface large-info"><div class="card-icon green">◈</div><span class="badge-soft">EXPERIMENTAL</span><h2>Evaluation results</h2><p>Held-out loss is reported when a training run completes. A smoke dataset does not measure general understanding or emergent abilities.</p>{#if report.final_val_loss !== undefined}<div class="info-grid"><div><small>HELD-OUT LOSS</small><b>{Number(report.final_val_loss).toFixed(3)}</b></div><div><small>BITS PER BYTE</small><b>{report.val_bpb == null ? 'Unavailable' : Number(report.val_bpb).toFixed(3)}</b></div><div><small>RUN STATUS</small><b>{job?.status}</b></div></div>{:else}<div class="notice-box">No completed evaluation yet. Start a run to see its measured results.</div>{/if}<p>Baseline comparisons, long-term retention, and real-world recall evaluations are not yet available in this workspace.</p><div class="try-model"><h3>Try the native checkpoint</h3><p>Generated text is likely nonsensical after a smoke run. The standard sample does not use memory; the second button is experimental.</p><label for="try-prompt">PROMPT</label><input id="try-prompt" type="text" bind:value={prompt} /><button class="start-button" onclick={generate} disabled={generating || !job || job.status !== 'done' || !prompt.trim()}>{generating ? 'Generating…' : 'Generate sample'} ↗</button><button class="learning-control" onclick={generateWithMemory} disabled={generating || !job || job.status !== 'done' || !prompt.trim()}>Try with memory ↗</button><p>Experimental: requires a configured memory anchor. Retrieved text is untrusted; the generated answer may be inaccurate.</p>{#if generated !== null}<div class="sample-output"><small>MODEL OUTPUT</small><p>{generated || '(empty output)'}</p><small>{sampleNote}</small>{#if evidence.length}<div class="sample-evidence"><small>INCLUDED MEMORY EVIDENCE</small>{#each evidence as fact}<p>{fact.text} <small>· {fact.source ?? 'source unavailable'}{fact.retrieval_store ? ` · ${fact.retrieval_store}` : ''}{fact.source_cluster_ids?.length ? ` · linked fact IDs: ${fact.source_cluster_ids.join(', ')}` : ''}{fact.source_url ? ` · ${fact.source_url}` : ''}</small></p>{/each}</div>{/if}</div>{/if}{#if error}<p class="workspace-error" role="alert">{error}</p>{/if}</div><button class="start-button" onclick={() => section = 'training'}>Open training lab <span>↗</span></button></section>
      {/if}
      <footer class="workspace-footer"><span>✳ &nbsp; RECAMEM LEARNING STUDIO</span><span>Small steps. Real progress.</span><span>{health ? 'CPU · LOCAL' : 'CONNECTING TO API'}</span></footer>
    </main>
  </div>
</div>
