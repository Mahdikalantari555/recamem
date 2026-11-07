<script lang="ts">
  import CpuWorkspace from './components/CpuWorkspace.svelte';
  import ChatPanel from './components/ChatPanel.svelte';
  import Pet from './components/Pet.svelte';
  import TrainingView from './components/TrainingView.svelte';
  import { getState } from './lib/api';
  import type { CompanionState, Pet as PetState } from './lib/types';
  let view = $state<'workspace' | 'companion' | 'advanced'>('workspace');
  let engineState = $state<CompanionState | null>(null);
  let pet = $state<PetState | null>(null);
  let factId = $state<number | null>(null);
  let error = $state('');
  async function refresh() {
    try { const snapshot = await getState(); engineState = snapshot.state; pet = snapshot.pet; factId = snapshot.last_fact_id; error = ''; }
    catch(e) { error = e instanceof Error ? e.message : String(e); }
  }
  $effect(() => { if(view === 'companion') void refresh(); });
</script>
<a class="skip-link" href="#main">Skip to content</a>
{#if view === 'workspace'}
  <CpuWorkspace openCompanion={() => view = 'companion'} openAdvanced={() => view = 'advanced'} />
{:else if view === 'advanced'}
  <div class="companion-page" id="main"><button class="back-button" onclick={() => view = 'workspace'}>← Back to learning studio</button><div class="companion-title"><div class="eyebrow">✳ &nbsp; ADVANCED TRAINING</div><h1>Advanced experiments.</h1><p>Existing training, checkpoint and hub tools.</p></div><TrainingView onChanged={refresh} /></div>
{:else}
  <div class="companion-page" id="main"><button class="back-button" onclick={() => view = 'workspace'}>← Back to learning studio</button><div class="companion-title"><div class="eyebrow">✳ &nbsp; MEMORY & COMPANION</div><h1>Learn together.</h1><p>Teach facts to memory and see the evidence behind a response. This does not update model weights.</p></div>{#if error}<div class="notice-box" role="alert">Memory is unavailable: {error}</div>{/if}<div class="grid"><Pet {pet} state={engineState} /><ChatPanel {factId} onChanged={refresh} /></div></div>
{/if}
