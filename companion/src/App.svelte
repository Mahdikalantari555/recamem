<script lang="ts">
  import RuntimeStatus from './components/RuntimeStatus.svelte';
  import Pet from './components/Pet.svelte';
  import ChatPanel from './components/ChatPanel.svelte';
  import TrainingView from './components/TrainingView.svelte';
  import Inspector from './components/Inspector.svelte';
  import { getHealth, getState } from './lib/api';
  import type { Checkpoint, CompanionState, Health, Pet as PetState } from './lib/types';

  let tab = $state<'companion' | 'training' | 'inspector'>('companion');
  let health = $state<Health | null>(null);
  let engineState = $state<CompanionState | null>(null);
  let pet = $state<PetState | null>(null);
  let best = $state<Checkpoint | null>(null);
  let error = $state('');
  let factId = $state<number | null>(null);

  async function refresh(): Promise<void> {
    try {
      const [h, snapshot] = await Promise.all([getHealth(), getState()]);
      health = h;
      engineState = snapshot.state;
      pet = snapshot.pet;
      error = '';
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    }
  }

  async function changed(): Promise<void> {
    // Re-read measured state after teaching, feedback and sleep.
    const snapshot = await getState();
    engineState = snapshot.state;
    pet = snapshot.pet;
    const events = snapshot.events as Array<{ target: number | null }>;
    if (events.length > 0 && events[events.length - 1].target !== null) {
      factId = events[events.length - 1].target;
    }
  }

  $effect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(timer);
  });
</script>

<a class="skip-link" href="#main">Skip to content</a>
<div class="app">
  <header class="topbar">
    <div class="brand">Reca<span>Mem</span> Companion</div>
    <RuntimeStatus {health} {best} {error} />
  </header>

  <nav aria-label="Companion sections">
    <button aria-current={tab === 'companion' ? 'page' : undefined} onclick={() => (tab = 'companion')}>Companion</button>
    <button aria-current={tab === 'training' ? 'page' : undefined} onclick={() => (tab = 'training')}>Training</button>
    <button aria-current={tab === 'inspector' ? 'page' : undefined} onclick={() => (tab = 'inspector')}>Inspector</button>
  </nav>

  <main id="main">
    {#if tab === 'companion'}
      <div class="grid">
        <div>
          <Pet {pet} state={engineState} />
          <div class="panel">
            <h2>How the game works</h2>
            <p class="muted">
              Petting calls <code>approve()</code>, feeding calls <code>teach()</code>,
              scolding calls <code>reject()</code> and sleep runs one consolidation
              cycle. Mood, energy, bond and knowledge are computed from measured
              engine signals. No cosmetic-only rewards.
            </p>
          </div>
        </div>
        <ChatPanel {factId} onChanged={changed} />
      </div>
    {:else if tab === 'training'}
      <TrainingView onChanged={changed} />
    {:else}
      <Inspector state={engineState} {pet} />
    {/if}
  </main>

  <footer class="muted" style="margin-top:24px">
    RecaMem companion · same-origin loopback API · relative URLs only · no invented metrics
  </footer>
</div>
