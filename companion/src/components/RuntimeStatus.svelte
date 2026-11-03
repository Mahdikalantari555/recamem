<script lang="ts">
  import type { Checkpoint, Health } from '../lib/types';

  let { health, best, error }: { health: Health | null; best: Checkpoint | null; error: string } = $props();
</script>

<div class="row" aria-live="polite">
  <span class="badge good">anchor: {health?.anchor ?? 'unknown'}</span>
  <span class="badge">backend: {health?.loaded ? 'engine loaded' : 'not loaded'}</span>
  <span class="badge">profile: {health?.profile ?? 'unknown'}</span>
  {#if best}
    <span class="badge good">model: {best.name} (BPB {best.held_out_bpb?.toFixed(4)})</span>
  {:else}
    <span class="badge warn">demo / recall-only — no eligible trained checkpoint</span>
  {/if}
  {#if error}<span class="badge bad">{error}</span>{/if}
</div>
