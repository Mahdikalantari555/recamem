<script lang="ts">
  import type { CompanionState, Pet } from '../lib/types';

  let { state, pet }: { state: CompanionState | null; pet: Pet | null } = $props();
</script>

<div class="panel">
  <h2>Measured signals</h2>
  <p class="muted">
    Everything the pet displays comes from these engine readings. An unavailable
    signal is shown as null, never replaced by a decorative counter.
  </p>
  <h3 style="font-size:13px">Sources</h3>
  <ul class="muted">
    {#each state?.signals ?? [] as signal (signal)}<li><code>{signal}</code></li>{/each}
  </ul>
  <h3 style="font-size:13px">Raw state</h3>
  <pre>{JSON.stringify(state ?? {}, null, 2)}</pre>
  <h3 style="font-size:13px">Derived pet</h3>
  <pre>{JSON.stringify(pet ?? {}, null, 2)}</pre>
</div>
