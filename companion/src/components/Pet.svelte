<script lang="ts">
  import type { CompanionState, Pet as PetState } from '../lib/types';
  import { formatMetric, petLabel } from '../lib/logic';

  let { pet, state }: { pet: PetState | null; state: CompanionState | null } = $props();
  const meters = $derived(pet ? (['mood', 'energy', 'bond', 'knowledge'] as const) : []);
</script>

<div class="panel">
  <h2>Your companion</h2>
  <svg class="pet-figure" width="160" height="140" viewBox="0 0 160 140" role="img" aria-label="A pet whose mood comes from measured memory signals">
    <ellipse cx="80" cy="122" rx="46" ry="8" fill="#0b0f15" />
    <circle cx="80" cy="70" r="48" fill={pet?.mood.value === null ? '#4a5566' : '#b5f36d'} opacity="0.18" />
    <circle cx="80" cy="70" r="38" fill="#222c38" stroke="#3a4757" stroke-width="2" />
    <circle cx="66" cy="64" r="5" fill="#eef2f7" />
    <circle cx="94" cy="64" r="5" fill="#eef2f7" />
    <path d="M70 84 Q80 {(pet?.mood.value ?? 0) >= 0.5 ? 94 : 82} 90 84" stroke="#eef2f7" stroke-width="2" fill="none" stroke-linecap="round" />
    <path d="M56 38 L48 24 L66 34" fill="#222c38" stroke="#3a4757" stroke-width="2" />
    <path d="M104 38 L112 24 L94 34" fill="#222c38" stroke="#3a4757" stroke-width="2" />
  </svg>
  <p class="muted" style="text-align:center">
    {pet ? `Mood: ${petLabel(pet.mood.value)}` : 'No state measured yet.'}
  </p>
  {#each meters as key (key)}
    {@const item = pet?.[key]}
    <div class="meter-line">
      <span>{key}</span>
      <div class="meter"><div style={`width:${Math.round((item?.value ?? 0) * 100)}%`}></div></div>
      <span class="muted">{item?.value === null || item?.value === undefined ? 'no signal' : formatMetric(item.value)}</span>
    </div>
  {/each}
  <p class="muted">
    Facts: {state?.facts_active ?? 'unavailable'} active / {state?.facts_total ?? 'unavailable'} total ·
    approvals: {state?.feedback_counts?.approve ?? 0} ·
    retrieval quality: {state?.retrieval_hit_rate === null || state?.retrieval_hit_rate === undefined ? 'no labelled feedback' : formatMetric(state.retrieval_hit_rate)}
  </p>
  {#if pet}<p class="muted">{pet.note}</p>{/if}
</div>
