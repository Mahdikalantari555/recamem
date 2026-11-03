<script lang="ts">
  import { act, inference } from '../lib/api';
  import type { CompanionEvent, Turn } from '../lib/types';

  let {
    factId,
    onChanged
  }: { factId: number | null; onChanged: () => Promise<void> } = $props();

  let prompt = $state('');
  let messages = $state<Turn[]>([]);
  let events = $state<CompanionEvent[]>([]);
  let busy = $state(false);
  let error = $state('');

  async function send(): Promise<void> {
    const text = prompt.trim();
    if (!text || busy) return;
    busy = true;
    error = '';
    try {
      const result = await inference(text);
      messages = [...messages, {
        prompt: text,
        response: result.response,
        memory: result.memory || [],
        anchor: result.anchor,
        backend: result.backend,
        delivery: 'full_turn'
      }];
      prompt = '';
      await onChanged();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }

  async function doAction(action: string): Promise<void> {
    if (busy) return;
    busy = true;
    error = '';
    try {
      const payload: Record<string, unknown> = {};
      if (['feed', 'correct'].includes(action)) payload.text = prompt.trim();
      if (factId !== null) payload.fact_id = factId;
      const outcome = await act(action, payload);
      events = [outcome.event, ...events].slice(0, 20);
      if (action === 'feed' || action === 'correct' || action === 'sleep') prompt = '';
      await onChanged();
    } catch (cause) {
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      busy = false;
    }
  }

  function onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) void send();
  }
</script>

<div class="panel">
  <h2>Talk online</h2>
  <p class="muted">
    Every turn goes through the real engine. Chat answers are full turns
    (<code>delivery=full_turn</code>); token streaming is only used when a backend
    advertises it. Recalled evidence is shown below each answer.
  </p>
  <textarea bind:value={prompt} onkeydown={onKeydown} aria-label="Message or fact to teach"
    placeholder="Ask something, or write a fact and press Feed …"></textarea>
  <div class="row" style="margin-top:10px">
    <button class="btn primary" onclick={() => void send()} disabled={busy || !prompt.trim()}>Send chat</button>
    <button class="btn" onclick={() => void doAction('feed')} disabled={busy || !prompt.trim()}>Feed a fact</button>
    <button class="btn" onclick={() => void doAction('pet')} disabled={busy || factId === null}
      title={factId === null ? 'Needs a recalled fact' : `Reinforce fact ${factId}`}>Pet</button>
    <button class="btn" onclick={() => void doAction('treat')} disabled={busy || factId === null}>Treat</button>
    <button class="btn" onclick={() => void doAction('scold')} disabled={busy || factId === null}>Scold</button>
    <button class="btn" onclick={() => void doAction('correct')} disabled={busy || factId === null || !prompt.trim()}>Correct</button>
    <button class="btn" onclick={() => void doAction('sleep')} disabled={busy}>Sleep / consolidate</button>
  </div>
  {#if factId === null}
    <p class="muted">No recalled fact yet — pet/treat/scold need a recalled fact id.</p>
  {:else}
    <p class="muted">Last recalled fact id: {factId}</p>
  {/if}
  {#if error}<p class="error" role="alert">{error}</p>{/if}

  <div class="messages" style="margin-top:14px" aria-live="polite">
    {#if messages.length === 0}<p class="empty">No turns yet.</p>{/if}
    {#each messages as turn, index (index)}
      <div class="msg user"><strong>You</strong> {turn.prompt}</div>
      <div class="msg pet">
        <strong>Companion</strong>
        <div>{turn.response || '(no information)'}</div>
        <div class="evidence">
          anchor={turn.anchor} · backend={turn.backend} · {turn.delivery}
          {#if turn.memory.length > 0}
            <ul>
              {#each turn.memory as hit, hitIndex (hitIndex)}
                <li>{hit.text ?? '(empty)'} {#if hit.source}— {hit.source}{/if}</li>
              {/each}
            </ul>
          {:else}
            <div>No recalled facts for this turn.</div>
          {/if}
        </div>
      </div>
    {/each}
  </div>
</div>

<div class="panel">
  <h2>What changed</h2>
  {#if events.length === 0}<p class="empty">No companion actions yet.</p>{/if}
  {#each events as event (event.id)}
    <div class="msg">
      <strong>{event.action}</strong> → <code>{event.engine_call}</code>
      <div class="muted">
        facts active: {String(event.before.facts_active ?? 'n/a')} → {String(event.after.facts_active ?? 'n/a')}
      </div>
    </div>
  {/each}
</div>
