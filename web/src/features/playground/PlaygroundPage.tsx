import * as React from "react";

import { Button } from "../../components/ui/button";
import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { Textarea } from "../../components/ui/input";
import { ConfirmDialog } from "../../components/ui/dialog";
import { api, ApiRequestError } from "../../lib/api/client";
import type { InferenceResult } from "../../lib/schemas";

export function PlaygroundPage({ onActivity }: { onActivity: () => void }) {
  const [prompt, setPrompt] = React.useState("");
  const [result, setResult] = React.useState<InferenceResult | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [confirmClear, setConfirmClear] = React.useState(false);
  const controller = React.useRef<AbortController | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (!prompt.trim() || busy) return;
    setBusy(true);
    setError(null);
    controller.current = new AbortController();
    try {
      const payload = await api.infer(prompt, controller.current.signal);
      setResult(payload);
    } catch (exc) {
      if ((exc as Error).name === "AbortError") {
        setError("Request cancelled.");
      } else {
        setError(exc instanceof ApiRequestError ? exc.message : String(exc));
      }
    } finally {
      setBusy(false);
      controller.current = null;
      onActivity();
    }
  }

  function cancel() {
    controller.current?.abort();
  }

  async function clearMemory() {
    setConfirmClear(false);
    try {
      await api.clear();
      setResult(null);
      setError(null);
      onActivity();
    } catch (exc) {
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    }
  }

  const decision = result?.memory?.[0];

  return (
    <section className="flex flex-col gap-4" aria-labelledby="playground-heading">
      <h2 id="playground-heading" className="text-sm font-semibold tracking-wide">
        Playground
      </h2>
      <Card>
        <form onSubmit={submit} className="flex flex-col gap-3">
          <label htmlFor="prompt" className="text-sm text-text-secondary">
            Prompt
          </label>
          <Textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="What is the calibration constant?"
            aria-describedby="prompt-hint"
          />
          <p id="prompt-hint" className="text-xs text-text-secondary">
            Enter submits; Shift+Enter adds a line. Answers cite stored sources or say nothing was found.
          </p>
          <div className="flex flex-wrap gap-2">
            <Button type="submit" disabled={busy || !prompt.trim()}>
              {busy ? "Running…" : "Submit"}
            </Button>
            <Button variant="secondary" onClick={cancel} disabled={!busy}>
              Cancel
            </Button>
            <Button variant="ghost" onClick={() => setConfirmClear(true)}>
              Reset memory…
            </Button>
          </div>
          <p role="status" aria-live="polite" className="text-xs text-text-secondary">
            {busy ? "Running inference" : result ? "Ready" : "Idle"}
          </p>
        </form>
      </Card>

      {error && (
        <Card role="alert" className="border-danger/60">
          <CardHeader>
            <CardTitle>Request failed</CardTitle>
          </CardHeader>
          <CardBody>{error}</CardBody>
        </Card>
      )}

      {result && (
        <div className="grid gap-4 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Answer</CardTitle>
              <span className="font-mono text-xs text-text-secondary">{result.backend}</span>
            </CardHeader>
            <CardBody>
              <p className="whitespace-pre-wrap text-text-primary">{result.response}</p>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Decision inspector</CardTitle>
            </CardHeader>
            <CardBody>
              {decision ? (
                <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
                  <dt>source</dt>
                  <dd className="truncate">{decision.source ?? "unavailable"}</dd>
                  <dt>score</dt>
                  <dd>{typeof decision.score === "number" ? decision.score.toFixed(3) : "unavailable"}</dd>
                  <dt>channels</dt>
                  <dd>
                    {decision.channels
                      ? Object.entries(decision.channels)
                          .map(([name, value]) => `${name}=${Number(value).toFixed(3)}`)
                          .join(" ")
                      : "unavailable"}
                  </dd>
                  <dt>facts</dt>
                  <dd>{result.memory.length}</dd>
                </dl>
              ) : (
                <p>unavailable — this run returned no memory rows.</p>
              )}
            </CardBody>
          </Card>
        </div>
      )}

      <ConfirmDialog
        open={confirmClear}
        title="Clear learned memory?"
        description="This deletes the profile's stored facts and plastic state. Export a package first if you need a copy."
        confirmLabel="Clear memory"
        onConfirm={clearMemory}
        onCancel={() => setConfirmClear(false)}
      />
    </section>
  );
}
