import * as React from "react";

import { Button } from "../../components/ui/button";
import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { api, ApiRequestError } from "../../lib/api/client";
import type { ExperimentRow } from "../../lib/schemas";

/** Benchmark status comes from `/bench`; recorded runs come from `/experiments`. */
export function BenchmarksPage() {
  const [status, setStatus] = React.useState<{ status: string; reason: string } | null>(null);
  const [runs, setRuns] = React.useState<ExperimentRow[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const payload = await api.experiments();
        if (!cancelled) setRuns(payload.experiments);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof ApiRequestError ? exc.message : String(exc));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function check() {
    setBusy(true);
    setError(null);
    try {
      const payload = await api.bench();
      setStatus({ status: payload.status, reason: payload.reason });
    } catch (exc) {
      setStatus(null);
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="flex flex-col gap-4" aria-labelledby="benchmarks-heading">
      <h2 id="benchmarks-heading" className="text-sm font-semibold tracking-wide">
        Benchmarks
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Benchmark status</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-text-secondary">
            A benchmark run needs a dataset. The backend reports <code>blocked</code> with the
            missing input instead of an invented score.
          </p>
          <Button type="button" onClick={check} disabled={busy} className="mt-2">
            {busy ? "Checking…" : "Check status"}
          </Button>
          {status && (
            <dl className="mt-3 grid grid-cols-2 gap-1 text-sm">
              <dt className="text-text-secondary">Status</dt>
              <dd>{status.status}</dd>
              <dt className="text-text-secondary">Reason</dt>
              <dd>{status.reason || "none"}</dd>
            </dl>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Recorded runs</CardTitle>
        </CardHeader>
        <CardBody>
          {runs === null && <p className="text-sm text-text-secondary">Unavailable.</p>}
          {runs !== null && runs.length === 0 && (
            <p className="text-sm text-text-secondary">
              No run has been recorded in this profile directory.
            </p>
          )}
          {runs !== null && runs.length > 0 && (
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-text-secondary">
                  <th scope="col" className="py-1">
                    Run
                  </th>
                  <th scope="col" className="py-1">
                    Kind
                  </th>
                  <th scope="col" className="py-1">
                    Size (bytes)
                  </th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => (
                  <tr key={run.id} className="border-t border-border">
                    <td className="py-1">{run.id}</td>
                    <td className="py-1">{run.kind}</td>
                    <td className="py-1">{run.bytes}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardBody>
      </Card>
    </section>
  );
}
