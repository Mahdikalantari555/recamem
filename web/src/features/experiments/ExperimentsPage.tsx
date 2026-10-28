import * as React from "react";

import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { api, ApiRequestError } from "../../lib/api/client";
import type { ExperimentRow, Job } from "../../lib/schemas";

export function ExperimentsPage() {
  const [runs, setRuns] = React.useState<ExperimentRow[] | null>(null);
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [error, setError] = React.useState<string | null>(null);
  const [selected, setSelected] = React.useState<ExperimentRow | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [experiments, jobList] = await Promise.all([api.experiments(), api.jobs()]);
        if (cancelled) return;
        setRuns(experiments.experiments);
        setJobs(jobList.jobs);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof ApiRequestError ? exc.message : String(exc));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="flex flex-col gap-4" aria-labelledby="experiments-heading">
      <h2 id="experiments-heading" className="text-sm font-semibold tracking-wide">
        Experiments
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Recorded runs</CardTitle>
          <span className="font-mono text-xs text-text-secondary">{runs?.length ?? "…"}</span>
        </CardHeader>
        <CardBody>
          {runs === null ? (
            <p>Loading…</p>
          ) : runs.length === 0 ? (
            <p>No runs recorded yet. Benchmark results appear here once a runner writes evidence.</p>
          ) : (
            <table className="w-full text-left text-xs">
              <caption className="sr-only">Recorded evidence files</caption>
              <thead>
                <tr className="text-text-secondary">
                  <th scope="col" className="py-1">run</th>
                  <th scope="col" className="py-1">bytes</th>
                  <th scope="col" className="py-1">modified</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {runs.map((run) => (
                  <tr key={run.id} className="border-t border-border">
                    <td className="py-1">
                      <button
                        type="button"
                        className="text-accent underline-offset-2 hover:underline"
                        onClick={() => setSelected(run)}
                      >
                        {run.id}
                      </button>
                    </td>
                    <td className="py-1">{run.bytes}</td>
                    <td className="py-1">{new Date(run.modified * 1000).toISOString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </CardBody>
      </Card>
      {selected && (
        <Card>
          <CardHeader>
            <CardTitle>Run detail</CardTitle>
          </CardHeader>
          <CardBody>
            <dl className="grid grid-cols-[auto_1fr] gap-x-3 font-mono text-xs">
              <dt>id</dt>
              <dd>{selected.id}</dd>
              <dt>kind</dt>
              <dd>{selected.kind}</dd>
              <dt>bytes</dt>
              <dd>{selected.bytes}</dd>
            </dl>
          </CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Jobs</CardTitle>
        </CardHeader>
        <CardBody>
          {jobs.length === 0 ? (
            <p>No jobs. Training and evaluation jobs appear here when the API creates them.</p>
          ) : (
            <ul className="flex flex-col gap-1 font-mono text-xs">
              {jobs.map((job) => (
                <li key={job.id}>
                  {job.id} · {job.kind} · {job.status}
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>
    </section>
  );
}
