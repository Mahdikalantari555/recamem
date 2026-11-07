import * as React from "react";

import { Button } from "../../components/ui/button";
import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { api, ApiRequestError } from "../../lib/api/client";
import type { Job } from "../../lib/schemas";

/** Training jobs are real `/jobs` entries; an empty list stays empty. */
export function TrainingPage() {
  const [jobs, setJobs] = React.useState<Job[] | null>(null);
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  const refresh = React.useCallback(async () => {
    setBusy(true);
    try {
      const payload = await api.jobs();
      setJobs(payload.jobs);
      setError(null);
    } catch (exc) {
      setJobs(null);
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }, []);

  React.useEffect(() => {
    void refresh();
  }, [refresh]);

  async function cancel(id: string) {
    try {
      await api.cancelJob(id);
      await refresh();
    } catch (exc) {
      setError(exc instanceof ApiRequestError ? exc.message : String(exc));
    }
  }

  return (
    <section className="flex flex-col gap-4" aria-labelledby="training-heading">
      <h2 id="training-heading" className="text-sm font-semibold tracking-wide">
        Training
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Jobs</CardTitle>
        </CardHeader>
        <CardBody className="flex flex-col gap-2">
          <div>
            <Button type="button" onClick={() => void refresh()} disabled={busy}>
              {busy ? "Refreshing…" : "Refresh"}
            </Button>
          </div>
          {jobs === null && <p className="text-sm text-text-secondary">Unavailable.</p>}
          {jobs !== null && jobs.length === 0 && (
            <p className="text-sm text-text-secondary">
              No job has been created in this process. Training starts from the CLI or the
              companion app; nothing is simulated here.
            </p>
          )}
          {jobs !== null && jobs.length > 0 && (
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="text-text-secondary">
                  <th scope="col" className="py-1">
                    Job
                  </th>
                  <th scope="col" className="py-1">
                    Kind
                  </th>
                  <th scope="col" className="py-1">
                    Status
                  </th>
                  <th scope="col" className="py-1">
                    Progress
                  </th>
                  <th scope="col" className="py-1">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id} className="border-t border-border">
                    <td className="py-1">{job.id}</td>
                    <td className="py-1">{job.kind}</td>
                    <td className="py-1">{job.status}</td>
                    <td className="py-1">{Math.round(job.progress * 100)}%</td>
                    <td className="py-1">
                      {job.status === "queued" || job.status === "running" ? (
                        <Button type="button" onClick={() => void cancel(job.id)}>
                          Cancel
                        </Button>
                      ) : (
                        "—"
                      )}
                    </td>
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
