import * as React from "react";

import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { api, ApiRequestError } from "../../lib/api/client";
import type { Health, ModelsResponse } from "../../lib/schemas";

/** Implementation metadata only: everything here is read from `/health` and `/models`. */
export function ArchitecturePage() {
  const [health, setHealth] = React.useState<Health | null>(null);
  const [models, setModels] = React.useState<ModelsResponse | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [healthPayload, modelPayload] = await Promise.all([api.health(), api.models()]);
        if (cancelled) return;
        setHealth(healthPayload);
        setModels(modelPayload);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof ApiRequestError ? exc.message : String(exc));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="flex flex-col gap-4" aria-labelledby="architecture-heading">
      <h2 id="architecture-heading" className="text-sm font-semibold tracking-wide">
        Architecture Lab
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Runtime</CardTitle>
        </CardHeader>
        <CardBody>
          {health === null ? (
            <p className="text-sm text-text-secondary">Unavailable.</p>
          ) : (
            <dl className="grid grid-cols-2 gap-1 text-sm">
              <dt className="text-text-secondary">Profile</dt>
              <dd>{health.profile}</dd>
              <dt className="text-text-secondary">Anchor</dt>
              <dd>{health.anchor}</dd>
              <dt className="text-text-secondary">Device</dt>
              <dd>{health.device}</dd>
              <dt className="text-text-secondary">Loaded</dt>
              <dd>{health.loaded ? "yes" : "no"}</dd>
              <dt className="text-text-secondary">Active operation</dt>
              <dd>{health.active_op ?? "idle"}</dd>
              <dt className="text-text-secondary">Version</dt>
              <dd>{health.version}</dd>
            </dl>
          )}
        </CardBody>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>Profiles</CardTitle>
        </CardHeader>
        <CardBody>
          {models === null ? (
            <p className="text-sm text-text-secondary">Unavailable.</p>
          ) : models.models.length === 0 ? (
            <p className="text-sm text-text-secondary">
              No profile exists in the configured home directory yet.
            </p>
          ) : (
            <ul className="list-disc pl-5 text-sm">
              {models.models.map((name) => (
                <li key={name}>
                  {name}
                  {name === models.selected ? " (selected)" : ""}
                </li>
              ))}
            </ul>
          )}
        </CardBody>
      </Card>
    </section>
  );
}
