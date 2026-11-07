import * as React from "react";

import { Card, CardBody, CardHeader, CardTitle } from "../../components/ui/card";
import { api, ApiRequestError } from "../../lib/api/client";
import type { Health } from "../../lib/schemas";

/** Read-only runtime settings. Nothing here is editable until a write API exists. */
export function SettingsPage() {
  const [health, setHealth] = React.useState<Health | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const payload = await api.health();
        if (!cancelled) setHealth(payload);
      } catch (exc) {
        if (!cancelled) setError(exc instanceof ApiRequestError ? exc.message : String(exc));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="flex flex-col gap-4" aria-labelledby="settings-heading">
      <h2 id="settings-heading" className="text-sm font-semibold tracking-wide">
        Settings
      </h2>
      {error && (
        <Card role="alert" className="border-danger/60">
          <CardBody>{error}</CardBody>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Runtime state</CardTitle>
        </CardHeader>
        <CardBody>
          <p className="text-sm text-text-secondary">
            Anchor mode and storage are chosen by the process environment
            (<code>RECAMEM_ANCHOR</code>, <code>RECAMEM_HOME</code>). The API is read-only here,
            so these values are reported and not edited.
          </p>
          {health === null ? (
            <p className="mt-3 text-sm text-text-secondary">Unavailable.</p>
          ) : (
            <dl className="mt-3 grid grid-cols-2 gap-1 text-sm">
              <dt className="text-text-secondary">Status</dt>
              <dd>{health.status}</dd>
              <dt className="text-text-secondary">Profile</dt>
              <dd>{health.profile}</dd>
              <dt className="text-text-secondary">Anchor</dt>
              <dd>{health.anchor}</dd>
              <dt className="text-text-secondary">Uptime (s)</dt>
              <dd>{Math.round(health.uptime_seconds)}</dd>
              <dt className="text-text-secondary">Last error</dt>
              <dd>{health.error ?? "none"}</dd>
            </dl>
          )}
        </CardBody>
      </Card>
    </section>
  );
}
