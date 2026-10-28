import * as React from "react";

export function SettingsPage() {
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold">Settings</h2>
      <p className="text-sm text-text-secondary">
        Anchor mode, storage location, and selected core are shown in the
        runtime status panel. Editing is not yet wired.
      </p>
      <p className="text-xs text-text-muted">Status: read-only.</p>
    </section>
  );
}
