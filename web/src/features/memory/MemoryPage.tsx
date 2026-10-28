import * as React from "react";

export function MemoryPage() {
  return (
    <section className="space-y-3">
      <h2 className="text-base font-semibold">Memory Explorer</h2>
      <p className="text-sm text-text-secondary">
        Browse, edit, and retire stored facts. Backend wiring pending — this surface
        is not yet connected to a live profile.
      </p>
      <p className="text-xs text-text-muted">Status: unavailable until the engine API exposes fact listing.</p>
    </section>
  );
}
