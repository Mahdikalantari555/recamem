import type { Health } from "../../lib/schemas";
import { cn } from "../../lib/utils";

interface RuntimeStatusProps {
  health: Health | null;
  error: string | null;
  collapsed: boolean;
}

export function RuntimeStatus({ health, error, collapsed }: RuntimeStatusProps) {
  const state = error ? "disconnected" : health ? "connected" : "connecting";
  const dot =
    state === "connected" ? "bg-success" : state === "connecting" ? "bg-warning" : "bg-danger";

  return (
    <section aria-label="Runtime status" className="rounded-card border border-border bg-surface-raised p-3 text-xs">
      <p className="flex items-center gap-2 text-text-secondary">
        <span aria-hidden className={cn("h-2 w-2 rounded-full", dot)} />
        <span>{state}</span>
      </p>
      {!collapsed && health && (
        <dl className="mt-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-1 text-text-secondary">
          <dt>profile</dt>
          <dd className="truncate text-text-primary">{health.profile}</dd>
          <dt>anchor</dt>
          <dd className="text-text-primary">
            {health.anchor}
            {health.anchor === "mock" && <span className="ml-1 text-warning">(mock)</span>}
          </dd>
          <dt>device</dt>
          <dd className="text-text-primary">{health.device}</dd>
          <dt>op</dt>
          <dd className="text-text-primary">{health.active_op ?? "idle"}</dd>
        </dl>
      )}
      {!collapsed && error && <p className="mt-2 text-danger">{error}</p>}
    </section>
  );
}
