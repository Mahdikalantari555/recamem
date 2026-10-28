import * as React from "react";

import { NavItem, Sidebar } from "../ui/sidebar";
import { RuntimeStatus } from "../status/RuntimeStatus";
import type { Health } from "../../lib/schemas";

interface AppShellProps {
  items: NavItem[];
  active: string;
  onSelect: (id: string) => void;
  health: Health | null;
  error: string | null;
  children: React.ReactNode;
}

export function AppShell({ items, active, onSelect, health, error, children }: AppShellProps) {
  const [collapsed, setCollapsed] = React.useState(false);

  return (
    <div className="flex h-full min-h-screen">
      <a
        href="#workspace"
        className="sr-only focus:not-sr-only focus:absolute focus:left-2 focus:top-2 focus:z-10 focus:rounded-control focus:bg-surface-raised focus:px-3 focus:py-2"
      >
        Skip to workspace
      </a>
      <Sidebar
        items={items}
        active={active}
        collapsed={collapsed}
        onSelect={onSelect}
        onToggle={() => setCollapsed((value) => !value)}
        footer={<RuntimeStatus health={health} error={error} collapsed={collapsed} />}
      />
      <main id="workspace" className="flex-1 overflow-y-auto p-6">
        <header className="mb-5">
          <h1 className="text-lg font-semibold">RecaMem</h1>
          <p className="text-sm text-text-secondary">
            Local-first plastic memory engine — {health?.version ?? "version unknown"}
          </p>
        </header>
        {children}
      </main>
    </div>
  );
}
