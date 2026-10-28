import * as React from "react";

import { cn } from "../../lib/utils";

export interface NavItem {
  id: string;
  label: string;
  hint: string;
  available: boolean;
}

interface SidebarProps {
  items: NavItem[];
  active: string;
  collapsed: boolean;
  onSelect: (id: string) => void;
  onToggle: () => void;
  footer?: React.ReactNode;
}

/** Left nav rail. Collapsible, keyboard reachable, labelled for screen readers. */
export function Sidebar({ items, active, collapsed, onSelect, onToggle, footer }: SidebarProps) {
  return (
    <nav
      aria-label="Primary"
      className={cn(
        "flex h-full flex-col gap-2 border-r border-border bg-surface p-3 transition-[width]",
        collapsed ? "w-16" : "w-60",
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={!collapsed}
        aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}
        className="mb-2 self-end rounded-control px-2 py-1 text-xs text-text-secondary hover:text-text-primary"
      >
        {collapsed ? "»" : "« Collapse"}
      </button>
      <ul className="flex flex-1 flex-col gap-1">
        {items.map((item) => {
          const selected = item.id === active;
          return (
            <li key={item.id}>
              <button
                type="button"
                onClick={() => onSelect(item.id)}
                aria-current={selected ? "page" : undefined}
                disabled={!item.available}
                title={item.available ? item.hint : `${item.hint} (not available yet)`}
                className={cn(
                  "w-full rounded-control px-3 py-2 text-left text-sm",
                  selected ? "bg-surface-raised text-text-primary" : "text-text-secondary",
                  item.available ? "hover:bg-surface-raised hover:text-text-primary" : "opacity-50",
                )}
              >
                <span className={cn(collapsed && "sr-only")}>{item.label}</span>
                {collapsed && <span aria-hidden>{item.label.slice(0, 1)}</span>}
              </button>
            </li>
          );
        })}
      </ul>
      {footer}
    </nav>
  );
}
