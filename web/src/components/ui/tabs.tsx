import * as React from "react";

import { cn } from "../../lib/utils";

export interface TabItem {
  id: string;
  label: string;
}

interface TabsProps {
  items: TabItem[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

/** Roving-tabindex tablist: ArrowLeft/ArrowRight move, Home/End jump. */
export function Tabs({ items, active, onChange, className }: TabsProps) {
  const refs = React.useRef<Array<HTMLButtonElement | null>>([]);

  function onKeyDown(event: React.KeyboardEvent, index: number) {
    const last = items.length - 1;
    let next: number | null = null;
    if (event.key === "ArrowRight") next = index === last ? 0 : index + 1;
    if (event.key === "ArrowLeft") next = index === 0 ? last : index - 1;
    if (event.key === "Home") next = 0;
    if (event.key === "End") next = last;
    if (next === null) return;
    event.preventDefault();
    onChange(items[next].id);
    refs.current[next]?.focus();
  }

  return (
    <div role="tablist" aria-label="Workspace sections" className={cn("flex gap-1 border-b border-border", className)}>
      {items.map((item, index) => {
        const selected = item.id === active;
        return (
          <button
            key={item.id}
            ref={(node) => {
              refs.current[index] = node;
            }}
            role="tab"
            id={`tab-${item.id}`}
            aria-selected={selected}
            aria-controls={`panel-${item.id}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.id)}
            onKeyDown={(event) => onKeyDown(event, index)}
            className={cn(
              "rounded-t-control px-3 py-2 text-sm",
              selected
                ? "border-b-2 border-accent text-text-primary"
                : "text-text-secondary hover:text-text-primary",
            )}
          >
            {item.label}
          </button>
        );
      })}
    </div>
  );
}
