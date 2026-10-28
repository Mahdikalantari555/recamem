import * as React from "react";

import { cn } from "../../lib/utils";
import { Button } from "./button";

interface ConfirmDialogProps {
  open: boolean;
  title: string;
  description: string;
  confirmLabel?: string;
  onConfirm: () => void;
  onCancel: () => void;
}

/**
 * Confirmation for destructive actions only (design §16: no modals for routine
 * work). Uses the native <dialog> element so focus trapping and Esc come from
 * the platform.
 */
export function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  onConfirm,
  onCancel,
}: ConfirmDialogProps) {
  const ref = React.useRef<HTMLDialogElement>(null);

  React.useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (open && !node.open) node.showModal();
    if (!open && node.open) node.close();
  }, [open]);

  return (
    <dialog
      ref={ref}
      aria-labelledby="confirm-title"
      onCancel={(event) => {
        event.preventDefault();
        onCancel();
      }}
      className={cn(
        "rounded-card border border-border bg-surface p-5 text-text-primary backdrop:bg-black/60",
      )}
    >
      <h2 id="confirm-title" className="mb-2 text-sm font-semibold">
        {title}
      </h2>
      <p className="mb-4 max-w-sm text-sm text-text-secondary">{description}</p>
      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onCancel}>
          Cancel
        </Button>
        <Button variant="danger" onClick={onConfirm}>
          {confirmLabel}
        </Button>
      </div>
    </dialog>
  );
}
