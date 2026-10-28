import * as React from "react";

import { cn } from "../../lib/utils";

export const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      ref={ref}
      className={cn(
        "h-10 w-full rounded-control border border-border bg-background px-3 text-sm text-text-primary placeholder:text-text-secondary focus-visible:border-accent",
        className,
      )}
      {...props}
    />
  ),
);
Input.displayName = "Input";

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, ...props }, ref) => (
  <textarea
    ref={ref}
    className={cn(
      "min-h-[88px] w-full rounded-control border border-border bg-background p-3 text-sm text-text-primary placeholder:text-text-secondary focus-visible:border-accent",
      className,
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";
