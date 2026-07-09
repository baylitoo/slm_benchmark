"use client";

import type { ReactNode } from "react";
import { SlidersHorizontal, RotateCcw } from "lucide-react";
import { Button } from "../ui";

/**
 * Filter row: a "Filters" label affordance, the filter controls, and a Reset
 * button. Purely presentational — the actual filtering is client-side state the
 * page owns.
 */
export function Toolbar({
  children,
  onReset,
  resetDisabled,
}: {
  children?: ReactNode;
  onReset?: () => void;
  resetDisabled?: boolean;
}) {
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2">
      <span className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-2.5 py-1.5 text-xs font-medium text-muted-foreground">
        <SlidersHorizontal className="h-3.5 w-3.5" />
        Filters
      </span>
      {children}
      {onReset && (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={onReset}
          disabled={resetDisabled}
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset
        </Button>
      )}
    </div>
  );
}
