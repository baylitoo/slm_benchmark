"use client";

import { formatBytes, type SeedProgress } from "@/lib/api";
import { JsonView } from "./JsonView";

/** Structured download progress renders as a bar; anything else as JSON.
 * Shared by the realtime stream and the polling fallback so both show the
 * identical percentage bar. */
export function ProgressView({ value }: { value: unknown }) {
  const p = value as SeedProgress | null;
  if (!p || typeof p !== "object" || typeof p.percent !== "number") {
    return <JsonView value={value} maxHeight="8rem" />;
  }
  const pct = Math.max(0, Math.min(100, p.percent));
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span className="truncate">
          {p.stage === "download-mmproj" || p.stage === "download-snapshot"
            ? "Vision projector"
            : "Model"}
          {p.file ? ` · ${p.file}` : ""}
        </span>
        <span className="shrink-0 pl-2 font-medium text-foreground">
          {pct.toFixed(0)}%
          {p.received_bytes != null && p.total_bytes != null
            ? ` · ${formatBytes(p.received_bytes)} / ${formatBytes(p.total_bytes)}`
            : ""}
        </span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full bg-accent transition-all duration-300"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
