"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { StatusDot } from "./ui";

function timeAgo(ts: number | null): string {
  if (ts == null) return "never";
  const secs = Math.max(0, Math.round((Date.now() - ts) / 1000));
  if (secs < 2) return "just now";
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  return `${hrs}h ago`;
}

/**
 * Compact "live / last-updated" indicator with a manual refresh button.
 * Self-ticks every second so the relative time stays fresh.
 */
export function LiveIndicator({
  live,
  refreshing,
  lastUpdated,
  onRefresh,
}: {
  live: boolean;
  refreshing: boolean;
  lastUpdated: number | null;
  onRefresh: () => void;
}) {
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className="flex items-center gap-2">
      <span
        className="flex items-center gap-1.5 rounded-md border border-border bg-muted px-2 py-1 text-xs text-muted-foreground"
        title={live ? "Auto-refresh is active" : "Auto-refresh paused (tab hidden or inactive)"}
      >
        <StatusDot tone={live ? "ok" : "neutral"} pulse={live} />
        {live ? "Live" : "Paused"}
        <span className="text-muted-foreground/60">·</span>
        <span className="tabular-nums">{timeAgo(lastUpdated)}</span>
      </span>
      <button
        type="button"
        onClick={onRefresh}
        aria-label="Refresh now"
        title="Refresh now"
        className="grid h-7 w-7 place-items-center rounded-lg border border-border bg-transparent text-muted-foreground transition hover:bg-muted hover:text-foreground"
      >
        <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
      </button>
    </div>
  );
}
