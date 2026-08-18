"use client";

import { useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";
import { useSectionActive } from "./shell/SectionActivity";
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
  const { t } = useI18n();
  const sectionActive = useSectionActive();
  const [, force] = useState(0);
  useEffect(() => {
    // The tick only keeps the relative time fresh — never run it inside a
    // hidden section or a hidden browser tab. (Previously this ticked 1 Hz
    // for the whole session in every mounted-but-hidden subtree.)
    if (!sectionActive) return;
    let id: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      if (id == null) id = setInterval(() => force((n) => n + 1), 1000);
    };
    const stop = () => {
      if (id != null) {
        clearInterval(id);
        id = null;
      }
    };
    const onVisibility = () => (document.hidden ? stop() : start());
    document.addEventListener("visibilitychange", onVisibility);
    onVisibility();
    return () => {
      document.removeEventListener("visibilitychange", onVisibility);
      stop();
    };
  }, [sectionActive]);

  return (
    <div className="flex items-center gap-2">
      <span
        className="flex items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 py-1 text-xs text-muted-foreground"
        title={t(live ? "Auto-refresh is active" : "Auto-refresh paused (tab hidden or inactive)")}
      >
        <StatusDot tone={live ? "ok" : "neutral"} pulse={live} />
        {t(live ? "Live" : "Paused")}
        <span className="text-muted-foreground/60">·</span>
        <span className="tabular-nums">{timeAgo(lastUpdated)}</span>
      </span>
      <button
        type="button"
        onClick={onRefresh}
        aria-label={t("Refresh now")}
        title={t("Refresh now")}
        className="grid h-7 w-7 place-items-center rounded-lg border border-border bg-muted text-muted-foreground transition hover:text-foreground"
      >
        <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
      </button>
    </div>
  );
}
