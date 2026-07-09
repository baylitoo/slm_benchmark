"use client";

import { Menu, PanelLeft, Layers } from "lucide-react";
import { API_BASE } from "@/lib/env";
import type { Health } from "@/lib/useBackendHealth";
import { Badge, StatusDot } from "../ui";
import { ThemeToggle } from "../ThemeToggle";

const GITHUB_URL = "https://github.com/baylitoo/slm_benchmark";

/** Inline GitHub brand mark (lucide dropped brand icons). */
function GithubMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="currentColor"
      aria-hidden
      className={className}
    >
      <path d="M12 .5C5.37.5 0 5.87 0 12.5c0 5.3 3.44 9.8 8.21 11.39.6.11.82-.26.82-.58v-2.03c-3.34.73-4.04-1.61-4.04-1.61-.55-1.39-1.34-1.76-1.34-1.76-1.09-.75.08-.73.08-.73 1.21.09 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.5 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.34-5.47-5.95 0-1.31.47-2.39 1.24-3.23-.13-.31-.54-1.53.11-3.19 0 0 1.01-.32 3.3 1.23a11.5 11.5 0 0 1 3-.4c1.02 0 2.05.14 3 .4 2.28-1.55 3.29-1.23 3.29-1.23.66 1.66.25 2.88.12 3.19.77.84 1.24 1.92 1.24 3.23 0 4.62-2.81 5.64-5.49 5.94.43.37.82 1.1.82 2.22v3.29c0 .32.22.7.83.58C20.56 22.3 24 17.8 24 12.5 24 5.87 18.63.5 12 .5z" />
    </svg>
  );
}

const HEALTH_META: Record<Health, { tone: "ok" | "warn" | "err"; label: string }> = {
  checking: { tone: "warn", label: "Connecting" },
  online: { tone: "ok", label: "Live" },
  offline: { tone: "err", label: "Offline" },
};

/** Short host label for the env pill (drops scheme, keeps host[:port]). */
function envLabel(): string {
  try {
    return new URL(API_BASE).host || API_BASE;
  } catch {
    return API_BASE.replace(/^https?:\/\//, "");
  }
}

/**
 * Sticky top bar: hamburger/collapse, logo echo + version pill on the left; env
 * pill, live status dot, theme toggle, GitHub link and account chip on the
 * right. The page title lives in the in-content PageHeader (LiteLLM pattern).
 */
export function TopBar({
  onToggleMobile,
  onToggleCollapse,
  health,
}: {
  onToggleMobile: () => void;
  onToggleCollapse: () => void;
  health: Health;
}) {
  const meta = HEALTH_META[health];
  return (
    <header className="sticky top-0 z-20 flex h-14 items-center justify-between gap-3 border-b border-border bg-card px-4">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onToggleMobile}
          aria-label="Open navigation"
          className="rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>
        <button
          type="button"
          onClick={onToggleCollapse}
          aria-label="Toggle sidebar"
          className="hidden rounded-md p-1.5 text-muted-foreground hover:bg-muted hover:text-foreground lg:inline-flex"
        >
          <PanelLeft className="h-5 w-5" />
        </button>
        <span className="flex items-center gap-2 lg:hidden">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-accent text-accent-foreground">
            <Layers className="h-4 w-4" />
          </span>
          <span className="text-sm font-semibold text-foreground">DocIE Studio</span>
        </span>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        <Badge tone="neutral" className="hidden font-mono text-[11px] sm:inline-flex">
          {envLabel()}
        </Badge>
        <span
          className="flex items-center gap-1.5 text-xs text-muted-foreground"
          title={meta.label}
        >
          <StatusDot tone={meta.tone} pulse={health !== "offline"} />
          <span className="hidden sm:inline">{meta.label}</span>
        </span>
        <ThemeToggle />
        <a
          href={GITHUB_URL}
          target="_blank"
          rel="noreferrer"
          aria-label="GitHub repository"
          className="grid h-9 w-9 place-items-center rounded-md border border-border bg-card text-muted-foreground transition hover:bg-muted hover:text-foreground"
        >
          <GithubMark className="h-4 w-4" />
        </a>
        <span className="flex items-center gap-2 rounded-md px-1 py-1">
          <span className="grid h-8 w-8 place-items-center rounded-full bg-accent/10 text-xs font-semibold text-accent">
            DS
          </span>
          <span className="hidden text-sm font-medium text-foreground md:inline">Operator</span>
        </span>
      </div>
    </header>
  );
}
