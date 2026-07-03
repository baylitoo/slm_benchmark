"use client";

import { useState } from "react";
import {
  FlaskConical,
  Rocket,
  Gauge,
  Activity,
  Layers,
  Menu,
  X,
} from "lucide-react";
import { API_BASE } from "@/lib/env";
import { useBackendHealth } from "@/lib/useBackendHealth";
import { cn } from "@/lib/cn";
import { Playground } from "./Playground";
import { Deploy } from "./Deploy";
import { Benchmark } from "./Benchmark";
import { Observability } from "./Observability";
import { ThemeToggle } from "./ThemeToggle";
import { StatusDot } from "./ui";

type SectionId = "playground" | "deploy" | "benchmark" | "observability";

const NAV: {
  id: SectionId;
  label: string;
  desc: string;
  icon: typeof FlaskConical;
}[] = [
  { id: "playground", label: "Playground", desc: "Run a live extraction", icon: FlaskConical },
  { id: "deploy", label: "Deploy", desc: "Serve & manage models", icon: Rocket },
  { id: "benchmark", label: "Benchmark", desc: "Evaluate & compare", icon: Gauge },
  { id: "observability", label: "Observability", desc: "Metrics & dashboards", icon: Activity },
];

const HEALTH_META = {
  checking: { tone: "warn" as const, label: "Connecting" },
  online: { tone: "ok" as const, label: "Backend online" },
  offline: { tone: "err" as const, label: "Backend offline" },
};

export function AppShell() {
  const [active, setActive] = useState<SectionId>("playground");
  const [mobileOpen, setMobileOpen] = useState(false);
  const health = useBackendHealth();
  const current = NAV.find((n) => n.id === active)!;

  return (
    <div className="flex min-h-screen bg-background">
      {/* Mobile sidebar backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden
        />
      )}

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-border bg-card transition-transform lg:static lg:translate-x-0",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-16 items-center justify-between gap-2 border-b border-border px-5">
          <div className="flex items-center gap-2.5">
            <span className="grid h-9 w-9 place-items-center rounded-xl bg-accent text-accent-foreground shadow-glow">
              <Layers className="h-5 w-5" />
            </span>
            <div className="leading-tight">
              <p className="text-sm font-semibold text-foreground">
                DocIE <span className="text-accent">Studio</span>
              </p>
              <p className="text-[11px] text-muted-foreground">Control panel</p>
            </div>
          </div>
          <button
            type="button"
            className="rounded-lg p-1.5 text-muted-foreground hover:text-foreground lg:hidden"
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-3" aria-label="Primary">
          {NAV.map((item) => {
            const Icon = item.icon;
            const isActive = item.id === active;
            return (
              <button
                key={item.id}
                onClick={() => {
                  setActive(item.id);
                  setMobileOpen(false);
                }}
                aria-current={isActive ? "page" : undefined}
                className={cn(
                  "group flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-left transition",
                  isActive
                    ? "bg-accent/10 text-accent"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground",
                )}
              >
                <Icon
                  className={cn(
                    "h-[18px] w-[18px] shrink-0",
                    isActive ? "text-accent" : "text-muted-foreground group-hover:text-foreground",
                  )}
                />
                <span className="min-w-0">
                  <span className="block text-sm font-medium">{item.label}</span>
                  <span className="block truncate text-[11px] text-muted-foreground">
                    {item.desc}
                  </span>
                </span>
              </button>
            );
          })}
        </nav>

        <div className="border-t border-border p-3">
          <div className="flex items-center gap-2 rounded-xl border border-border bg-muted/50 px-3 py-2">
            <StatusDot tone={HEALTH_META[health].tone} pulse={health !== "offline"} />
            <div className="min-w-0">
              <p className="truncate text-xs font-medium text-foreground">
                {HEALTH_META[health].label}
              </p>
              <p className="truncate text-[11px] text-muted-foreground" title={API_BASE}>
                {API_BASE.replace(/^https?:\/\//, "")}
              </p>
            </div>
          </div>
        </div>
      </aside>

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-3 border-b border-border bg-card/80 px-4 backdrop-blur sm:px-6">
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="rounded-lg p-1.5 text-muted-foreground hover:text-foreground lg:hidden"
              onClick={() => setMobileOpen(true)}
              aria-label="Open navigation"
            >
              <Menu className="h-5 w-5" />
            </button>
            <div>
              <h1 className="text-base font-semibold text-foreground">{current.label}</h1>
              <p className="hidden text-xs text-muted-foreground sm:block">{current.desc}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <code className="hidden max-w-[260px] truncate rounded-lg border border-border bg-muted px-2.5 py-1.5 text-xs text-muted-foreground md:inline-block">
              {API_BASE}
            </code>
            <ThemeToggle />
          </div>
        </header>

        <main className="flex-1 bg-grid">
          <div className="mx-auto max-w-6xl p-4 sm:p-6">
            {/* All sections stay mounted; only the active one is shown so a
                running extraction (or any in-flight job) survives nav changes. */}
            {NAV.map((item) => (
              <div
                key={item.id}
                hidden={item.id !== active}
                className={item.id === active ? "animate-fade-in" : undefined}
              >
                {item.id === "playground" && <Playground active={active === "playground"} />}
                {item.id === "deploy" && <Deploy active={active === "deploy"} />}
                {item.id === "benchmark" && <Benchmark />}
                {item.id === "observability" && <Observability />}
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
