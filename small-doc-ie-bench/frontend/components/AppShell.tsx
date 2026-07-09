"use client";

import { useState } from "react";
import { useBackendHealth } from "@/lib/useBackendHealth";
import { Playground } from "./Playground";
import { Deploy } from "./Deploy";
import { Benchmark } from "./Benchmark";
import { Observability } from "./Observability";
import { Sidebar } from "./shell/Sidebar";
import { TopBar } from "./shell/TopBar";
import { SECTIONS, DEFAULT_VIEW, type SectionId } from "./shell/nav";

export function AppShell() {
  // Section switch — the single source of truth for navigation (no router).
  const [active, setActive] = useState<SectionId>("playground");
  // Lightweight deep-link hint for pages that split into sub-views. Purely
  // presentational; never affects data flow.
  const [view, setView] = useState<string>(DEFAULT_VIEW.playground);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const health = useBackendHealth();

  function onNavigate(id: SectionId, nextView?: string) {
    setActive(id);
    setView(nextView ?? DEFAULT_VIEW[id]);
  }

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

      <Sidebar
        active={active}
        view={view}
        onNavigate={onNavigate}
        collapsed={collapsed}
        onToggleCollapse={() => setCollapsed((c) => !c)}
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
        health={health}
      />

      {/* Main column */}
      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          onToggleMobile={() => setMobileOpen(true)}
          onToggleCollapse={() => setCollapsed((c) => !c)}
          health={health}
        />

        <main className="flex-1 overflow-y-auto bg-background">
          <div className="px-6 py-5">
            {/* All FOUR sections stay mounted; only the active one is shown so a
                running extraction / deploy / benchmark survives nav changes.
                Iterates the de-duped SECTIONS list — never the grouped nav —
                so each section (and its pollers) mounts exactly once. */}
            {SECTIONS.map((id) => (
              <div
                key={id}
                hidden={id !== active}
                className={id === active ? "animate-fade-in" : undefined}
              >
                {id === "playground" && <Playground active={active === "playground"} />}
                {id === "deploy" && <Deploy active={active === "deploy"} view={view} />}
                {id === "benchmark" && <Benchmark view={view} />}
                {id === "observability" && <Observability view={view} />}
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
