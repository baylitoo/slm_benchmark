"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useBackendHealth } from "@/lib/useBackendHealth";
import { Playground } from "./Playground";
import { Deploy } from "./Deploy";
import { Agents } from "./Agents";
import { Benchmark } from "./Benchmark";
import { Observability } from "./Observability";
import { Sidebar } from "./shell/Sidebar";
import { TopBar } from "./shell/TopBar";
import { SectionActivityProvider } from "./shell/SectionActivity";
import { SECTIONS, DEFAULT_VIEW, type SectionId } from "./shell/nav";

function isSection(value: string | undefined): value is SectionId {
  return value != null && (SECTIONS as readonly string[]).includes(value);
}

export function AppShell() {
  // The URL is the navigation source of truth: /{section}/{view?}. Deep links,
  // refresh and the back button all work, and each section owns its OWN view —
  // the old single shared `view` string was broadcast to all mounted sections,
  // so navigating to Deploy→Ports silently reset the Agents create form.
  const router = useRouter();
  const pathname = usePathname();
  const [seg0, seg1] = pathname.split("/").filter(Boolean);
  const active: SectionId = isSection(seg0) ? seg0 : "playground";
  const urlView = isSection(seg0) ? seg1 : undefined;

  // Per-section view memory so returning to a section restores its last
  // sub-view even when the nav click carries none.
  const [views, setViews] = useState<Record<SectionId, string>>(() => ({ ...DEFAULT_VIEW }));
  useEffect(() => {
    const next = urlView ?? DEFAULT_VIEW[active];
    setViews((current) => (current[active] === next ? current : { ...current, [active]: next }));
  }, [active, urlView]);

  const [mobileOpen, setMobileOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const health = useBackendHealth();

  function onNavigate(id: SectionId, nextView?: string) {
    const view = nextView ?? views[id] ?? DEFAULT_VIEW[id];
    router.push(view ? `/${id}/${view}` : `/${id}`);
  }

  // The active section reads the URL directly (no one-render lag); hidden
  // sections keep their remembered view untouched.
  const viewFor = (id: SectionId) =>
    id === active ? (urlView ?? views[id] ?? DEFAULT_VIEW[id]) : views[id];

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
        view={viewFor(active)}
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
            {/* All FIVE sections stay mounted; only the active one is shown so a
                running extraction / deploy / benchmark survives nav changes.
                Iterates the de-duped SECTIONS list — never the grouped nav —
                so each section (and its pollers) mounts exactly once. The
                activity provider lets periodic cosmetic work (LiveIndicator
                ticks) pause inside hidden subtrees. */}
            {SECTIONS.map((id) => (
              <div
                key={id}
                hidden={id !== active}
                className={id === active ? "animate-fade-in" : undefined}
              >
                <SectionActivityProvider value={id === active}>
                  {id === "playground" && (
                    <Playground active={active === "playground"} onNavigate={onNavigate} />
                  )}
                  {id === "deploy" && <Deploy active={active === "deploy"} view={viewFor(id)} />}
                  {id === "agents" && (
                    <Agents view={viewFor(id)} onNavigate={onNavigate} />
                  )}
                  {id === "benchmark" && <Benchmark view={viewFor(id)} />}
                  {id === "observability" && (
                    <Observability active={active === "observability"} />
                  )}
                </SectionActivityProvider>
              </div>
            ))}
          </div>
        </main>
      </div>
    </div>
  );
}
