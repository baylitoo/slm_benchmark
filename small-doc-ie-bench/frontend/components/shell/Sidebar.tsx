"use client";

import { useState } from "react";
import { Layers, ChevronDown, ChevronsLeft, X } from "lucide-react";
import { API_BASE } from "@/lib/env";
import type { Health } from "@/lib/useBackendHealth";
import { cn } from "@/lib/cn";
import { Badge, StatusDot } from "../ui";
import { NavItem } from "./NavItem";
import { NAV_GROUPS, type NavItem as NavItemData, type SectionId } from "./nav";

const HEALTH_META: Record<Health, { tone: "ok" | "warn" | "err"; label: string }> = {
  checking: { tone: "warn", label: "Connecting" },
  online: { tone: "ok", label: "Backend online" },
  offline: { tone: "err", label: "Backend offline" },
};

/** Is this grouped nav item the active one (matches section + optional view)? */
function isItemActive(item: NavItemData, active: SectionId, view: string): boolean {
  if (item.id !== active) return false;
  if (item.view == null) return true;
  return item.view === view;
}

/**
 * Persistent left sidebar (LiteLLM style): logo + version pill, grouped nav
 * under gray uppercase headers, active indigo pill, collapse rail, and a
 * hairline backend-health footer. Presentation only — clicks route through
 * `onNavigate`, which is the existing `setActive`.
 */
export function Sidebar({
  active,
  view,
  onNavigate,
  collapsed,
  onToggleCollapse,
  mobileOpen,
  onCloseMobile,
  health,
}: {
  active: SectionId;
  view: string;
  onNavigate: (id: SectionId, view?: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
  mobileOpen: boolean;
  onCloseMobile: () => void;
  health: Health;
}) {
  // Which groups are expanded (presentation only). All open by default.
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const isOpen = (heading: string) => openGroups[heading] !== false;
  const toggleGroup = (heading: string) =>
    setOpenGroups((g) => ({ ...g, [heading]: !isOpen(heading) }));

  const meta = HEALTH_META[health];

  return (
    <aside
      className={cn(
        "fixed inset-y-0 left-0 z-40 flex flex-col border-r border-border bg-card transition-all lg:static lg:translate-x-0",
        collapsed ? "w-14" : "w-60",
        mobileOpen ? "translate-x-0" : "-translate-x-full",
      )}
    >
      {/* Header: logo + wordmark + version pill */}
      <div className="flex h-14 items-center gap-2 border-b border-border px-3">
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-accent text-accent-foreground">
          <Layers className="h-4 w-4" />
        </span>
        {!collapsed && (
          <>
            <span className="truncate text-sm font-semibold text-foreground">
              DocIE Studio
            </span>
            <Badge tone="neutral" className="ml-auto font-mono text-[10px]">
              v0.1
            </Badge>
          </>
        )}
        <button
          type="button"
          onClick={onCloseMobile}
          aria-label="Close navigation"
          className="ml-auto rounded-md p-1 text-muted-foreground hover:bg-muted hover:text-foreground lg:hidden"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Grouped nav */}
      <nav className="scroll-thin flex-1 space-y-1 overflow-y-auto px-2 py-3" aria-label="Primary">
        {NAV_GROUPS.map((group) => {
          const open = isOpen(group.heading);
          return (
            <div key={group.heading} className="pb-1">
              {!collapsed && (
                <button
                  type="button"
                  onClick={() => toggleGroup(group.heading)}
                  className="flex w-full items-center justify-between px-3 pb-1 pt-3 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground"
                >
                  {group.heading}
                  <ChevronDown
                    className={cn(
                      "h-3.5 w-3.5 transition-transform",
                      open ? "" : "-rotate-90",
                    )}
                  />
                </button>
              )}
              {(collapsed || open) && (
                <div className="space-y-0.5">
                  {group.items.map((item) => (
                    <NavItem
                      key={`${item.id}:${item.view ?? ""}`}
                      item={item}
                      active={isItemActive(item, active, view)}
                      collapsed={collapsed}
                      onClick={() => {
                        onNavigate(item.id, item.view);
                        onCloseMobile();
                      }}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </nav>

      {/* Collapse toggle (desktop) */}
      <button
        type="button"
        onClick={onToggleCollapse}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        className="hidden items-center gap-2 border-t border-border px-3 py-2 text-xs text-muted-foreground hover:bg-muted hover:text-foreground lg:flex"
      >
        <ChevronsLeft className={cn("h-4 w-4 transition-transform", collapsed && "rotate-180")} />
        {!collapsed && <span>Collapse</span>}
      </button>

      {/* Backend health footer */}
      <div className="border-t border-border px-3 py-2.5">
        <div className={cn("flex items-center gap-2", collapsed && "justify-center")}>
          <StatusDot tone={meta.tone} pulse={health !== "offline"} />
          {!collapsed && (
            <div className="min-w-0">
              <p className="truncate text-xs font-medium text-foreground">{meta.label}</p>
              <p className="truncate text-[11px] text-muted-foreground" title={API_BASE}>
                {API_BASE.replace(/^https?:\/\//, "")}
              </p>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
