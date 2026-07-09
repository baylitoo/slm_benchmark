"use client";

import { cn } from "@/lib/cn";
import type { NavItem as NavItemData } from "./nav";

/**
 * A single sidebar nav row: icon + label. Active = soft indigo pill.
 * `onClick` still routes through the existing `setActive` in AppShell.
 */
export function NavItem({
  item,
  active,
  collapsed,
  onClick,
}: {
  item: NavItemData;
  active: boolean;
  collapsed: boolean;
  onClick: () => void;
}) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      title={collapsed ? item.label : undefined}
      className={cn(
        "flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-sm transition",
        collapsed && "justify-center px-0",
        active
          ? "bg-accent/10 font-medium text-accent"
          : "text-foreground/70 hover:bg-muted hover:text-foreground",
      )}
    >
      <Icon className="h-4 w-4 shrink-0" />
      {!collapsed && <span className="min-w-0 truncate">{item.label}</span>}
    </button>
  );
}
