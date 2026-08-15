// Navigation data for the LiteLLM-style sidebar.
//
// IMPORTANT: two separate structures on purpose.
//   • NAV_GROUPS  — PRESENTATION only (grouped, with duplicate SectionIds for
//                   split sub-views like Models/Deployments). The sidebar reads
//                   this.
//   • SECTIONS    — the flat, de-duped list of the unique sections. The
//                   AppShell mount loop reads THIS so each section (and its
//                   pollers) mounts exactly once.
//
// Clicking a nav item still calls the existing `setActive(id)` — no router, no
// context. An optional `view` is a lightweight deep-link hint the target page
// reads to pick its default sub-view; it never changes data flow.

import {
  FlaskConical,
  Boxes,
  Server,
  Gauge,
  Play,
  History,
  BarChart3,
  Bot,
  LayoutGrid,
  PlusCircle,
  ClipboardCheck,
  Download,
  type LucideIcon,
} from "lucide-react";

export type SectionId =
  | "playground"
  | "deploy"
  | "agents"
  | "benchmark"
  | "review"
  | "observability";

export interface NavItem {
  id: SectionId;
  label: string;
  icon: LucideIcon;
  /** Optional sub-view hint for pages that split one section into tabs. */
  view?: string;
}

export interface NavGroup {
  heading: string;
  items: NavItem[];
}

/** Presentation: the grouped sidebar. May repeat a SectionId across sub-items. */
export const NAV_GROUPS: NavGroup[] = [
  {
    heading: "Serving",
    items: [
      { id: "playground", label: "Playground", icon: FlaskConical },
      { id: "deploy", label: "Catalog", icon: LayoutGrid, view: "catalog" },
      { id: "deploy", label: "Models", icon: Boxes, view: "models" },
      { id: "deploy", label: "Deployments", icon: Server, view: "deployments" },
      { id: "deploy", label: "Downloads", icon: Download, view: "downloads" },
      { id: "deploy", label: "Sizing", icon: Gauge, view: "sizing" },
    ],
  },
  {
    heading: "Agents",
    items: [
      // "Templates", not "Catalog" — Serving's Catalog (HF Hub browse-to-seed)
      // already owns that word; this is the preconfigured agent starting points
      // (Security Proxy, OCR Agent, …), a different concept entirely.
      { id: "agents", label: "Templates", icon: LayoutGrid, view: "catalog" },
      { id: "agents", label: "My Agents", icon: Bot, view: "instances" },
      { id: "agents", label: "Create", icon: PlusCircle, view: "create" },
    ],
  },
  {
    heading: "Benchmark",
    items: [
      { id: "benchmark", label: "Run", icon: Play, view: "run" },
      { id: "benchmark", label: "Results", icon: History, view: "results" },
    ],
  },
  {
    heading: "Review",
    // A single view — the queue table itself carries the status filter, so
    // there's nothing left to split into separate sub-tabs (mirrors
    // Observability/Playground's single-view sections).
    items: [{ id: "review", label: "Queue", icon: ClipboardCheck }],
  },
  {
    heading: "Observability",
    // A single view — no sub-tabs (mirrors Playground): the page combines the
    // quick-link tiles and the embedded dashboard together, so there is
    // nothing left to split into a separate "Links" tab.
    items: [{ id: "observability", label: "Dashboards", icon: BarChart3 }],
  },
];

/** The SIX unique sections — the single source of truth for the mount loop. */
export const SECTIONS: SectionId[] = [
  "playground",
  "deploy",
  "agents",
  "benchmark",
  "review",
  "observability",
];

/** Default sub-view applied when a section is entered via a view-less path. */
export const DEFAULT_VIEW: Record<SectionId, string> = {
  playground: "",
  deploy: "deployments",
  agents: "catalog",
  benchmark: "run",
  review: "",
  observability: "",
};
