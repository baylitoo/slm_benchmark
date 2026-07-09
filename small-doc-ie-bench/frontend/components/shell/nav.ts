// Navigation data for the LiteLLM-style sidebar.
//
// IMPORTANT: two separate structures on purpose.
//   • NAV_GROUPS  — PRESENTATION only (grouped, with duplicate SectionIds for
//                   split sub-views like Models/Deployments). The sidebar reads
//                   this.
//   • SECTIONS    — the flat, de-duped list of the FOUR unique sections. The
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
  Network,
  Play,
  History,
  BarChart3,
  ExternalLink,
  type LucideIcon,
} from "lucide-react";

export type SectionId = "playground" | "deploy" | "benchmark" | "observability";

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
      { id: "deploy", label: "Models", icon: Boxes, view: "models" },
      { id: "deploy", label: "Deployments", icon: Server, view: "deployments" },
      { id: "deploy", label: "Ports", icon: Network, view: "ports" },
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
    heading: "Observability",
    items: [
      { id: "observability", label: "Dashboards", icon: BarChart3, view: "dashboards" },
      { id: "observability", label: "Links", icon: ExternalLink, view: "links" },
    ],
  },
];

/** The FOUR unique sections — the single source of truth for the mount loop. */
export const SECTIONS: SectionId[] = [
  "playground",
  "deploy",
  "benchmark",
  "observability",
];

/** Default sub-view applied when a section is entered via a view-less path. */
export const DEFAULT_VIEW: Record<SectionId, string> = {
  playground: "",
  deploy: "deployments",
  benchmark: "run",
  observability: "dashboards",
};
