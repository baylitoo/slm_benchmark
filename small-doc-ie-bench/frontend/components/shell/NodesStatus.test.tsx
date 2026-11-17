import { render, screen } from "@testing-library/react";
import { SWRConfig } from "swr";
import { describe, expect, it, vi } from "vitest";
import * as api from "@/lib/api";
import type { FleetNode, FleetView } from "@/lib/api";
import { NodesStatus, summarizeFleet } from "./NodesStatus";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, getFleet: vi.fn() };
});

function node(node_id: string, reporting: boolean, never_reported = false): FleetNode {
  return { node_id, reporting, never_reported, deployments: 1 } as FleetNode;
}

const fleet = (nodes: FleetNode[], observed_available = true): FleetView => ({
  nodes,
  observed_available,
});

describe("summarizeFleet", () => {
  it("counts a silent node as stale but not one that never started", () => {
    expect(
      summarizeFleet(
        fleet([node("rocky-llm", true), node("ash-llm", false), node("new-box", false, true)]),
      ),
    ).toEqual({ total: 3, reporting: 1, stale: ["ash-llm"] });
  });

  it("shows nothing without a catalog or without nodes", () => {
    expect(summarizeFleet(fleet([node("rocky-llm", true)], false))).toBeNull();
    expect(summarizeFleet(fleet([]))).toBeNull();
    expect(summarizeFleet(undefined)).toBeNull();
  });
});

describe("NodesStatus", () => {
  it("names the stale node in the top bar", async () => {
    vi.mocked(api.getFleet).mockResolvedValue(
      fleet([node("rocky-llm", true), node("ash-llm", false)]),
    );
    render(
      <SWRConfig value={{ provider: () => new Map() }}>
        <NodesStatus />
      </SWRConfig>,
    );
    const pill = await screen.findByTestId("nodes-status");
    expect(pill.textContent).toContain("1/2 nodes");
    expect(pill.textContent).toContain("1 stale");
    expect(pill.getAttribute("title")).toContain("ash-llm");
  });
});
