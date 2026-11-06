import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { NodeBudgets, parseGib } from "@/components/NodeBudgets";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import type { FleetNode } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, getFleet: vi.fn(), setNodeBudget: vi.fn() };
});

const GIB = 1024 ** 3;

function node(overrides: Partial<FleetNode> = {}): FleetNode {
  return {
    node_id: "ovh-cpu-02",
    total_bytes: 6 * GIB,
    free_bytes: 2 * GIB,
    source: "cgroup",
    sum_rss_bytes: GIB,
    reporting: true,
    never_reported: false,
    deployments: 1,
    ceiling_bytes: 8 * GIB,
    working_set_bytes: 4 * GIB,
    effective_budget_bytes: 6 * GIB,
    budget_source: "catalog",
    budget_stale: false,
    mem_budget_bytes: 6 * GIB,
    mem_budget_version: 3,
    ...overrides,
  } as FleetNode;
}

function renderWith(nodes: FleetNode[]) {
  vi.mocked(api.getFleet).mockResolvedValue({ nodes, observed_available: true });
  return render(
    <ToastProvider>
      <NodeBudgets />
    </ToastProvider>,
  );
}

describe("NodeBudgets", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.setNodeBudget).mockResolvedValue({
      node_id: "ovh-cpu-02",
      mem_budget_bytes: 5 * GIB,
      mem_budget_version: 4,
      mem_budget_updated_at: null,
      mem_budget_actor: null,
    });
  });

  it("shows ceiling, budget, usage and headroom", async () => {
    renderWith([node()]);

    expect(await screen.findByText("8.0 GB")).toBeInTheDocument();
    expect(screen.getByText("6.0 GB")).toBeInTheDocument();
    expect(screen.getByText("4.0 GB")).toBeInTheDocument();
    expect(screen.getByText("2.0 GB")).toBeInTheDocument();
    expect(screen.getByText("set here")).toBeInTheDocument();
  });

  it("marks a host with no kernel limit as budget-only", async () => {
    renderWith([node({ ceiling_bytes: null, source: "vm" })]);

    expect(await screen.findByText("none — budget only")).toBeInTheDocument();
  });

  it("names the overage instead of showing zero headroom", async () => {
    renderWith([node({ working_set_bytes: 7 * GIB, free_bytes: 0 })]);

    expect(await screen.findByText("over by 1.0 GB")).toBeInTheDocument();
  });

  it("does not present an unreachable node's last numbers as current", async () => {
    renderWith([node({ reporting: false })]);

    expect(await screen.findByText("unreachable")).toBeInTheDocument();
    expect(screen.queryByText("4.0 GB")).not.toBeInTheDocument();
    expect(screen.queryByText("8.0 GB")).not.toBeInTheDocument();
  });

  it("marks an override the node has not applied yet", async () => {
    renderWith([node({ mem_budget_bytes: 5 * GIB })]);

    expect(await screen.findByText("applying 5.0 GB")).toBeInTheDocument();
  });

  it("flags a budget kept through a catalog outage", async () => {
    renderWith([node({ budget_source: "catalog_cache", budget_stale: true })]);

    expect(await screen.findByText("catalog unreachable")).toBeInTheDocument();
  });

  it("sends the new budget with the version it read", async () => {
    const user = userEvent.setup();
    renderWith([node()]);

    await user.type(await screen.findByLabelText(/New budget for ovh-cpu-02/), "5");
    await user.click(screen.getByRole("button", { name: "Set" }));

    await waitFor(() =>
      expect(api.setNodeBudget).toHaveBeenCalledWith("ovh-cpu-02", 5 * GIB, 3),
    );
  });

  it("clears the override", async () => {
    const user = userEvent.setup();
    renderWith([node()]);

    await user.click(await screen.findByRole("button", { name: "Clear" }));

    await waitFor(() =>
      expect(api.setNodeBudget).toHaveBeenCalledWith("ovh-cpu-02", null, 3),
    );
  });

  it("says an operator key is needed on 403", async () => {
    const user = userEvent.setup();
    vi.mocked(api.setNodeBudget).mockRejectedValueOnce(new ApiError(403, "forbidden"));
    renderWith([node()]);

    await user.type(await screen.findByLabelText(/New budget for ovh-cpu-02/), "5");
    await user.click(screen.getByRole("button", { name: "Set" }));

    expect(await screen.findByText(/needs an operator key/)).toBeInTheDocument();
  });
});

describe("parseGib", () => {
  it("accepts decimals and commas, refuses zero and junk", () => {
    expect(parseGib("1.5")).toBe(1.5 * GIB);
    expect(parseGib("1,5")).toBe(1.5 * GIB);
    expect(parseGib("0")).toBeNull();
    expect(parseGib("abc")).toBeNull();
    expect(parseGib("")).toBeNull();
  });
});
