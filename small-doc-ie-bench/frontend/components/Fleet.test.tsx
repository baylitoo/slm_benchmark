import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Fleet } from "@/components/Fleet";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import type { FleetNode } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getFleet: vi.fn(),
    getStore: vi.fn(),
    drainNode: vi.fn(),
    resumeNode: vi.fn(),
  };
});

const GIB = 1024 ** 3;

function node(overrides: Partial<FleetNode> & { node_id: string }): FleetNode {
  return {
    total_bytes: 8 * GIB,
    free_bytes: 4 * GIB,
    source: "cgroup",
    sum_rss_bytes: GIB,
    reporting: true,
    never_reported: false,
    deployments: 1,
    drain_state: "admitting",
    ...overrides,
  } as FleetNode;
}

function renderFleet(nodes: FleetNode[]) {
  vi.mocked(api.getFleet).mockResolvedValue({
    nodes,
    observed_available: true,
    stale_after_s: 30,
  });
  vi.mocked(api.getStore).mockResolvedValue([]);
  return render(
    <ToastProvider>
      <Fleet active />
    </ToastProvider>,
  );
}

describe("Fleet — drain", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.drainNode).mockResolvedValue({
      node_id: "node-a",
      drained_at: new Date().toISOString(),
      desired_generation: 2,
      models_offline: [],
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("drains a node that has a replica elsewhere without asking", async () => {
    const user = userEvent.setup();
    renderFleet([node({ node_id: "node-a" }), node({ node_id: "node-b" })]);

    const buttons = await screen.findAllByRole("button", { name: /Drain/i });
    await user.click(buttons[0]);

    await waitFor(() => expect(api.drainNode).toHaveBeenCalledWith("node-a"));
  });

  it("confirms with the server's own sentence before draining the last replica", async () => {
    // The 409 names which models go offline. That sentence IS the decision, so
    // it is what the operator is asked to confirm — not a generic warning the
    // UI made up.
    const user = userEvent.setup();
    const detail =
      "draining 'node-a' would take 1 model(s) offline, because this node " +
      "holds their only live placement: lfm2.5-350m.";
    vi.mocked(api.drainNode).mockRejectedValueOnce(new ApiError(409, detail));
    vi.mocked(api.drainNode).mockResolvedValueOnce({
      node_id: "node-a",
      drained_at: new Date().toISOString(),
      desired_generation: 2,
      models_offline: ["lfm2.5-350m"],
    });
    const confirm = vi.fn().mockReturnValue(true);
    vi.stubGlobal("confirm", confirm);

    renderFleet([node({ node_id: "node-a" }), node({ node_id: "node-b" })]);
    const buttons = await screen.findAllByRole("button", { name: /Drain/i });
    await user.click(buttons[0]);

    await waitFor(() => expect(confirm).toHaveBeenCalled());
    expect(String(confirm.mock.calls[0][0])).toContain("lfm2.5-350m");
    await waitFor(() =>
      expect(api.drainNode).toHaveBeenLastCalledWith("node-a", {
        acknowledgeOutage: true,
      }),
    );
  });

  it("does not drain when the operator declines the outage", async () => {
    const user = userEvent.setup();
    vi.mocked(api.drainNode).mockRejectedValueOnce(
      new ApiError(409, "would take 1 model(s) offline"),
    );
    vi.stubGlobal("confirm", vi.fn().mockReturnValue(false));

    renderFleet([node({ node_id: "node-a" }), node({ node_id: "node-b" })]);
    const buttons = await screen.findAllByRole("button", { name: /Drain/i });
    await user.click(buttons[0]);

    await waitFor(() => expect(api.drainNode).toHaveBeenCalledTimes(1));
    expect(api.drainNode).not.toHaveBeenCalledWith("node-a", {
      acknowledgeOutage: true,
    });
  });

  it("offers Resume on a drained node, and names what the drain is waiting for", async () => {
    const user = userEvent.setup();
    renderFleet([
      node({
        node_id: "node-a",
        drained_at: new Date().toISOString(),
        drain_state: "draining",
        deployments: 2,
      }),
      node({ node_id: "node-b" }),
    ]);

    // "draining", not "drained": the machine is still finishing work, and the
    // distinction is the whole point of the two states.
    expect(await screen.findByText("draining")).toBeInTheDocument();
    expect(screen.getByText("2 still finishing")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Resume/i }));

    await waitFor(() => expect(api.resumeNode).toHaveBeenCalledWith("node-a"));
  });
});
