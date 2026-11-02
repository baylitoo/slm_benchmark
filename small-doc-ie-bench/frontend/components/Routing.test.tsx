import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Routing, buildPolicy, candidatesFrom } from "@/components/Routing";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getDeployments: vi.fn(),
    listRoutingPolicies: vi.fn(),
    createRoutingPolicy: vi.fn(),
    deleteRoutingPolicy: vi.fn(),
  };
});

function record(overrides: Partial<DeploymentRecord> = {}): DeploymentRecord {
  return {
    spec: { name: "lfm2.5-350m", launch: { runtime: "llamacpp" } },
    state: "ready",
    observed: { phase: "hot" },
    replica_group: "lfm2.5-350m",
    ...overrides,
  } as DeploymentRecord;
}

// --- candidates ------------------------------------------------------------

describe("candidatesFrom", () => {
  it("collapses replicas of one model into a single choice", () => {
    // Three records, one routable name: a policy names the model, so offering
    // three entries would invite three stages that are all the same stage.
    const candidates = candidatesFrom([
      record({ spec: { name: "lfm3-1.7b-63aac1" }, replica_group: "lfm3-1.7b" }),
      record({ spec: { name: "lfm3-1.7b-6da942" }, replica_group: "lfm3-1.7b" }),
      record({ spec: { name: "lfm3-1.7b-c684a0" }, replica_group: "lfm3-1.7b" }),
    ]);

    expect(candidates).toHaveLength(1);
    expect(candidates[0].model).toBe("lfm3-1.7b");
    expect(candidates[0].replicas).toBe(3);
  });

  it("offers a model that is deployed but not hot", () => {
    // Stages resolve per request, so a cold stage loads when the ladder
    // reaches it. Filtering these out would hide most of the catalogue.
    const candidates = candidatesFrom([
      record({
        spec: { name: "lfm2.5-1.2b-instruct" },
        replica_group: "lfm2.5-1.2b-instruct",
        state: "stopped",
        observed: { phase: "evicted" },
      }),
    ]);

    expect(candidates.map((c) => c.model)).toEqual(["lfm2.5-1.2b-instruct"]);
    expect(candidates[0].phase).toBe("evicted");
  });

  it("reports the liveliest replica, not the last one seen", () => {
    // One hot replica makes the model reachable without a load; showing it as
    // evicted would misstate what the stage costs.
    const candidates = candidatesFrom([
      record({ spec: { name: "a-1" }, replica_group: "a", observed: { phase: "evicted" } }),
      record({ spec: { name: "a-2" }, replica_group: "a", observed: { phase: "hot" } }),
      record({ spec: { name: "a-3" }, replica_group: "a", observed: { phase: "evicted" } }),
    ]);

    expect(candidates[0].phase).toBe("hot");
  });

  it("treats an unscaled deployment with no group as its own model", () => {
    const candidates = candidatesFrom([
      record({ spec: { name: "solo" }, replica_group: null }),
    ]);

    expect(candidates.map((c) => c.model)).toEqual(["solo"]);
  });

  it("ignores a record with no name at all", () => {
    expect(candidatesFrom([{ } as DeploymentRecord])).toEqual([]);
  });
});

// --- the generated policy --------------------------------------------------

describe("buildPolicy", () => {
  it("falls through every stage but the last, which escalates", () => {
    // The last stage has nothing below it. Accepting its failure would turn a
    // routing policy into one model with extra latency.
    const policy = buildPolicy([{ model: "small" }, { model: "medium" }, { model: "large" }]);
    const stages = policy.stages as Array<Record<string, unknown>>;

    expect(stages.map((s) => s.name)).toEqual(["small", "medium", "large"]);
    expect(stages.map((s) => s.default_decision)).toEqual([
      "fallback",
      "fallback",
      "escalate",
    ]);
    expect(policy.exhausted_decision).toBe("escalate");
  });

  it("accepts a valid extraction when no confidence gate is set", () => {
    const policy = buildPolicy([{ model: "only" }]);
    const rule = (policy.stages as Array<{ rules: Array<Record<string, unknown>> }>)[0].rules[0];

    expect(rule.when).toEqual({ status: "success", validation_valid: true });
    expect(rule.decision).toBe("accept");
  });

  it("adds min_confidence only to the stages that asked for one", () => {
    const policy = buildPolicy([
      { model: "small", minConfidence: 0.85 },
      { model: "large" },
    ]);
    const rules = (policy.stages as Array<{ rules: Array<{ when: Record<string, unknown> }> }>)
      .map((stage) => stage.rules[0].when);

    expect(rules[0]).toHaveProperty("min_confidence", 0.85);
    expect(rules[1]).not.toHaveProperty("min_confidence");
  });

  it("budgets exactly the ladder's length", () => {
    // A budget below the ladder makes later stages unreachable, which reads as
    // a broken policy rather than as a budget doing its job.
    const policy = buildPolicy([{ model: "a" }, { model: "b" }]);

    expect(policy.budget).toEqual({ max_stages: 2 });
  });
});

// --- the page ---------------------------------------------------------------

function renderRouting() {
  return render(
    <ToastProvider>
      <Routing />
    </ToastProvider>,
  );
}

describe("Routing", () => {
  beforeEach(() => {
    vi.mocked(api.listRoutingPolicies).mockResolvedValue([]);
    vi.mocked(api.createRoutingPolicy).mockReset();
  });

  it("saves the ladder in the order it was built", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([
      record({ spec: { name: "small" }, replica_group: "small" }),
      record({ spec: { name: "large" }, replica_group: "large" }),
    ]);
    vi.mocked(api.createRoutingPolicy).mockResolvedValue({
      name: "cascade",
      policy: {},
      created_at: "",
      updated_at: "",
    });
    const user = userEvent.setup();
    renderRouting();

    await waitFor(() => expect(screen.getByText("small")).toBeInTheDocument());
    await user.click(screen.getByLabelText("Add small to the ladder"));
    await user.click(screen.getByLabelText("Add large to the ladder"));
    await user.type(screen.getByLabelText(/Policy name/i), "cascade");
    await user.click(screen.getByRole("button", { name: /Save policy/i }));

    await waitFor(() => expect(api.createRoutingPolicy).toHaveBeenCalled());
    const [name, policy] = vi.mocked(api.createRoutingPolicy).mock.calls[0];
    expect(name).toBe("cascade");
    expect((policy.stages as Array<{ name: string }>).map((s) => s.name)).toEqual([
      "small",
      "large",
    ]);
  });

  it("will not save a ladder with no name", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([
      record({ spec: { name: "small" }, replica_group: "small" }),
    ]);
    const user = userEvent.setup();
    renderRouting();

    await waitFor(() => expect(screen.getByText("small")).toBeInTheDocument());
    await user.click(screen.getByLabelText("Add small to the ladder"));

    expect(screen.getByRole("button", { name: /Save policy/i })).toBeDisabled();
  });

  it("says why there is nothing to route when nothing is deployed", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([]);
    renderRouting();

    await waitFor(() =>
      expect(screen.getByText(/nothing to route between/i)).toBeInTheDocument(),
    );
  });
});
