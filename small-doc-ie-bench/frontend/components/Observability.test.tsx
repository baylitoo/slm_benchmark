import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { UsageCard, formatCount, llamaCppRunningDeploymentNames } from "@/components/Observability";
import * as api from "@/lib/api";
import type { DeploymentRecord, UsageDeployment } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getUsageSummary: vi.fn(),
  };
});

function makeEntry(overrides: Partial<UsageDeployment> = {}): UsageDeployment {
  return {
    deployment: "lfm2.5-350m",
    requests: 12,
    errors: 0,
    prompt_tokens: 3400,
    completion_tokens: 800,
    avg_latency_ms: 240.5,
    p95_latency_ms: 910,
    last_used_at: new Date().toISOString(),
    tool_calls: [],
    ...overrides,
  };
}

describe("formatCount", () => {
  it("keeps small counts verbatim and compacts thousands/millions", () => {
    expect(formatCount(0)).toBe("0");
    expect(formatCount(950)).toBe("950");
    expect(formatCount(1000)).toBe("1k");
    expect(formatCount(12400)).toBe("12.4k");
    expect(formatCount(3_200_000)).toBe("3.2M");
  });
});

describe("llamaCppRunningDeploymentNames", () => {
  function makeRecord(overrides: Partial<DeploymentRecord> = {}): DeploymentRecord {
    return {
      spec: { name: "lfm2.5-350m", launch: { runtime: "llamacpp" } },
      state: "ready",
      ...overrides,
    };
  }

  it("matches a live llama.cpp deployment (state: ready, the real backend value)", () => {
    expect(llamaCppRunningDeploymentNames([makeRecord()])).toEqual(["lfm2.5-350m"]);
  });

  it("excludes a non-llamacpp runtime even when ready", () => {
    const record = makeRecord({ spec: { name: "x", launch: { runtime: "vllm" } } });
    expect(llamaCppRunningDeploymentNames([record])).toEqual([]);
  });

  it("excludes a llamacpp deployment that isn't ready (starting/stopped/degraded/failed)", () => {
    for (const state of ["starting", "stopped", "degraded", "failed", "running"]) {
      expect(llamaCppRunningDeploymentNames([makeRecord({ state })])).toEqual([]);
    }
  });

  it("drops a record with no resolvable name", () => {
    const record = makeRecord({ spec: { launch: { runtime: "llamacpp" } } });
    expect(llamaCppRunningDeploymentNames([record])).toEqual([]);
  });
});

describe("UsageCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders per-deployment aggregates from the ledger", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({
      window: "24h",
      deployments: [
        makeEntry(),
        makeEntry({ deployment: "nuextract3", requests: 3, errors: 2, prompt_tokens: 0, completion_tokens: 0 }),
      ],
    });
    render(<UsageCard active />);
    expect(await screen.findAllByText("lfm2.5-350m")).not.toHaveLength(0);
    expect(screen.getAllByText("nuextract3").length).toBeGreaterThan(0);
    // Token columns are compact-formatted.
    expect(screen.getAllByText("3.4k").length).toBeGreaterThan(0);
    // Errors render as a badge when non-zero.
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(api.getUsageSummary).toHaveBeenCalledWith("24h");
  });

  it("refetches with the selected window", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({ window: "24h", deployments: [] });
    render(<UsageCard active />);
    await screen.findByText(/No usage recorded/);
    await userEvent.click(screen.getByRole("button", { name: "7d" }));
    await waitFor(() => expect(api.getUsageSummary).toHaveBeenCalledWith("7d"));
  });

  it("shows the empty state when the window has no rows", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({ window: "24h", deployments: [] });
    render(<UsageCard active />);
    expect(await screen.findByText(/No usage recorded/)).toBeInTheDocument();
  });

  it("expands an agent's tool-call trace on demand", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({
      window: "24h",
      deployments: [
        makeEntry({
          deployment: "calc-helper",
          tool_calls: [
            { tool: "calc__add", calls: 5, errors: 1, avg_latency_ms: 12.5 },
            { tool: "calc__subtract", calls: 2, errors: 0, avg_latency_ms: 8 },
          ],
        }),
      ],
    });
    render(<UsageCard active />);
    await screen.findAllByText("calc-helper");
    // Collapsed by default -- no tool rows visible yet.
    expect(screen.queryByText("calc__add")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /expand calc-helper tool calls/i }));
    expect(await screen.findByText("calc__add")).toBeInTheDocument();
    expect(screen.getByText("calc__subtract")).toBeInTheDocument();

    await userEvent.click(
      screen.getByRole("button", { name: /collapse calc-helper tool calls/i }),
    );
    expect(screen.queryByText("calc__add")).not.toBeInTheDocument();
  });

  it("shows no expand affordance for a deployment with no tool calls", async () => {
    vi.mocked(api.getUsageSummary).mockResolvedValue({ window: "24h", deployments: [makeEntry()] });
    render(<UsageCard active />);
    await screen.findAllByText("lfm2.5-350m");
    expect(screen.queryByRole("button", { name: /tool calls/i })).not.toBeInTheDocument();
  });
});
