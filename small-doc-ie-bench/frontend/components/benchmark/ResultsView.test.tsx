import { render, screen } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Benchmark } from "@/components/Benchmark";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { BenchmarkRun } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getBenchmarks: vi.fn(),
    listDatasets: vi.fn(async () => []),
    getDeployments: vi.fn(async () => []),
    listSchemas: vi.fn(async () => []),
    listModelProfiles: vi.fn(async () => []),
    listRoutingPolicies: vi.fn(async () => []),
  };
});

const RUNS: BenchmarkRun[] = [
  {
    event_id: "ev-1",
    status: "completed",
    dataset: "invoices@1.0.0",
    model_profile: "lfm2.5-350m",
    created_at: "2026-01-01T00:00:00Z",
    metrics: {
      summary: [
        {
          model_profile: "lfm2.5-350m",
          docs: 40,
          field_accuracy: 0.8125,
          valid_rate: 1,
          hallucination_rate: 0.05,
          evidence_coverage: 0.95,
          p50_latency_ms: 2400,
          p95_latency_ms: 5100,
        },
      ],
    },
  },
  { event_id: "ev-2", status: "failed", dataset: "invoices@1.0.0", model_profile: "nuextract3" },
];

function renderResults() {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <Benchmark view="results" />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("Benchmark results", () => {
  it("shows each run's model and headline scores, and its tiles when opened", async () => {
    vi.mocked(api.getBenchmarks).mockResolvedValue(RUNS);
    renderResults();

    const accuracy = await screen.findAllByText("81.3%");
    expect(accuracy.length).toBeGreaterThan(0);
    expect(screen.getAllByText("lfm2.5-350m").length).toBeGreaterThan(0);
    expect(screen.getByText("2.4 s")).toBeInTheDocument();
    expect(screen.getByText("1 completed · 1 failed")).toBeInTheDocument();

    await userEvent.click(screen.getByText("nuextract3"));
    expect(await screen.findByText("This run has no scores yet.")).toBeInTheDocument();

    await userEvent.click(screen.getAllByText("lfm2.5-350m").at(-1)!);
    expect(await screen.findByText("p95 5.1 s")).toBeInTheDocument();
    expect(screen.getByText("95.0%")).toBeInTheDocument();
  });

  it("filters by status", async () => {
    vi.mocked(api.getBenchmarks).mockResolvedValue(RUNS);
    renderResults();
    await screen.findByText("nuextract3");

    await userEvent.click(screen.getByRole("button", { name: "Completed" }));
    expect(screen.queryByText("nuextract3")).not.toBeInTheDocument();
    expect(screen.getAllByText("lfm2.5-350m").length).toBeGreaterThan(0);
  });
});
