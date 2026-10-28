import { render, screen } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Benchmark } from "@/components/Benchmark";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getBenchmarks: vi.fn(async () => []),
    listDatasets: vi.fn(async () => [{ name: "invoices", latest: "1.0.0", versions: ["1.0.0"], documents: 40 }]),
    getDeployments: vi.fn(async () => []),
    selectableDeployments: vi.fn(() => [{ spec: { name: "lfm2.5-350m" } }]),
    listSchemas: vi.fn(async () => ["invoice"]),
    listModelProfiles: vi.fn(async () => []),
    listRoutingPolicies: vi.fn(async () => [{ name: "fast-then-strong" }]),
    triggerBenchmark: vi.fn(async () => ({ event_ids: ["ev-1"], channel: "benchmark:ev-1", topics: ["progress"] })),
  };
});

// The live progress panel subscribes to realtime; it is not under test here.
vi.mock("@/components/ResultPanel", () => ({ ResultPanel: () => null }));

function renderRun() {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <Benchmark view="run" />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("Benchmark run", () => {
  it("sends the model, or only the routing policy once the toggle switches", async () => {
    renderRun();
    const datasetSelect = await screen.findByRole("combobox", { name: "Dataset" });
    await screen.findByRole("option", { name: /invoices/ });
    await userEvent.selectOptions(datasetSelect, "invoices");
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Model" }), "lfm2.5-350m");
    await userEvent.click(screen.getByRole("button", { name: /Start benchmark/ }));
    expect(api.triggerBenchmark).toHaveBeenLastCalledWith(
      expect.objectContaining({ dataset: "invoices", model_profile: "lfm2.5-350m" }),
    );

    await userEvent.click(screen.getByRole("button", { name: /Routing policy/ }));
    expect(screen.queryByRole("combobox", { name: "Model" })).not.toBeInTheDocument();
    await screen.findByRole("option", { name: "fast-then-strong" });
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Routing policy" }), "fast-then-strong");
    await userEvent.click(screen.getByRole("button", { name: /Start benchmark/ }));
    const payload = vi.mocked(api.triggerBenchmark).mock.lastCall![0];
    expect(payload).toMatchObject({ routing_policy_name: "fast-then-strong" });
    expect(payload).not.toHaveProperty("model_profile");
  });
});
