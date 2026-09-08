import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeploymentsView } from "@/components/deploy/DeploymentsView";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord, StoreEntry, WhatIfView } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    resizeDeployment: vi.fn(),
    whatifSizing: vi.fn(),
    getPorts: vi.fn(),
  };
});

function polling<T>(data: T): PollingState<T> {
  return {
    data,
    error: null,
    loading: false,
    refreshing: false,
    lastUpdated: Date.now(),
    live: true,
    refresh: vi.fn(),
  };
}

function makeDeployment(overrides: Partial<DeploymentRecord> = {}): DeploymentRecord {
  return {
    spec: {
      name: "nuextract3",
      launch: {
        runtime: "llamacpp",
        model: "/store/nuextract3/model.gguf",
        alias: "nuextract3",
        port: 8090,
        context_length: 8192,
      },
      desired_state: "running",
    },
    state: "ready",
    endpoint: "http://127.0.0.1:8090/v1",
    ...overrides,
  };
}

function renderView(deployment: DeploymentRecord) {
  return render(
    <ToastProvider>
      <DeploymentsView
        deployments={polling<DeploymentRecord[]>([deployment])}
        embeddingNames={new Set()}
        rerankerNames={new Set()}
        store={polling<StoreEntry[]>([])}
      />
    </ToastProvider>,
  );
}

describe("DeploymentsView resize action", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("shows a Resize action for a llama.cpp deployment", async () => {
    renderView(makeDeployment());
    expect(await screen.findByTitle(/resize/i)).toBeInTheDocument();
  });

  it("hides the Resize action for a non-llamacpp deployment (e.g. an encoder)", () => {
    renderView(
      makeDeployment({
        spec: {
          name: "guardrails-pii",
          launch: { runtime: "encoder", model: "org/gliner", alias: "guardrails-pii" },
          desired_state: "running",
        },
      }),
    );
    expect(screen.queryByTitle(/resize/i)).not.toBeInTheDocument();
  });

  it("opens the resize dialog, previews RAM via whatifSizing, and confirms the resize", async () => {
    vi.mocked(api.whatifSizing).mockResolvedValue({
      observed_available: true,
      total_predicted_bytes: 4_000_000_000,
      remaining_bytes: 2_000_000_000,
      ok: true,
      deficit_bytes: 0,
      per_item: [],
    } as WhatIfView);
    vi.mocked(api.resizeDeployment).mockResolvedValue({
      event_ids: ["evt-1"],
      channel: "resize:abc",
      name: "nuextract3",
    });

    renderView(makeDeployment());
    const user = userEvent.setup();

    await user.click(await screen.findByTitle(/resize/i));
    expect(await screen.findByText("Resize deployment")).toBeInTheDocument();

    const input = screen.getByLabelText(/target context window/i);
    await user.clear(input);
    await user.type(input, "32768");

    await waitFor(() =>
      expect(api.whatifSizing).toHaveBeenCalledWith([
        { model: "nuextract3", instances: 1, context_length: 32768 },
      ]),
    );
    expect(await screen.findByText(/fits/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /resize \(zero downtime\)/i }));

    await waitFor(() =>
      expect(api.resizeDeployment).toHaveBeenCalledWith("nuextract3", 32768),
    );
  });

  it("surfaces a deficit verdict when the target context would not fit", async () => {
    vi.mocked(api.whatifSizing).mockResolvedValue({
      observed_available: true,
      total_predicted_bytes: 40_000_000_000,
      remaining_bytes: null,
      ok: false,
      deficit_bytes: 12_000_000_000,
      per_item: [],
    } as WhatIfView);

    renderView(makeDeployment());
    const user = userEvent.setup();

    await user.click(await screen.findByTitle(/resize/i));
    const input = screen.getByLabelText(/target context window/i);
    await user.clear(input);
    await user.type(input, "1000000");

    expect(await screen.findByText(/does not fit/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /resize \(zero downtime\)/i }),
    ).toBeDisabled();
  });
});
