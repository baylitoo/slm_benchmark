import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DeployForm } from "@/components/deploy/DeployForm";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { StoreEntry } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getPorts: vi.fn(),
    deployModel: vi.fn(),
  };
});

function makeStorePolling(entries: StoreEntry[]): PollingState<StoreEntry[]> {
  return {
    data: entries,
    error: null,
    loading: false,
    refreshing: false,
    lastUpdated: Date.now(),
    live: true,
    refresh: () => {},
  };
}

function renderForm(entries: StoreEntry[]) {
  return render(
    <ToastProvider>
      <DeployForm store={makeStorePolling(entries)} active onDeployed={() => {}} />
    </ToastProvider>,
  );
}

async function openAdvanced(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByText("Advanced options"));
}

describe("DeployForm — parallel slots", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getPorts).mockResolvedValue({
      range: { start: 8088, end: 8188 },
      deployments: [],
      used: [],
      free_sample: [8088],
      recommended_next: 8088,
    });
    vi.mocked(api.deployModel).mockResolvedValue({
      event_ids: ["evt-1"],
      channel: "deploy:test",
      topics: [],
    });
  });

  it("shows Parallel slots for an Auto (store-entry) deploy and omits it from the payload at the default of 1", async () => {
    const user = userEvent.setup();
    renderForm([{ name: "lfm2.5-350m", available_backends: ["llamacpp"] }]);

    await user.click(screen.getByText("lfm2.5-350m"));
    await openAdvanced(user);

    expect(screen.getByText("Parallel slots")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    const payload = vi.mocked(api.deployModel).mock.calls[0][0];
    expect(payload).not.toHaveProperty("n_parallel");
  });

  it("sends n_parallel when raised above 1", async () => {
    const user = userEvent.setup();
    renderForm([{ name: "lfm2.5-350m", available_backends: ["llamacpp"] }]);

    await user.click(screen.getByText("lfm2.5-350m"));
    await openAdvanced(user);

    const input = screen.getByLabelText("Parallel slots");
    await user.clear(input);
    await user.type(input, "4");
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    const payload = vi.mocked(api.deployModel).mock.calls[0][0];
    expect(payload.n_parallel).toBe(4);
  });

  it("hides Parallel slots for an explicit non-llamacpp runtime", async () => {
    const user = userEvent.setup();
    renderForm([
      { name: "remote-gpt", available_backends: ["vllm", "llamacpp"] },
    ]);

    await user.click(screen.getByText("remote-gpt"));
    const runtimeGroup = screen.getByRole("radiogroup", { name: "Runtime" });
    await user.click(within(runtimeGroup).getByRole("radio", { name: "vllm" }));
    await openAdvanced(user);

    expect(screen.queryByText("Parallel slots")).not.toBeInTheDocument();
  });

  it("shows Parallel slots again for an explicit llamacpp runtime pick", async () => {
    const user = userEvent.setup();
    renderForm([
      { name: "remote-gpt", available_backends: ["vllm", "llamacpp"] },
    ]);

    await user.click(screen.getByText("remote-gpt"));
    const runtimeGroup = screen.getByRole("radiogroup", { name: "Runtime" });
    await user.click(within(runtimeGroup).getByRole("radio", { name: "llamacpp" }));
    await openAdvanced(user);

    expect(screen.getByText("Parallel slots")).toBeInTheDocument();
  });
});
