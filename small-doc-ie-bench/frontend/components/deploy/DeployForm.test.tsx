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

  it("deploys native embeddings through Auto without chat launch controls", async () => {
    const user = userEvent.setup();
    renderForm([{
      name: "embeddinggemma-2",
      family: "embeddinggemma2",
      available_backends: ["dense_embedding"],
    }]);
    await user.click(screen.getByText("embeddinggemma-2"));
    const group = screen.getByRole("radiogroup", { name: "Runtime" });
    expect(within(group).getAllByRole("radio")).toHaveLength(1);
    await openAdvanced(user);
    expect(screen.queryByText("Parallel slots")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Deploy/i }));
    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    const payload = vi.mocked(api.deployModel).mock.calls[0][0];
    expect(payload).not.toHaveProperty("runtime");
    expect(payload).not.toHaveProperty("n_parallel");
  });
});

describe("DeployForm — target node", () => {
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

  it("hides the picker when only one node knows the model, and sends no node_id", async () => {
    const user = userEvent.setup();
    renderForm([
      {
        name: "lfm2.5-350m",
        available_backends: ["llamacpp"],
        nodes: { present: ["node-a"], missing: [], unreported: [] },
      },
    ]);

    await user.click(screen.getByText("lfm2.5-350m"));
    expect(screen.queryByText("Target node")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Deploy/i }));
    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    expect(vi.mocked(api.deployModel).mock.calls[0][0]).not.toHaveProperty("node_id");
  });

  it("offers the nodes that hold the blob, disables the ones that do not, and reports a recorded deploy without mounting progress", async () => {
    const user = userEvent.setup();
    // A targeted deploy is DESIRED STATE: no event, no channel. The form must
    // show the server's sentence instead of a progress panel that would never
    // connect.
    vi.mocked(api.deployModel).mockResolvedValue({
      event_ids: [],
      channel: null,
      topics: [],
      detail: "recorded as desired state for node 'node-b' (generation 4)",
    });
    renderForm([
      {
        name: "lfm2.5-350m",
        available_backends: ["llamacpp"],
        nodes: { present: ["node-a", "node-b"], missing: ["node-c"], unreported: [] },
      },
    ]);

    await user.click(screen.getByText("lfm2.5-350m"));
    const picker = screen.getByLabelText("Target node") as HTMLSelectElement;

    // A node that reported its store WITHOUT the model is refused by the
    // server, so it must not be selectable here.
    const missing = within(picker).getByRole("option", {
      name: /node-c/,
    }) as HTMLOptionElement;
    expect(missing.disabled).toBe(true);

    await user.selectOptions(picker, "node-b");
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    expect(vi.mocked(api.deployModel).mock.calls[0][0].node_id).toBe("node-b");
    expect(
      await screen.findByText(/recorded as desired state for node 'node-b'/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Connecting to live updates/)).not.toBeInTheDocument();
  });
});

describe("DeployForm — auto placement", () => {
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
      event_ids: [],
      channel: null,
      topics: [],
      detail: "placed by fit as desired state: lfm2.5-350m-ab12 on node-b",
    });
  });

  const FLEET: StoreEntry[] = [
    {
      name: "lfm2.5-350m",
      available_backends: ["llamacpp"],
      nodes: { present: ["node-a", "node-b"], missing: [], unreported: [] },
    },
  ];

  it("asks the platform to choose instead of naming a node", async () => {
    const user = userEvent.setup();
    renderForm(FLEET);

    await user.click(screen.getByText("lfm2.5-350m"));
    await user.click(screen.getByLabelText(/Let the platform choose/));
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    const payload = vi.mocked(api.deployModel).mock.calls[0][0];
    expect(payload.auto_place).toBe(true);
    // Naming a node and asking for one to be chosen is a 422; the form must
    // not send both.
    expect(payload).not.toHaveProperty("node_id");
  });

  it("allows replicas above 1, because spreading them is the point", async () => {
    // A TARGETED deploy with replicas > 1 is refused (one target, one node).
    // Auto placement is the opposite case: it spreads them over distinct
    // nodes, so the same guard must not block it.
    const user = userEvent.setup();
    renderForm(FLEET);

    await user.click(screen.getByText("lfm2.5-350m"));
    await user.click(screen.getByLabelText(/Let the platform choose/));
    await openAdvanced(user);
    const replicas = screen.getByLabelText("Replicas");
    await user.clear(replicas);
    await user.type(replicas, "2");
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalled());
    const payload = vi.mocked(api.deployModel).mock.calls[0][0];
    expect(payload.replicas).toBe(2);
    expect(payload.auto_place).toBe(true);
  });

  it("still refuses replicas on a deploy aimed at ONE named node", async () => {
    const user = userEvent.setup();
    renderForm(FLEET);

    await user.click(screen.getByText("lfm2.5-350m"));
    await user.selectOptions(screen.getByLabelText("Target node"), "node-b");
    await openAdvanced(user);
    const replicas = screen.getByLabelText("Replicas");
    await user.clear(replicas);
    await user.type(replicas, "2");
    await user.click(screen.getByRole("button", { name: /Deploy/i }));

    expect(await screen.findByText(/cannot be combined with replicas/)).toBeInTheDocument();
    expect(api.deployModel).not.toHaveBeenCalled();
  });
});
