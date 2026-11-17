import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreateView } from "@/components/Agents";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { AgentTemplate, DeploymentRecord } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getDeployments: vi.fn(async () => []),
    getStore: vi.fn(async () => []),
    listSchemas: vi.fn(async () => []),
    listDynamicSchemas: vi.fn(async () => []),
    listRoutingPolicies: vi.fn(async () => []),
    listMcpServers: vi.fn(async () => []),
    deployModel: vi.fn(async () => ({ event_ids: [], channel: null, topics: [] })),
  };
});

const PROXY: AgentTemplate = {
  id: "proxy-security",
  kind: "proxy_security",
  display_name: "Security proxy",
  description: "",
};

const PLACED_GUARD: DeploymentRecord = {
  spec: { name: "gliner-pii-f09ae4", launch: { runtime: "encoder", alias: "gliner-pii" } },
  replica_group: "gliner-pii",
  node_id: "rocky-llm",
  state: "ready",
  endpoint: "http://rocky-llm:8090/v1",
};

function renderProxy() {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <CreateView templates={[PROXY]} prefill={PROXY} editAgent={null} onCreated={vi.fn()} />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("Agents CreateView — seeded guard encoder", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getStore).mockResolvedValue([
      { name: "gliner-pii", family: "gliner", analyzer: true },
    ]);
  });

  it("deploys by model, without a fixed name, and points the guard at store:<model>", async () => {
    renderProxy();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "gliner-pii" }));

    await waitFor(() => expect(api.deployModel).toHaveBeenCalledWith({ model: "gliner-pii" }));
    expect(screen.getByLabelText("Guard model")).toHaveValue("store:gliner-pii");
  });

  it("does not offer a model already placed on a node under a minted name", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([PLACED_GUARD]);
    renderProxy();

    await screen.findByRole("option", { name: /gliner-pii-f09ae4/ });
    expect(screen.queryByRole("button", { name: "gliner-pii" })).toBeNull();
  });
});
