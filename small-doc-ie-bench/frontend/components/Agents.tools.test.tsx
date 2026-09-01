import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreateView } from "@/components/Agents";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";

// CreateView's kind-agnostic hooks (deployments/store/schemas/routing
// policies) all need SOME mock or they'd hit a real network call in jsdom;
// only listMcpServers/testMcpServer/createAgent matter to this suite.
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getDeployments: vi.fn(async () => []),
    getStore: vi.fn(async () => []),
    listSchemas: vi.fn(async () => []),
    listDynamicSchemas: vi.fn(async () => []),
    listRoutingPolicies: vi.fn(async () => []),
    listMcpServers: vi.fn(),
    testMcpServer: vi.fn(),
    createAgent: vi.fn(),
  };
});

function renderCreate() {
  // Fresh SWR cache per render -- CreateView's useAsync hooks share
  // module-global keys otherwise.
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <CreateView templates={[]} prefill={null} editAgent={null} onCreated={vi.fn()} />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("Agents CreateView — custom-kind Tools section", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "calc", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    vi.mocked(api.testMcpServer).mockResolvedValue({
      name: "calc",
      ok: true,
      tools: [
        { name: "add", description: "Add two numbers", input_schema: {} },
        { name: "subtract", description: "Subtract two numbers", input_schema: {} },
      ],
    });
    vi.mocked(api.createAgent).mockResolvedValue({
      name: "helper",
      kind: "custom",
      display_name: "",
      description: "",
      model_profile: null,
      system_prompt: null,
      options: {},
      enabled: true,
      created_at: "",
      updated_at: "",
    });
  });

  it("selecting a server lists its live tools as checked-by-default checkboxes", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "calc" }));

    await waitFor(() => expect(api.testMcpServer).toHaveBeenCalledWith("calc"));
    const add = await screen.findByRole("checkbox", { name: "add" });
    const subtract = await screen.findByRole("checkbox", { name: "subtract" });
    expect(add).toBeChecked();
    expect(subtract).toBeChecked();
  });

  it("submits mcp_servers unrestricted when no tool is unchecked", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "calc" }));
    await screen.findByRole("checkbox", { name: "add" });

    await user.type(screen.getByPlaceholderText("pii-proxy"), "helper");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    await waitFor(() => expect(api.createAgent).toHaveBeenCalled());
    const payload = vi.mocked(api.createAgent).mock.calls[0][0];
    expect(payload.options).toMatchObject({ mcp_servers: ["calc"], mcp_tools: null });
  });

  it("unchecking a tool restricts the allowlist for just that server", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "calc" }));
    await user.click(await screen.findByRole("checkbox", { name: "subtract" }));

    await user.type(screen.getByPlaceholderText("pii-proxy"), "helper");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    await waitFor(() => expect(api.createAgent).toHaveBeenCalled());
    const payload = vi.mocked(api.createAgent).mock.calls[0][0];
    expect(payload.options).toMatchObject({ mcp_tools: { calc: ["add"] } });
  });

  it("deselecting a server removes it (and its restriction) from the submission", async () => {
    renderCreate();
    const user = userEvent.setup();
    const chip = await screen.findByRole("button", { name: "calc" });
    await user.click(chip);
    await user.click(await screen.findByRole("checkbox", { name: "subtract" }));
    await user.click(chip); // toggle off

    await user.type(screen.getByPlaceholderText("pii-proxy"), "helper");
    await user.click(screen.getByRole("button", { name: /create agent/i }));

    await waitFor(() => expect(api.createAgent).toHaveBeenCalled());
    const payload = vi.mocked(api.createAgent).mock.calls[0][0];
    expect(payload.options).toMatchObject({ mcp_servers: null, mcp_tools: null });
  });
});
