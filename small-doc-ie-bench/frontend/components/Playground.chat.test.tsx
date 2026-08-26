import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SWRConfig } from "swr";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "@/components/Playground";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    chatCompletionStream: vi.fn(),
    chatCompletion: vi.fn(),
    listMcpServers: vi.fn(),
    listDynamicSchemas: vi.fn(),
    listRoutingPolicies: vi.fn(),
    triggerExtract: vi.fn(),
    fileToBase64: vi.fn(async () => "ZmFrZQ=="),
  };
});

// The extraction path mounts ResultPanel, which owns its own SSE/polling
// connection -- irrelevant to what THIS suite tests (that ChatPanel routes
// to /v1/extract and renders the returned trigger). Stub it to a marker.
vi.mock("./ResultPanel", () => ({
  ResultPanel: ({ trigger }: { trigger: { channel: string } }) => (
    <div data-testid="result-panel">result-panel:{trigger.channel}</div>
  ),
}));

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
  };
}

const RECORDS = [makeDeployment("lfm2.5-350m")];

function makePolling(data: DeploymentRecord[]): PollingState<DeploymentRecord[]> {
  return {
    data,
    error: null,
    loading: false,
    refreshing: false,
    lastUpdated: Date.now(),
    live: true,
    refresh: () => {},
  };
}

function renderChat(records: DeploymentRecord[] = RECORDS) {
  // Fresh SWR cache per render -- ChatPanel's useAsync hooks (mcp servers,
  // dynamic schemas, routing policies) share module-global keys otherwise.
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <ChatPanel deployments={makePolling(records)} selectable={records} />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("ChatPanel", () => {
  beforeAll(() => {
    Element.prototype.scrollIntoView = vi.fn();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listMcpServers).mockResolvedValue([]);
    vi.mocked(api.listDynamicSchemas).mockResolvedValue([]);
    vi.mocked(api.listRoutingPolicies).mockResolvedValue([]);
    vi.mocked(api.chatCompletionStream).mockImplementation(async (_model, _messages, onToken) => {
      onToken("hi there");
    });
  });

  it("sends a plain chat message and streams the answer", async () => {
    renderChat();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/Type a message/), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("hi there")).toBeInTheDocument();
    expect(api.chatCompletionStream).toHaveBeenCalledTimes(1);
    const [model, messages] = vi.mocked(api.chatCompletionStream).mock.calls[0];
    expect(model).toBe("lfm2.5-350m");
    expect(messages.at(-1)).toEqual({ role: "user", content: "hello" });
  });

  it("attaches an image and sends it as multimodal content (vision folded into chat)", async () => {
    renderChat();
    const user = userEvent.setup();
    const file = new File(["fake-bytes"], "invoice.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(api.chatCompletionStream).toHaveBeenCalledTimes(1));
    const [, messages] = vi.mocked(api.chatCompletionStream).mock.calls[0];
    const last = messages.at(-1) as { role: string; content: unknown };
    expect(Array.isArray(last.content)).toBe(true);
    const parts = last.content as { type: string; text?: string; image_url?: { url: string } }[];
    expect(parts[0]).toEqual({ type: "text", text: "what's the total?" });
    expect(parts[1].type).toBe("image_url");
    expect(parts[1].image_url?.url).toMatch(/^data:image\/png;base64,/);
  });

  it("routes Send to extraction when the toggle is on, and renders the result inline", async () => {
    vi.mocked(api.triggerExtract).mockResolvedValue({
      event_ids: ["e1"],
      channel: "extract:e1",
      topics: ["result"],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.type(screen.getByPlaceholderText(/Paste document text/), "invoice body text");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    expect(await screen.findByTestId("result-panel")).toHaveTextContent("extract:e1");
    expect(api.triggerExtract).toHaveBeenCalledWith(
      expect.objectContaining({
        schema_name: "invoice",
        deployment: "lfm2.5-350m",
        text: "invoice body text",
      }),
    );
    // Chat's streaming path must NOT have been used for this turn.
    expect(api.chatCompletionStream).not.toHaveBeenCalled();
  });

  it("routes extraction through a routing policy instead of the deployment when selected", async () => {
    vi.mocked(api.listRoutingPolicies).mockResolvedValue([
      { name: "cheap-first", policy: {}, created_at: "", updated_at: "" },
    ]);
    vi.mocked(api.triggerExtract).mockResolvedValue({
      event_ids: ["e2"],
      channel: "extract:e2",
      topics: ["result"],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Routing policy" }));
    await user.selectOptions(await screen.findByLabelText("Routing policy"), "cheap-first");
    await user.type(screen.getByPlaceholderText(/Paste document text/), "body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await waitFor(() =>
      expect(api.triggerExtract).toHaveBeenCalledWith(
        expect.objectContaining({ routing_policy: "cheap-first" }),
      ),
    );
    const call = vi.mocked(api.triggerExtract).mock.calls[0][0];
    expect(call.deployment).toBeUndefined();
  });

  it("hides MCP tool chips while extraction is on", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "calc", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    renderChat();
    expect(await screen.findByText("Tools:")).toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    expect(screen.queryByText("Tools:")).not.toBeInTheDocument();
  });
});
