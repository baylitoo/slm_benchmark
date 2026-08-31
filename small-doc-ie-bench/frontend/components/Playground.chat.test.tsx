import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SWRConfig } from "swr";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "@/components/Playground";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord, StoreEntry } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    chatCompletionStream: vi.fn(),
    chatCompletionMcpStream: vi.fn(),
    listMcpServers: vi.fn(),
    listSchemas: vi.fn(),
    listDynamicSchemas: vi.fn(),
    getSchemaFields: vi.fn(),
    listRoutingPolicies: vi.fn(),
    triggerExtract: vi.fn(),
    fileToBase64: vi.fn(async () => "ZmFrZQ=="),
    renderDocument: vi.fn(async () => ({ images: ["data:image/png;base64,ZmFrZQ=="], pages: 1 })),
    uploadSessionDocument: vi.fn(),
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

function makePolling<T>(data: T): PollingState<T> {
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

function renderChat(records: DeploymentRecord[] = RECORDS, storeEntries: StoreEntry[] = []) {
  // Fresh SWR cache per render -- ChatPanel's useAsync hooks (mcp servers,
  // dynamic schemas, routing policies) share module-global keys otherwise.
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <ChatPanel
          deployments={makePolling(records)}
          selectable={records}
          store={makePolling(storeEntries)}
        />
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
    vi.mocked(api.listSchemas).mockResolvedValue(["invoice", "identity_card"]);
    vi.mocked(api.listDynamicSchemas).mockResolvedValue([]);
    vi.mocked(api.getSchemaFields).mockResolvedValue(["invoice_number", "vendor_name", "total_ttc"]);
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
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
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

  it("renders only the generic vision presets when no schema is selected", async () => {
    vi.mocked(api.listSchemas).mockResolvedValue([]);
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["fake-bytes"], "invoice.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    expect(
      await screen.findByRole("button", { name: /Extract all the text/ }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Describe this image/ })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /What is written/ })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Extract:/ })).not.toBeInTheDocument();
  });

  it("adds a schema-derived preset naming the selected schema's fields", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["fake-bytes"], "invoice.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    expect(
      await screen.findByRole("button", { name: /^Extract: invoice_number/ }),
    ).toBeInTheDocument();
    expect(api.getSchemaFields).toHaveBeenCalledWith("invoice");
    // The generic presets stay put -- this is additive, not a replacement.
    expect(screen.getByRole("button", { name: /Describe this image/ })).toBeInTheDocument();
  });

  it("populates the message input from the schema-derived preset the same way a generic preset does", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["fake-bytes"], "invoice.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    const preset = await screen.findByRole("button", { name: /^Extract: invoice_number/ });
    await user.click(preset);

    expect(screen.getByPlaceholderText(/Type a message/)).toHaveValue(
      "Extract: invoice_number, vendor_name, total_ttc",
    );
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

  it("lists built-in and saved schemas in one Select, not a free-text field", async () => {
    vi.mocked(api.listDynamicSchemas).mockResolvedValue([
      {
        name: "purchase_order",
        spec: { document_type: "purchase_order", fields: [] },
        created_at: "",
        updated_at: "",
      },
    ]);
    vi.mocked(api.triggerExtract).mockResolvedValue({
      event_ids: ["e3"],
      channel: "extract:e3",
      topics: ["result"],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const select = await screen.findByRole("combobox", { name: "Schema" });
    // Both buckets are real options in ONE list -- no separate free-text
    // field and no chance of a typo'd, non-existent schema name.
    expect(await screen.findByRole("option", { name: "identity_card" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "purchase_order" })).toBeInTheDocument();

    await user.selectOptions(select, "purchase_order");
    await user.type(screen.getByPlaceholderText(/Paste document text/), "po body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await waitFor(() =>
      expect(api.triggerExtract).toHaveBeenCalledWith(
        expect.objectContaining({ dynamic_schema_name: "purchase_order" }),
      ),
    );
  });

  it("defaults to the built-in invoice schema and can switch back from a saved one", async () => {
    vi.mocked(api.listDynamicSchemas).mockResolvedValue([
      {
        name: "purchase_order",
        spec: { document_type: "purchase_order", fields: [] },
        created_at: "",
        updated_at: "",
      },
    ]);
    vi.mocked(api.triggerExtract).mockResolvedValue({
      event_ids: ["e4"],
      channel: "extract:e4",
      topics: ["result"],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const select = await screen.findByRole("combobox", { name: "Schema" });
    await user.selectOptions(select, "purchase_order");
    await user.selectOptions(select, "identity_card");
    await user.type(screen.getByPlaceholderText(/Paste document text/), "id body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await waitFor(() =>
      expect(api.triggerExtract).toHaveBeenCalledWith(
        expect.objectContaining({ schema_name: "identity_card" }),
      ),
    );
    const call = vi.mocked(api.triggerExtract).mock.calls[0][0];
    expect(call.dynamic_schema_name).toBeUndefined();
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

  it("renders the tool-call trace under the answer when a selected MCP server ran a tool", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    vi.mocked(api.chatCompletionMcpStream).mockResolvedValue({
      choices: [{ message: { role: "assistant", content: "found it" } }],
      docie_agent: {
        tool_calls: [
          {
            tool: "docs-search__search_text",
            status: "ok",
            latency_ms: 42,
            arguments: '{"query":"invoice"}',
            result: "page 1: ...",
          },
        ],
      },
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "search the doc");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("found it")).toBeInTheDocument();
    expect(screen.getByText("docs-search__search_text")).toBeInTheDocument();
    expect(screen.getByText("42ms")).toBeInTheDocument();
    expect(screen.getByText('{"query":"invoice"}')).toBeInTheDocument();
  });

  it("uploads a PDF attachment to docs-search and carries the session id into the next turn", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    vi.mocked(api.uploadSessionDocument).mockResolvedValue({
      session_id: "abc123",
      stored_name: "xyz.pdf",
    });
    vi.mocked(api.chatCompletionMcpStream).mockResolvedValue({
      choices: [{ message: { role: "assistant", content: "read it" } }],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));

    const file = new File(["%PDF-fake"], "invoice.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's in this?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(api.uploadSessionDocument).toHaveBeenCalledTimes(1));
    // First upload of the conversation: no session id yet, so undefined —
    // the backend mints a fresh one and returns it.
    expect(api.uploadSessionDocument).toHaveBeenCalledWith("ZmFrZQ==", "invoice.pdf", undefined);
    await waitFor(() => expect(api.chatCompletionMcpStream).toHaveBeenCalledTimes(1));
    expect(api.chatCompletionMcpStream).toHaveBeenCalledWith(
      "lfm2.5-350m",
      expect.any(Array),
      ["docs-search"],
      expect.any(Function),
      "abc123",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );

    // A second turn with no new attachment still carries the SAME session id
    // forward -- docs-search keeps seeing the file from turn 1.
    await user.type(screen.getByPlaceholderText(/Type a message/), "anything else?");
    await user.click(screen.getByRole("button", { name: "Send" }));
    await waitFor(() => expect(api.chatCompletionMcpStream).toHaveBeenCalledTimes(2));
    expect(api.chatCompletionMcpStream).toHaveBeenLastCalledWith(
      "lfm2.5-350m",
      expect.any(Array),
      ["docs-search"],
      expect.any(Function),
      "abc123",
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
  });

  it("does not upload an image attachment to docs-search (not a suffix it accepts)", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    vi.mocked(api.chatCompletionMcpStream).mockResolvedValue({
      choices: [{ message: { role: "assistant", content: "described it" } }],
    });
    // Vision on (this deployment is vision-capable) -- otherwise the new
    // vision-off guard refuses to send an image attachment at all, and this
    // test wants to reach chatCompletionMcpStream to prove docs-search skips it.
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));

    const file = new File(["fake-bytes"], "photo.png", { type: "image/png" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    await user.type(screen.getByPlaceholderText(/Type a message/), "describe this");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(api.chatCompletionMcpStream).toHaveBeenCalledTimes(1));
    expect(api.uploadSessionDocument).not.toHaveBeenCalled();
  });

  it("refuses to send a PDF attachment on a non-vision model with docs-search unselected", async () => {
    // The real bug this guards against: sending image content to a model
    // with no mmproj 500s upstream. Vision defaults off for a non-vision
    // deployment, and with no docs-search selected there's no other way
    // for the model to see this file -- refuse up front instead.
    renderChat();
    const user = userEvent.setup();
    const file = new File(["%PDF-fake"], "invoice.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    await user.type(screen.getByPlaceholderText(/Type a message/), "who signed this?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/Vision is off/)).toBeInTheDocument();
    expect(api.chatCompletionStream).not.toHaveBeenCalled();
    expect(api.chatCompletionMcpStream).not.toHaveBeenCalled();
  });

  it("renders a tool call the moment it streams in, before the final answer arrives", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (_model, _messages, _servers, onToolCall) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onToolCall({
            tool: "docs-search__list_files",
            status: "ok",
            latency_ms: 19,
            arguments: "{}",
            result: "invoice.pdf",
          });
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "who signed this?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    // The tool call is visible WHILE the request is still in flight -- the
    // whole point of streaming instead of "Waiting for the model…" with no
    // visibility into the agentic search running underneath it.
    expect(await screen.findByText("docs-search__list_files")).toBeInTheDocument();
    expect(screen.queryByText("who found it")).not.toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "who found it" } }] });
    expect(await screen.findByText("who found it")).toBeInTheDocument();
    expect(screen.getByText("docs-search__list_files")).toBeInTheDocument();
  });

  it("renders reasoning_content the moment it streams in, before the tool call", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (_model, _messages, _servers, onToolCall, _sessionId, onReasoning) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onReasoning?.("the user wants the TTC, so I should search the CVEC document");
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(
      await screen.findByText("the user wants the TTC, so I should search the CVEC document"),
    ).toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "no info" } }] });
    expect(await screen.findByText("no info")).toBeInTheDocument();
    expect(
      screen.getByText("the user wants the TTC, so I should search the CVEC document"),
    ).toBeInTheDocument();
  });

  it("interleaves reasoning and tool calls in the order they streamed", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (_model, _messages, _servers, onToolCall, _sessionId, onReasoning) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onReasoning?.("first, list the files");
          onToolCall({
            tool: "docs-search__list_files",
            status: "ok",
            latency_ms: 5,
            arguments: "{}",
            result: "[]",
          });
          onReasoning?.("now search for the total");
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const traceLabel = await screen.findByText("Trace");
    const traceContainer = traceLabel.closest("div");
    if (!traceContainer) throw new Error("trace container not found");
    const texts = within(traceContainer)
      .getAllByRole("listitem")
      .map((el) => el.textContent ?? "");
    const firstReasoning = texts.findIndex((t) => t.includes("first, list the files"));
    const toolCall = texts.findIndex((t) => t.includes("docs-search__list_files"));
    const secondReasoning = texts.findIndex((t) => t.includes("now search for the total"));
    expect(firstReasoning).toBeGreaterThanOrEqual(0);
    expect(toolCall).toBeGreaterThan(firstReasoning);
    expect(secondReasoning).toBeGreaterThan(toolCall);

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "42 EUR" } }] });
    expect(await screen.findByText("42 EUR")).toBeInTheDocument();
  });

  it("renders the system-prompt addendum collapsed by default, and expands it on click", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    const addendum =
      "Tool discipline: when a tool takes an identifier (a path, id, or name) " +
      "that another tool lists, call the listing tool first and use one of its " +
      "results EXACTLY as returned.";
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (_model, _messages, _servers, _onToolCall, _sessionId, _onReasoning, onSystemAddendum) => {
        onSystemAddendum?.(addendum);
        return Promise.resolve({
          choices: [{ message: { role: "assistant", content: "found it" } }],
        });
      },
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "search the doc");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("found it")).toBeInTheDocument();
    const summary = screen.getByText("System-prompt addendum");
    const details = summary.closest("details");
    if (!details) throw new Error("details element not found");

    // Collapsed by default: the addendum text exists in the DOM (a native
    // <details> keeps its content mounted) but the element is closed.
    expect(details).not.toHaveAttribute("open");
    expect(screen.getByText(addendum, { exact: false })).toBeInTheDocument();

    await user.click(summary);
    expect(details).toHaveAttribute("open");
  });

  it("renders per-round token usage inline in the trace", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (_model, _messages, _servers, _onToolCall, _sessionId, _onReasoning, _onSystemAddendum, onUsage) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onUsage?.({
            round: { prompt_tokens: 120, completion_tokens: 8, total_tokens: 128 },
            cumulative: { prompt_tokens: 120, completion_tokens: 8, total_tokens: 128 },
          });
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/120/)).toBeInTheDocument();
    expect(screen.getByText(/128/)).toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "42 EUR" } }] });
    expect(await screen.findByText("42 EUR")).toBeInTheDocument();
  });

  it("renders a warning banner when cumulative usage crosses the context budget", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (
        _model,
        _messages,
        _servers,
        _onToolCall,
        _sessionId,
        _onReasoning,
        _onSystemAddendum,
        _onUsage,
        onContextBudget,
      ) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onContextBudget?.({
            cumulative_tokens: 3300,
            context_length: 4096,
            threshold_fraction: 0.8,
          });
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "keep going");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/Context budget warning/)).toBeInTheDocument();
    expect(screen.getByText(/3300/)).toBeInTheDocument();
    expect(screen.getByText(/4096/)).toBeInTheDocument();
    expect(screen.getByText(/80%/)).toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "still going" } }] });
    expect(await screen.findByText("still going")).toBeInTheDocument();
    // The warning is a standing risk for the rest of the exchange -- it
    // must still be visible once the final answer lands, not just during
    // the live/streaming phase.
    expect(screen.getByText(/Context budget warning/)).toBeInTheDocument();
  });

  it("does not render a context budget warning when no context_budget event fires", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    vi.mocked(api.chatCompletionMcpStream).mockResolvedValue({
      choices: [{ message: { role: "assistant", content: "all fine" } }],
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "quick question");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("all fine")).toBeInTheDocument();
    expect(screen.queryByText(/Context budget warning/)).not.toBeInTheDocument();
  });
});
