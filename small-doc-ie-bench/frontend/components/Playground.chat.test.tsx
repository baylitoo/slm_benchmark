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
    pauseChatExchange: vi.fn(),
    respondToChatExchange: vi.fn(),
    listMcpServers: vi.fn(),
    listSchemas: vi.fn(),
    listDynamicSchemas: vi.fn(),
    getSchemaFields: vi.fn(),
    listRoutingPolicies: vi.fn(),
    extractStream: vi.fn(),
    fileToBase64: vi.fn(async () => "ZmFrZQ=="),
    renderDocument: vi.fn(async () => ({
      images: ["data:image/png;base64,ZmFrZQ=="],
      pages: 1,
      total_pages: 1,
    })),
    uploadSessionDocument: vi.fn(),
  };
});

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

  it("keeps the DPI selector and presets usable when thumbnail rendering fails", async () => {
    // A failed/slow rasterization (onAttach's own catch -> preview stays
    // null) must never hide the DPI selector or recommended-prompt presets,
    // which don't depend on the thumbnail image at all -- they used to be
    // nested inside the same `{preview && ...}` gate as the <img>, so a
    // render-document failure silently took the whole panel down with it.
    vi.mocked(api.renderDocument).mockRejectedValueOnce(new Error("rasterization failed"));
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["%PDF-fake"], "report.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    expect(await screen.findByText("No preview")).toBeInTheDocument();
    expect(screen.getByText("report.pdf")).toBeInTheDocument();
    expect(screen.getByText(/Render quality/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Describe this image/ })).toBeInTheDocument();
  });

  it("sets a preview thumbnail for a multi-page PDF (regression: used to always fail)", async () => {
    // The bug: onAttach used to call renderDocument with max_pages=1, which
    // rasterized every page THEN rejected any PDF with 2+ pages. Attaching a
    // 5-page PDF must now succeed and show a real thumbnail, not "No preview".
    vi.mocked(api.renderDocument).mockResolvedValueOnce({
      images: ["data:image/png;base64,cGFnZTE="],
      pages: 1,
      total_pages: 5,
    });
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["%PDF-fake"], "multipage.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    const thumb = await screen.findByRole("button", { name: /Enlarge attachment preview/ });
    expect(within(thumb).getByAltText("attachment preview")).toHaveAttribute(
      "src",
      "data:image/png;base64,cGFnZTE=",
    );
    expect(screen.queryByText("No preview")).not.toBeInTheDocument();
    // page 1 only, explicit page list -- no more max_pages misuse.
    expect(api.renderDocument).toHaveBeenCalledWith("ZmFrZQ==", "multipage.pdf", 150, [1]);
  });

  it("opens an enlarge modal on thumbnail click and fetches the rest of a multi-page PDF", async () => {
    vi.mocked(api.renderDocument).mockResolvedValueOnce({
      images: ["data:image/png;base64,cGFnZTE="],
      pages: 1,
      total_pages: 3,
    });
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["%PDF-fake"], "multipage.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    const thumb = await screen.findByRole("button", { name: /Enlarge attachment preview/ });

    vi.mocked(api.renderDocument).mockResolvedValueOnce({
      images: ["data:image/png;base64,cGFnZTE=", "data:image/png;base64,cGFnZTI=", "data:image/png;base64,cGFnZTM="],
      pages: 3,
      total_pages: 3,
    });
    await user.click(thumb);

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    await waitFor(() => expect(api.renderDocument).toHaveBeenCalledTimes(2));
    expect(api.renderDocument).toHaveBeenLastCalledWith("ZmFrZQ==", "multipage.pdf", 200, [1, 2, 3]);
    await waitFor(() =>
      expect(screen.getAllByRole("img").filter((img) => img.getAttribute("alt")?.startsWith("page "))).toHaveLength(3),
    );
  });

  it("does not re-fetch pages opening the enlarge modal for a single-page PDF", async () => {
    vi.mocked(api.renderDocument).mockResolvedValueOnce({
      images: ["data:image/png;base64,cGFnZTE="],
      pages: 1,
      total_pages: 1,
    });
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const file = new File(["%PDF-fake"], "onepage.pdf", { type: "application/pdf" });
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);
    const thumb = await screen.findByRole("button", { name: /Enlarge attachment preview/ });

    await user.click(thumb);

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    // Only the one call made when attaching -- no redundant enlarge fetch.
    expect(api.renderDocument).toHaveBeenCalledTimes(1);
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

  it("routes Send to extraction when the toggle is on, streams deltas live, and swaps to the JSON result", async () => {
    let resolveStream!: (value: unknown) => void;
    vi.mocked(api.extractStream).mockImplementation(
      (_payload, onDelta) =>
        new Promise((resolve) => {
          resolveStream = resolve;
          onDelta?.('{"invoice_number":');
          onDelta?.(' "INV-1"}');
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.type(screen.getByPlaceholderText(/Paste document text/), "invoice body text");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    // The raw delta buffer is visible WHILE the request is still in flight --
    // accumulated live, before the terminal result event ever lands.
    expect(await screen.findByText('{"invoice_number": "INV-1"}')).toBeInTheDocument();
    expect(api.extractStream).toHaveBeenCalledWith(
      expect.objectContaining({
        schema_name: "invoice",
        deployment: "lfm2.5-350m",
        text: "invoice body text",
      }),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    // Chat's streaming path must NOT have been used for this turn.
    expect(api.chatCompletionStream).not.toHaveBeenCalled();

    resolveStream({ invoice_number: { value: "INV-1" } });
    // The raw delta text is replaced by the authoritative JsonView render of
    // the terminal result -- never left showing the accumulated deltas.
    await waitFor(() =>
      expect(screen.queryByText('{"invoice_number": "INV-1"}')).not.toBeInTheDocument(),
    );
    expect(await screen.findByText(/"INV-1"/)).toBeInTheDocument();
  });

  it("clears the live buffer on a reset event before painting the retry's own deltas", async () => {
    vi.mocked(api.extractStream).mockImplementation(async (_payload, onDelta, onReset) => {
      onDelta?.("garbage-from-abandoned-attempt");
      onReset?.();
      onDelta?.('{"vendor_name": "Acme"}');
      return { vendor_name: { value: "Acme" } };
    });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.type(screen.getByPlaceholderText(/Paste document text/), "body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await screen.findByText(/"Acme"/);
    expect(screen.queryByText(/garbage-from-abandoned-attempt/)).not.toBeInTheDocument();
  });

  it("routes extraction through a routing policy instead of the deployment when selected", async () => {
    vi.mocked(api.listRoutingPolicies).mockResolvedValue([
      { name: "cheap-first", policy: {}, created_at: "", updated_at: "" },
    ]);
    vi.mocked(api.extractStream).mockResolvedValue({ ok: true });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    await user.click(screen.getByRole("button", { name: "Routing policy" }));
    await user.selectOptions(await screen.findByLabelText("Routing policy"), "cheap-first");
    await user.type(screen.getByPlaceholderText(/Paste document text/), "body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await waitFor(() =>
      expect(api.extractStream).toHaveBeenCalledWith(
        expect.objectContaining({ routing_policy: "cheap-first" }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      ),
    );
    const call = vi.mocked(api.extractStream).mock.calls[0][0];
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
    vi.mocked(api.extractStream).mockResolvedValue({ ok: true });
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
      expect(api.extractStream).toHaveBeenCalledWith(
        expect.objectContaining({ dynamic_schema_name: "purchase_order" }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
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
    vi.mocked(api.extractStream).mockResolvedValue({ ok: true });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const select = await screen.findByRole("combobox", { name: "Schema" });
    await user.selectOptions(select, "purchase_order");
    await user.selectOptions(select, "identity_card");
    await user.type(screen.getByPlaceholderText(/Paste document text/), "id body");
    await user.click(screen.getByRole("button", { name: "Run extraction" }));

    await waitFor(() =>
      expect(api.extractStream).toHaveBeenCalledWith(
        expect.objectContaining({ schema_name: "identity_card" }),
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      ),
    );
    const call = vi.mocked(api.extractStream).mock.calls[0][0];
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
      expect.any(Function),
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
      expect.any(Function),
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

  it("renders a distinct banner when the deployment's chat template does not support tool-calling", async () => {
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
        _onContextBudget,
        onToolCallsUnsupported,
      ) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onToolCallsUnsupported?.({
            message:
              "This deployment's chat template does not support real tool-calling " +
              "(llama-server reports chat_template_caps.supports_tool_calls=false) — " +
              "the model may describe using tools instead of actually calling them.",
          });
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "read the pdf");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/Tool-calling not supported/)).toBeInTheDocument();
    expect(screen.queryByText(/Context budget warning/)).not.toBeInTheDocument();

    resolveCompletion({
      choices: [{ message: { role: "assistant", content: "I don't have a tool for that" } }],
    });
    expect(await screen.findByText("I don't have a tool for that")).toBeInTheDocument();
    // Standing warning for the rest of the exchange, same as contextBudgetWarning.
    expect(screen.getByText(/Tool-calling not supported/)).toBeInTheDocument();
  });

  it("does not render the tool-calls-unsupported banner when the event never fires", async () => {
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
    expect(screen.queryByText(/Tool-calling not supported/)).not.toBeInTheDocument();
  });

  it("regenerates the last assistant message by replaying the same request and replacing it", async () => {
    vi.mocked(api.chatCompletionStream)
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("first answer"))
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("second answer"));
    renderChat();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/Type a message/), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("first answer")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Regenerate" }));

    expect(await screen.findByText("second answer")).toBeInTheDocument();
    // Replaced, not duplicated: the old answer is gone and only one
    // assistant bubble remains.
    expect(screen.queryByText("first answer")).not.toBeInTheDocument();
    expect(api.chatCompletionStream).toHaveBeenCalledTimes(2);
    const [firstModel, firstMessages] = vi.mocked(api.chatCompletionStream).mock.calls[0];
    const [secondModel, secondMessages] = vi.mocked(api.chatCompletionStream).mock.calls[1];
    expect(secondModel).toBe(firstModel);
    expect(secondMessages).toEqual(firstMessages);
  });

  it("edits an earlier user message, truncating everything after it and resending the edit", async () => {
    vi.mocked(api.chatCompletionStream)
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("answer one"))
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("answer two"))
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("answer three"));
    renderChat();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/Type a message/), "first message");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer one")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText(/Type a message/), "second message");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer two")).toBeInTheDocument();

    // Edit the FIRST user turn -- everything after it (the second turn) must
    // be dropped once the edit is resent.
    const editButtons = screen.getAllByRole("button", { name: "Edit" });
    await user.click(editButtons[0]);
    const input = screen.getByPlaceholderText(/Edit your message/);
    expect(input).toHaveValue("first message");
    await user.clear(input);
    await user.type(input, "first message, edited");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("answer three")).toBeInTheDocument();
    expect(screen.getByText("first message, edited")).toBeInTheDocument();
    expect(screen.queryByText("first message")).not.toBeInTheDocument();
    expect(screen.queryByText("second message")).not.toBeInTheDocument();
    expect(screen.queryByText("answer two")).not.toBeInTheDocument();

    expect(api.chatCompletionStream).toHaveBeenCalledTimes(3);
    const [, thirdMessages] = vi.mocked(api.chatCompletionStream).mock.calls[2];
    // Only the edited message is in the payload -- the truncated second turn
    // never rides along as history.
    expect(thirdMessages).toEqual([{ role: "user", content: "first message, edited" }]);
  });

  it("only offers Regenerate on the last assistant message, and Edit on every user message", async () => {
    vi.mocked(api.chatCompletionStream)
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("answer one"))
      .mockImplementationOnce(async (_model, _messages, onToken) => onToken("answer two"));
    renderChat();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/Type a message/), "first message");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer one")).toBeInTheDocument();

    await user.type(screen.getByPlaceholderText(/Type a message/), "second message");
    await user.click(screen.getByRole("button", { name: "Send" }));
    expect(await screen.findByText("answer two")).toBeInTheDocument();

    // Exactly one Regenerate control, on the last assistant message.
    expect(screen.getAllByRole("button", { name: "Regenerate" })).toHaveLength(1);
    // An Edit control on every user message.
    expect(screen.getAllByRole("button", { name: "Edit" })).toHaveLength(2);
  });

  // ── human-in-the-loop pause/resume (#383) ────────────────────────────────

  it("shows the Pause button only while a tool-calling exchange is streaming", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
        }),
    );
    renderChat();
    const user = userEvent.setup();
    expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();

    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "search the doc");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByRole("button", { name: "Pause" })).toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "done" } }] });
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument(),
    );
  });

  it("never shows Pause for a plain chat exchange with no MCP servers selected", async () => {
    let resolveToken!: () => void;
    vi.mocked(api.chatCompletionStream).mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveToken = () => resolve(undefined);
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText(/Type a message/), "hello");
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(screen.queryByRole("button", { name: "Pause" })).not.toBeInTheDocument();
    resolveToken();
  });

  it("calls pauseChatExchange with the exchange id reported by the stream", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (..._args) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          const onExchangeId = _args[10] as ((id: string) => void) | undefined;
          onExchangeId?.("exch-123");
        }),
    );
    vi.mocked(api.pauseChatExchange).mockResolvedValue(undefined);
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "search the doc");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const pauseButton = await screen.findByRole("button", { name: "Pause" });
    await waitFor(() => expect(pauseButton).not.toBeDisabled());
    await user.click(pauseButton);

    await waitFor(() => expect(api.pauseChatExchange).toHaveBeenCalledWith("exch-123"));

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "done" } }] });
  });

  it("renders a multiple-choice picker for a model-issued ask_user question and answers it", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (..._args) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          const onExchangeId = _args[10] as ((id: string) => void) | undefined;
          const onAwaitingInput = _args[11] as
            | ((payload: { question?: string; choices?: string[] }) => void)
            | undefined;
          onExchangeId?.("exch-ask");
          onAwaitingInput?.({ question: "which invoice?", choices: ["A", "B"] });
        }),
    );
    vi.mocked(api.respondToChatExchange).mockResolvedValue(undefined);
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "search the invoices");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("which invoice?")).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: "A" }));

    await waitFor(() =>
      expect(api.respondToChatExchange).toHaveBeenCalledWith("exch-ask", "A"),
    );
    await waitFor(() => expect(screen.queryByText("which invoice?")).not.toBeInTheDocument());

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "picked A" } }] });
    expect(await screen.findByText("picked A")).toBeInTheDocument();
  });

  it("renders a free-text prompt for a question with no choices and round-trips the typed answer", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (..._args) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          const onExchangeId = _args[10] as ((id: string) => void) | undefined;
          const onAwaitingInput = _args[11] as
            | ((payload: { question?: string; choices?: string[] }) => void)
            | undefined;
          onExchangeId?.("exch-pause");
          onAwaitingInput?.({});
        }),
    );
    vi.mocked(api.respondToChatExchange).mockResolvedValue(undefined);
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "keep going");
    await user.click(screen.getByRole("button", { name: "Send" }));

    const dialog = await screen.findByRole("dialog");
    const answerBox = within(dialog).getByPlaceholderText("Type your answer…");
    await user.type(answerBox, "also check page 2");
    await user.click(within(dialog).getByRole("button", { name: "Send" }));

    await waitFor(() =>
      expect(api.respondToChatExchange).toHaveBeenCalledWith("exch-pause", "also check page 2"),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "done" } }] });
    expect(await screen.findByText("done")).toBeInTheDocument();
  });

  // ── content_delta / reasoning_delta: real token-by-token streaming (#389) ──

  it("renders content_delta fragments live in the assistant bubble as they stream in", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (..._args) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          const onContentDelta = _args[12] as ((text: string) => void) | undefined;
          onContentDelta?.("The ");
          onContentDelta?.("total ");
          onContentDelta?.("is 42 EUR.");
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    // Visible WHILE the request is still in flight -- accumulated live from
    // deltas, before the final content event ever lands.
    expect(await screen.findByText("The total is 42 EUR.")).toBeInTheDocument();

    resolveCompletion({
      choices: [{ message: { role: "assistant", content: "The total is 42 EUR." } }],
    });
    expect(await screen.findByText("The total is 42 EUR.")).toBeInTheDocument();
  });

  it("renders reasoning_delta fragments live in the trace, replaced by that round's final reasoning entry", async () => {
    vi.mocked(api.listMcpServers).mockResolvedValue([
      { name: "docs-search", transport: "stdio", url: null, command: null, headers: null, env: null },
    ]);
    let resolveCompletion!: (value: api.AgentChatResponse) => void;
    let onReasoning: ((text: string) => void) | undefined;
    vi.mocked(api.chatCompletionMcpStream).mockImplementation(
      (..._args) =>
        new Promise((resolve) => {
          resolveCompletion = resolve;
          onReasoning = _args[5] as ((text: string) => void) | undefined;
          const onReasoningDelta = _args[13] as ((text: string) => void) | undefined;
          onReasoningDelta?.("Let ");
          onReasoningDelta?.("me think");
        }),
    );
    renderChat();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: "docs-search" }));
    await user.type(screen.getByPlaceholderText(/Type a message/), "what's the total?");
    await user.click(screen.getByRole("button", { name: "Send" }));

    // The in-progress fragments, accumulated live -- before the round's
    // own one-shot reasoning event has fired at all.
    expect(await screen.findByText("Let me think")).toBeInTheDocument();

    // The round's stream ends: the one-shot reasoning event carries the
    // COMPLETE text, replacing the live in-progress entry rather than
    // sitting alongside it.
    onReasoning?.("Let me think it over.");
    expect(await screen.findByText("Let me think it over.")).toBeInTheDocument();
    expect(screen.queryByText("Let me think")).not.toBeInTheDocument();

    resolveCompletion({ choices: [{ message: { role: "assistant", content: "42 EUR" } }] });
    expect(await screen.findByText("42 EUR")).toBeInTheDocument();
    expect(screen.getByText("Let me think it over.")).toBeInTheDocument();
  });
});
