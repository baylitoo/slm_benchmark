import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SWRConfig } from "swr";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "@/components/Playground";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord, StoreEntry } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

// Multi-attach (up to 10 files per turn): each attached file gets its own
// extraction run (N separate extractStream calls, N live result cards) --
// NOT one call merging every file's content. Chat/vision keeps the existing
// single-message convention: every attached image (or PDF page) rides the
// SAME chat message as additional image_url parts, same as a multi-page PDF
// already does today.
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

function pngFile(name: string): File {
  return new File(["fake-bytes"], name, { type: "image/png" });
}

describe("ChatPanel multi-attach", () => {
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

  it("attaching multiple files renders one thumbnail per file", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const files = [pngFile("a.png"), pngFile("b.png"), pngFile("c.png")];
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input).toHaveAttribute("multiple");
    await user.upload(input, files);

    expect(await screen.findByText("a.png")).toBeInTheDocument();
    expect(screen.getByText("b.png")).toBeInTheDocument();
    expect(screen.getByText("c.png")).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /Enlarge attachment preview/ }),
    ).toHaveLength(3);
  });

  it("caps a selection over 10 files at the first 10 and shows an inline message", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const files = Array.from({ length: 12 }, (_, i) => pngFile(`f${i}.png`));
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, files);

    expect(await screen.findByText("f0.png")).toBeInTheDocument();
    expect(screen.getByText("f9.png")).toBeInTheDocument();
    expect(screen.queryByText("f10.png")).not.toBeInTheDocument();
    expect(screen.queryByText("f11.png")).not.toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /Enlarge attachment preview/ }),
    ).toHaveLength(10);
    expect(screen.getByText(/Only 10 more files? could be added/)).toBeInTheDocument();
  });

  it("a second, separate pick ADDS to what's already attached, not replaces it", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    await user.upload(input, pngFile("first.png"));
    expect(await screen.findByText("first.png")).toBeInTheDocument();

    await user.upload(input, pngFile("second.png"));
    expect(await screen.findByText("second.png")).toBeInTheDocument();
    // The first pick must still be there -- this is the exact regression:
    // a second pick used to wipe the attachment array instead of adding to it.
    expect(screen.getByText("first.png")).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /Enlarge attachment preview/ }),
    ).toHaveLength(2);
  });

  it("a second pick beyond the remaining room is capped against the TOTAL, not the new pick alone", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;

    await user.upload(input, Array.from({ length: 8 }, (_, i) => pngFile(`a${i}.png`)));
    expect(await screen.findByText("a0.png")).toBeInTheDocument();

    await user.upload(input, Array.from({ length: 5 }, (_, i) => pngFile(`b${i}.png`)));
    expect(await screen.findByText("b0.png")).toBeInTheDocument();
    // 8 already attached + room for only 2 more = 10 total.
    expect(screen.getByText("b1.png")).toBeInTheDocument();
    expect(screen.queryByText("b2.png")).not.toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /Enlarge attachment preview/ }),
    ).toHaveLength(10);
    expect(screen.getByText(/Only 2 more files could be added/)).toBeInTheDocument();
  });

  it("removes a single attachment without disturbing the others", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const files = [pngFile("a.png"), pngFile("b.png")];
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, files);
    expect(await screen.findByText("a.png")).toBeInTheDocument();

    const removeButtons = screen.getAllByRole("button", { name: "Remove attachment" });
    await user.click(removeButtons[0]);

    await waitFor(() => expect(screen.queryByText("a.png")).not.toBeInTheDocument());
    expect(screen.getByText("b.png")).toBeInTheDocument();
    expect(
      screen.getAllByRole("button", { name: /Enlarge attachment preview/ }),
    ).toHaveLength(1);
  });

  it("fires one extraction call per attached file and renders one result card per file", async () => {
    vi.mocked(api.extractStream).mockImplementation(async (payload) => ({
      invoice_number: { value: payload.filename },
    }));
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const files = [pngFile("invoice1.png"), pngFile("invoice2.png")];
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, files);
    await user.click(await screen.findByRole("button", { name: "Run extraction" }));

    await waitFor(() => expect(api.extractStream).toHaveBeenCalledTimes(2));
    const filenames = vi
      .mocked(api.extractStream)
      .mock.calls.map((c) => c[0].filename)
      .sort();
    expect(filenames).toEqual(["invoice1.png", "invoice2.png"]);

    expect(await screen.findByText(/"invoice1\.png"/)).toBeInTheDocument();
    expect(await screen.findByText(/"invoice2\.png"/)).toBeInTheDocument();

    // Attachments are cleared once the run starts, same as the single-file
    // path.
    expect(screen.queryByText("invoice1.png")).not.toBeInTheDocument();
  });

  it("shows the specific API error, not a generic count, when a single attached file's extraction fails (N=1 regression guard)", async () => {
    vi.mocked(api.extractStream).mockRejectedValue(new Error("schema not found"));
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, pngFile("solo.png"));
    await user.click(await screen.findByRole("button", { name: "Run extraction" }));

    // The specific API error message (not a generic "N of M failed" count)
    // lands in the error banner AND the per-file live card, same as today's
    // single-attachment path.
    const alerts = await screen.findAllByRole("alert");
    expect(alerts.length).toBeGreaterThan(0);
    for (const alert of alerts) expect(alert).toHaveTextContent("schema not found");
    expect(screen.queryByText(/extraction requests failed/)).not.toBeInTheDocument();
  });

  it("routes a single attached file through extraction exactly as before (N=1 regression guard)", async () => {
    vi.mocked(api.extractStream).mockResolvedValue({ invoice_number: { value: "solo-result" } });
    renderChat();
    const user = userEvent.setup();
    await user.click(screen.getByRole("checkbox"));
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, pngFile("solo.png"));
    await user.click(await screen.findByRole("button", { name: "Run extraction" }));

    await waitFor(() => expect(api.extractStream).toHaveBeenCalledTimes(1));
    expect(api.extractStream).toHaveBeenCalledWith(
      expect.objectContaining({ schema_name: "invoice", deployment: "lfm2.5-350m", filename: "solo.png" }),
      expect.any(Function),
      expect.any(Function),
      expect.any(Function),
    );
    expect(await screen.findByText(/"solo-result"/)).toBeInTheDocument();
  });

  it("sends multiple attached images as one chat message with one image_url per file", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const files = [pngFile("front.png"), pngFile("back.png")];
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, files);
    await user.type(screen.getByPlaceholderText(/Type a message/), "compare these");
    await user.click(screen.getByRole("button", { name: "Send" }));

    await waitFor(() => expect(api.chatCompletionStream).toHaveBeenCalledTimes(1));
    const [, messages] = vi.mocked(api.chatCompletionStream).mock.calls[0];
    const last = messages.at(-1) as { role: string; content: unknown };
    const parts = last.content as { type: string; image_url?: { url: string } }[];
    const imageParts = parts.filter((p) => p.type === "image_url");
    expect(imageParts).toHaveLength(2);
  });

  it("shows a multi-file display label in the conversation when no text was typed", async () => {
    renderChat(RECORDS, [{ name: "lfm2.5-350m", vision: true }]);
    const user = userEvent.setup();
    const files = [pngFile("front.png"), pngFile("back.png"), pngFile("side.png")];
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, files);
    await user.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText("📎 3 files")).toBeInTheDocument();
  });
});
