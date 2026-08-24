import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { McpView } from "@/components/deploy/McpView";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { McpCatalogEntry } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listMcpCatalog: vi.fn(),
    enableMcpServer: vi.fn(),
    disableMcpServer: vi.fn(),
    testMcpServer: vi.fn(),
  };
});

function makeEntry(overrides: Partial<McpCatalogEntry> = {}): McpCatalogEntry {
  return {
    name: "calculator",
    title: "Calculator",
    description: "Exact arithmetic.",
    tools: ["calc", "sum_check"],
    params: [],
    enabled: false,
    ...overrides,
  };
}

function renderView() {
  // Fresh SWR cache per render -- useAsync keys are module-global otherwise.
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <McpView />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("McpView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("lists catalog entries with tools and enabled state", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry(),
      makeEntry({ name: "web-fetch", title: "Web fetch", tools: ["fetch"], enabled: true }),
    ]);
    renderView();
    expect(await screen.findByText("Calculator")).toBeInTheDocument();
    expect(screen.getByText("Web fetch")).toBeInTheDocument();
    expect(screen.getByText("calc")).toBeInTheDocument();
    expect(screen.getByText("enabled")).toBeInTheDocument();
    expect(screen.getByText("off")).toBeInTheDocument();
  });

  it("enables an entry with its params", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({
        name: "web-fetch",
        title: "Web fetch",
        tools: ["fetch"],
        params: [
          { name: "allowed_hosts", description: "Comma-separated hostnames", required: false },
        ],
      }),
    ]);
    vi.mocked(api.enableMcpServer).mockResolvedValue({ name: "web-fetch", registered: true });
    renderView();
    const user = userEvent.setup();
    await user.type(
      await screen.findByLabelText("web-fetch allowed_hosts"),
      "docs.example.com",
    );
    await user.click(screen.getByRole("button", { name: /enable/i }));
    await waitFor(() =>
      expect(api.enableMcpServer).toHaveBeenCalledWith("web-fetch", {
        allowed_hosts: "docs.example.com",
      }),
    );
  });

  it("tests an enabled entry and shows the live tools", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({ enabled: true, tools: [] }),
    ]);
    vi.mocked(api.testMcpServer).mockResolvedValue({
      name: "calculator",
      ok: true,
      tools: [
        { name: "calc", description: "", input_schema: {} },
        { name: "sum_check", description: "", input_schema: {} },
      ],
    });
    renderView();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /test/i }));
    await waitFor(() => expect(api.testMcpServer).toHaveBeenCalledWith("calculator"));
    expect(await screen.findByText("sum_check")).toBeInTheDocument();
  });

  it("disables an enabled entry", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([makeEntry({ enabled: true })]);
    vi.mocked(api.disableMcpServer).mockResolvedValue({
      name: "calculator",
      registered: false,
    });
    renderView();
    const user = userEvent.setup();
    await user.click(await screen.findByRole("button", { name: /disable/i }));
    await waitFor(() => expect(api.disableMcpServer).toHaveBeenCalledWith("calculator"));
  });
});
