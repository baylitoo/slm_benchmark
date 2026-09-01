import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { McpView } from "@/components/deploy/McpView";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { DeploymentRecord, McpCatalogEntry } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listMcpCatalog: vi.fn(),
    enableMcpServer: vi.fn(),
    disableMcpServer: vi.fn(),
    testMcpServer: vi.fn(),
    getCodeInterpreterWorkers: vi.fn(),
    getDeployments: vi.fn(async () => []),
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

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
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
          {
            name: "allowed_hosts",
            description: "Comma-separated hostnames",
            required: false,
            secret: false,
            kind: "text",
            choices: [],
          },
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

  it("renders a number-kind param as a numeric input", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({
        name: "docs-search",
        title: "Document Search",
        tools: ["list_files", "read_document", "search_text"],
        params: [
          {
            name: "snippet_window",
            description: "Characters of context",
            required: false,
            secret: false,
            kind: "number",
            choices: [],
          },
        ],
      }),
    ]);
    renderView();
    const input = await screen.findByLabelText("docs-search snippet_window");
    expect(input).toHaveAttribute("type", "number");
  });

  it("renders an enum-kind param as a select of its choices", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({
        name: "docs-search",
        title: "Document Search",
        tools: ["list_files", "read_document", "search_text"],
        params: [
          {
            name: "backend",
            description: "Retrieval strategy",
            required: false,
            secret: false,
            kind: "enum",
            choices: ["substring", "hybrid"],
          },
        ],
      }),
    ]);
    vi.mocked(api.enableMcpServer).mockResolvedValue({ name: "docs-search", registered: true });
    renderView();
    const user = userEvent.setup();
    const select = await screen.findByLabelText("docs-search backend");
    expect(select.tagName).toBe("SELECT");
    await user.selectOptions(select, "hybrid");
    await user.click(screen.getByRole("button", { name: /enable/i }));
    await waitFor(() =>
      expect(api.enableMcpServer).toHaveBeenCalledWith("docs-search", { backend: "hybrid" }),
    );
  });

  it("renders a model_profile-kind param as a select of live chat deployments", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([
      makeDeployment("lfm2.5-350m"),
      makeDeployment("nuextract3"),
    ]);
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({
        name: "call-llm",
        title: "Call LLM (sub-agent dispatch)",
        tools: ["call_llm"],
        params: [
          {
            name: "default_model_profile",
            description: "model_profile every call_llm call dispatches to",
            required: false,
            secret: false,
            kind: "model_profile",
            choices: [],
          },
        ],
      }),
    ]);
    vi.mocked(api.enableMcpServer).mockResolvedValue({ name: "call-llm", registered: true });
    renderView();
    const user = userEvent.setup();
    const select = await screen.findByLabelText("call-llm default_model_profile");
    expect(select.tagName).toBe("SELECT");
    await user.selectOptions(select, "nuextract3");
    await user.click(screen.getByRole("button", { name: /enable/i }));
    await waitFor(() =>
      expect(api.enableMcpServer).toHaveBeenCalledWith("call-llm", {
        default_model_profile: "nuextract3",
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

  it("shows the sandbox worker pool for an enabled code-interpreter entry", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({
        name: "code-interpreter",
        title: "Code Interpreter",
        tools: ["run_python"],
        enabled: true,
      }),
    ]);
    vi.mocked(api.getCodeInterpreterWorkers).mockResolvedValue({
      queues: [
        { queue: "default", size: 1, available: 2, idle: 1, working: 1, paused: 0, failed: 0 },
      ],
    });
    renderView();
    await waitFor(() => expect(api.getCodeInterpreterWorkers).toHaveBeenCalled());
    expect(await screen.findByText("2 workers")).toBeInTheDocument();
    expect(screen.getByText(/1 idle · 1 working · 1 queued/)).toBeInTheDocument();
  });

  it("does not fetch the worker pool for a non-code-interpreter entry", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([makeEntry({ enabled: true })]);
    renderView();
    await screen.findByText("Calculator");
    expect(api.getCodeInterpreterWorkers).not.toHaveBeenCalled();
  });

  it("shows an error when the sandbox is unreachable", async () => {
    vi.mocked(api.listMcpCatalog).mockResolvedValue([
      makeEntry({ name: "code-interpreter", title: "Code Interpreter", enabled: true }),
    ]);
    vi.mocked(api.getCodeInterpreterWorkers).mockRejectedValue(new Error("could not reach judge0"));
    renderView();
    expect(await screen.findByText(/could not reach judge0/)).toBeInTheDocument();
  });
});
