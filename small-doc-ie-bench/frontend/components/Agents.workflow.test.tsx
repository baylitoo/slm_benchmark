import { render, screen } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { CreateView, TryPanel } from "@/components/Agents";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { AgentChatResponse, AgentTemplate, AgentView, DeploymentRecord } from "@/lib/api";

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
    testMcpServer: vi.fn(),
    createAgent: vi.fn(),
    agentChat: vi.fn(),
  };
});

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
  };
}

const WORKFLOW_TEMPLATE: AgentTemplate = {
  id: "workflow-agent",
  kind: "workflow",
  display_name: "Workflow Agent",
  description: "Prompt chaining.",
  defaults: { system_prompt: null, options: { steps: [] } },
};
const CUSTOM_TEMPLATE: AgentTemplate = {
  id: "custom",
  kind: "custom",
  display_name: "Custom Agent",
  description: "Bring your own.",
  defaults: { system_prompt: "", options: {} },
};

function renderCreate() {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>
        <CreateView
          templates={[CUSTOM_TEMPLATE, WORKFLOW_TEMPLATE]}
          prefill={null}
          editAgent={null}
          onCreated={vi.fn()}
        />
      </ToastProvider>
    </SWRConfig>,
  );
}

describe("Agents CreateView — workflow-kind Steps section (#265)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.getDeployments).mockResolvedValue([
      makeDeployment("lfm2.5-350m"),
      makeDeployment("qwen-mini"),
    ]);
    vi.mocked(api.createAgent).mockResolvedValue({
      name: "chain",
      kind: "workflow",
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

  it("hides the backing-model/system-prompt fields and shows the step editor", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.selectOptions(
      await screen.findByLabelText("Template", { exact: false }),
      "Workflow Agent",
    );
    expect(screen.queryByLabelText("Backing model")).not.toBeInTheDocument();
    expect(await screen.findByText("Step 1")).toBeInTheDocument();
  });

  it("submits an ordered list of steps, dropping any with no model chosen", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.selectOptions(await screen.findByLabelText("Template", { exact: false }), "Workflow Agent");
    await user.type(screen.getByPlaceholderText("pii-proxy"), "chain");

    await user.selectOptions(screen.getByLabelText("Step 1 model"), "lfm2.5-350m");
    await user.type(screen.getByLabelText("Step 1 system prompt"), "extract facts");

    await user.click(screen.getByRole("button", { name: /add step/i }));
    await user.selectOptions(screen.getByLabelText("Step 2 model"), "qwen-mini");
    await user.type(screen.getByLabelText("Step 2 system prompt"), "summarize");

    await user.click(screen.getByRole("button", { name: /create agent/i }));

    const payload = vi.mocked(api.createAgent).mock.calls[0][0];
    expect(payload.model_profile).toBeNull();
    expect(payload.options).toEqual({
      steps: [
        { model_profile: "lfm2.5-350m", system_prompt: "extract facts" },
        { model_profile: "qwen-mini", system_prompt: "summarize" },
      ],
    });
  });

  it("can remove a step", async () => {
    renderCreate();
    const user = userEvent.setup();
    await user.selectOptions(await screen.findByLabelText("Template", { exact: false }), "Workflow Agent");
    await user.click(screen.getByRole("button", { name: /add step/i }));
    expect(screen.getByText("Step 2")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /remove step 2/i }));
    expect(screen.queryByText("Step 2")).not.toBeInTheDocument();
  });
});

describe("Agents TryPanel — workflow step trace (#265)", () => {
  const WORKFLOW_AGENT: AgentView = { name: "chain", kind: "workflow", options: {} };

  it("renders each step's model and content in order", async () => {
    const response: AgentChatResponse = {
      choices: [{ message: { role: "assistant", content: "final summary" } }],
      docie_agent: {
        agent: "chain",
        kind: "workflow",
        steps: [
          { step: 0, model_profile: "lfm2.5-350m", content: "- fact one\n- fact two" },
          { step: 1, model_profile: "qwen-mini", content: "final summary" },
        ],
      },
    };
    vi.mocked(api.agentChat).mockResolvedValue(response);

    render(<TryPanel agent={WORKFLOW_AGENT} />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("textbox"), "summarize this");
    await user.click(screen.getByRole("button", { name: /run/i }));

    expect((await screen.findAllByText("lfm2.5-350m")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText("qwen-mini").length).toBeGreaterThanOrEqual(1);
    // Each string also appears in the "Raw completion JSON" <details> dump,
    // so use getAllByText (not getByText) throughout this assertion block.
    expect(screen.getAllByText(/fact one/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/fact two/).length).toBeGreaterThanOrEqual(1);
    // "final summary" appears twice: once as step 2's own content, once as
    // the completion's overall Response text -- both are the same answer.
    expect(screen.getAllByText("final summary").length).toBeGreaterThanOrEqual(2);
  });

  it("shows a route step's jump target (#266)", async () => {
    const response: AgentChatResponse = {
      choices: [{ message: { role: "assistant", content: "handled: billing" } }],
      docie_agent: {
        agent: "chain",
        kind: "workflow",
        steps: [
          {
            step: 0,
            name: "0",
            model_profile: "classifier",
            content: "Label: billing",
            routed_to: "handle-billing",
          },
          {
            step: 1,
            name: "handle-billing",
            model_profile: "billing-handler",
            content: "handled: billing",
            routed_to: null,
          },
        ],
      },
    };
    vi.mocked(api.agentChat).mockResolvedValue(response);

    render(<TryPanel agent={WORKFLOW_AGENT} />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("textbox"), "why was I charged twice?");
    await user.click(screen.getByRole("button", { name: /run/i }));

    expect(await screen.findAllByText("handle-billing")).not.toHaveLength(0);
  });

  it("tags a workflow's tool calls with the step that made them", async () => {
    const response: AgentChatResponse = {
      choices: [{ message: { role: "assistant", content: "5" } }],
      docie_agent: {
        agent: "chain",
        kind: "workflow",
        tool_calls: [
          {
            tool: "calc__add",
            status: "ok",
            latency_ms: 12,
            step: 0,
            step_name: "first",
          },
        ],
      },
    };
    vi.mocked(api.agentChat).mockResolvedValue(response);

    render(<TryPanel agent={WORKFLOW_AGENT} />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("textbox"), "add some numbers");
    await user.click(screen.getByRole("button", { name: /run/i }));

    expect(await screen.findAllByText(/step: first/)).not.toHaveLength(0);
  });
});
