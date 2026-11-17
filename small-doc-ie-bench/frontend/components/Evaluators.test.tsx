import { render, screen, waitFor } from "@testing-library/react";
import { SWRConfig } from "swr";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { Evaluators, fromDraft, toDraft } from "@/components/Evaluators";
import { Review } from "@/components/Review";
import { ToastProvider } from "@/components/Toast";
import * as api from "@/lib/api";
import type { EvaluatorSpec, ReviewTaskView } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    listEvaluators: vi.fn(),
    getEvaluatorTemplates: vi.fn(),
    getAgents: vi.fn(),
    getDeployments: vi.fn(),
    getStore: vi.fn(),
    createEvaluator: vi.fn(),
    listEvaluationResults: vi.fn(),
    listReviews: vi.fn(),
    getAgentRun: vi.fn(),
    approveReview: vi.fn(),
  };
});

const TEMPLATES = {
  categories: [
    { id: "start", title: "Start here" },
    { id: "scratch", title: "Create from scratch" },
    { id: "security", title: "Security", summary: "Injections and data leaks." },
  ],
  templates: [
    {
      id: "task-completion",
      title: "Task completion",
      category: "start",
      recommended: true,
      summary: "Whether the agent fully completed what the user asked.",
      judge: "decision" as const,
      questions: [{ id: "completed", type: "noul" as const, instructions: "Done?" }],
      review_when: [{ question: "completed", when: "noul_below" as const, value: 0.5 }],
    },
    {
      id: "llm-judge-scratch",
      title: "LLM-as-a-judge evaluator",
      category: "scratch",
      summary: "Ask a chat model.",
      judge: "llm_judge" as const,
      questions: [],
      review_when: [],
    },
    {
      id: "prompt-injection",
      title: "Prompt injection",
      category: "security",
      summary: "Whether a user message tries to override the agent's instructions.",
      judge: "decision" as const,
      questions: [{ id: "injection", type: "noul" as const, instructions: "Override?" }],
      review_when: [],
    },
  ],
};

function renderPage(node: React.ReactNode) {
  return render(
    <SWRConfig value={{ provider: () => new Map() }}>
      <ToastProvider>{node}</ToastProvider>
    </SWRConfig>,
  );
}

describe("question drafts", () => {
  it("round-trip choice options with descriptions and score levels", () => {
    const choice = { id: "intent", type: "choice" as const, instructions: "Why?", criteria: { invoice: "a bill", other: null } };
    expect(fromDraft(toDraft(choice))).toEqual(choice);
    const score = { id: "tone", type: "score" as const, instructions: "Tone?", criteria: ["bad", "ok", "good"] };
    expect(fromDraft(toDraft(score))).toEqual(score);
  });
});

describe("Evaluators", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.listEvaluators).mockResolvedValue([]);
    vi.mocked(api.getEvaluatorTemplates).mockResolvedValue(TEMPLATES);
    vi.mocked(api.getAgents).mockResolvedValue([
      { name: "billing-bot", kind: "custom" },
      { name: "pii-proxy", kind: "proxy_security" },
    ] as never);
    vi.mocked(api.getDeployments).mockResolvedValue([
      { spec: { name: "laya-q8_0", launch: { runtime: "llama_cpp" } }, state: "ready", endpoint: "http://x" },
    ] as never);
    vi.mocked(api.getStore).mockResolvedValue([
      { name: "laya-q8_0", family: "decision", decision: true },
      { name: "lfm2.5-350m", family: "lfm2" },
    ]);
    vi.mocked(api.createEvaluator).mockImplementation(async (spec) => spec);
  });

  it("shows the template gallery grouped like the reference layout when there is none", async () => {
    renderPage(<Evaluators />);
    expect(await screen.findByText("Get started with Evaluators")).toBeInTheDocument();
    expect(screen.getByText("Start here")).toBeInTheDocument();
    expect(screen.getByText("Create from scratch")).toBeInTheDocument();
    expect(screen.getByText("Security")).toBeInTheDocument();
    await userEvent.setup().type(screen.getByLabelText("Search templates"), "injection");
    expect(screen.queryByText("Task completion")).not.toBeInTheDocument();
    expect(screen.getByText("Prompt injection")).toBeInTheDocument();
  });

  it("creates an evaluator from a template, never offering a proxy_security agent", async () => {
    const user = userEvent.setup();
    renderPage(<Evaluators />);
    await user.click(await screen.findByText("Task completion"));

    expect(screen.queryByRole("button", { name: "pii-proxy" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "billing-bot" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "" }), "laya-q8_0");
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.createEvaluator).toHaveBeenCalledTimes(1));
    const spec = vi.mocked(api.createEvaluator).mock.calls[0][0] as EvaluatorSpec;
    expect(spec).toMatchObject({
      name: "task-completion",
      judge: "decision",
      model: "laya-q8_0",
      agents: ["billing-bot"],
      sample_rate: 1,
      questions: [{ id: "completed", type: "noul", instructions: "Done?" }],
      review_when: [{ question: "completed", when: "noul_below", value: 0.5 }],
    });
  });

  it("offers a decision judge only decision models, by model first, then per node", async () => {
    vi.mocked(api.getDeployments).mockResolvedValue([
      {
        spec: { name: "laya-q8_0-f09ae4", launch: { runtime: "llama_cpp", alias: "laya-q8_0" } },
        replica_group: "laya-q8_0",
        node_id: "rocky-llm",
        state: "ready",
        endpoint: "http://rocky-llm:8090/v1",
      },
      { spec: { name: "lfm2.5-350m", launch: { runtime: "llama_cpp" } }, state: "ready", endpoint: "http://x" },
    ] as never);
    const user = userEvent.setup();
    renderPage(<Evaluators />);
    await user.click(await screen.findByText("Task completion"));

    const picker = screen.getByRole("combobox", { name: "" });
    await waitFor(() =>
      expect([...picker.querySelectorAll("option")].map((o) => o.textContent)).toEqual([
        "Select a deployment…",
        "laya-q8_0",
        "laya-q8_0-f09ae4 · on rocky-llm",
      ]),
    );
  });
});

function agentTask(): ReviewTaskView {
  return {
    id: 7,
    kind: "agent_run",
    source_request_id: "run-1",
    schema_name: "billing-bot",
    model_profile: "task-completion",
    document_hash: null,
    status: "pending",
    priority: 1,
    priority_reasons: [{ code: "evaluator", score: 1, detail: "task-completion · completed: 0.10 < 0.5" }],
    original_prediction: {
      run_id: "run-1",
      agent: "billing-bot",
      verdicts: [
        {
          evaluator: "task-completion",
          question_id: "completed",
          type: "noul",
          provenance: "measured",
          noul: 0.1,
        },
      ],
    },
    latest_prediction: {},
    validation_errors: [],
    validation_warnings: [],
    dynamic_schema: null,
    metadata: {},
    claimed_by: null,
    claim_expires_at: null,
    version: 1,
    created_at: "2026-11-15T10:00:00Z",
    updated_at: "2026-11-15T10:00:00Z",
    decided_at: null,
    decided_by: null,
    decision_comment: null,
    corrections: [],
    events: [],
    suggested_corrections: [],
    evidence_available: false,
  } as ReviewTaskView;
}

describe("Review · Agent runs", () => {
  beforeEach(() => vi.clearAllMocks());

  it("lists agent-run tasks and shows the run beside its verdicts", async () => {
    vi.mocked(api.listReviews).mockResolvedValue([agentTask()]);
    vi.mocked(api.getAgentRun).mockResolvedValue({
      id: "run-1",
      agent: "billing-bot",
      tenant_id: "acme",
      messages: [
        { role: "system", content: "secret instructions" },
        { role: "user", content: "Refund F-12 please" },
      ],
      output: "I cannot help with that.",
      tool_calls: [],
      evaluators: ["task-completion"],
      created_at: "2026-11-15T10:00:00Z",
      expires_at: "2026-12-15T10:00:00Z",
    });
    renderPage(<Review view="agent-runs" />);

    await waitFor(() =>
      expect(api.listReviews).toHaveBeenCalledWith({ status: "pending", kind: "agent_run" }),
    );
    await userEvent.setup().click(await screen.findByText("run-1"));
    expect(await screen.findByText("Refund F-12 please")).toBeInTheDocument();
    expect(screen.getByText("I cannot help with that.")).toBeInTheDocument();
    expect(screen.queryByText("secret instructions")).not.toBeInTheDocument();
    expect(screen.getByText("yes 10%")).toBeInTheDocument();
    expect(screen.getByText("model probability")).toBeInTheDocument();
  });
});
