import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DecidePanel, buildDecisionQuestions } from "@/components/Playground";
import * as api from "@/lib/api";
import type { DeploymentRecord, StoreEntry } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, decide: vi.fn() };
});

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
  };
}

const JULIA = makeDeployment("julia-1-q8_0");
const CHAT = makeDeployment("lfm2.5-1.2b");
const STORE: StoreEntry[] = [
  { name: "julia-1-q8_0", family: "decision", decision: true },
  { name: "openjev", family: "decision_vl", decision: true, vision: true },
  { name: "lfm2.5-1.2b", family: "lfm2" },
  { name: "lfm2.5-vl-1.6b", family: "lfm2_vl", vision: true },
];

describe("decision deployments in pickers", () => {
  it("are typed decision and dropped from chat/extraction candidates", () => {
    const names = api.decisionDeploymentNames(STORE);
    expect(api.deploymentModelType(JULIA, undefined, undefined, names)).toBe("decision");
    expect(api.deploymentModelType(CHAT, undefined, undefined, names)).toBe("chat");
    expect(api.withoutDecisionModels([JULIA, CHAT], STORE)).toEqual([CHAT]);
  });

  it("a vision decision model is not offered as a vision extractor", () => {
    expect([...api.visionDeploymentNames(STORE)]).toEqual(["lfm2.5-vl-1.6b"]);
  });
});

describe("buildDecisionQuestions", () => {
  it("parses options, descriptions and levels", () => {
    expect(
      buildDecisionQuestions([
        { id: "route", type: "choice", instructions: "Team?", options: "billing: money\nsales" },
        { id: "mood", type: "score", instructions: "", options: "calm\nangry" },
        { id: "urgent", type: "noul", instructions: "Today?", options: "ignored" },
      ]),
    ).toEqual({
      route: { type: "choice", instructions: "Team?", criteria: { billing: "money", sales: null } },
      mood: { type: "score", instructions: undefined, criteria: ["calm", "angry"] },
      urgent: { type: "noul", instructions: "Today?" },
    });
  });

  it("rejects a choice with one option, a duplicate id and an out-of-range score", () => {
    const one = buildDecisionQuestions([
      { id: "a", type: "choice", instructions: "", options: "only" },
    ]);
    expect(one).toMatch(/at least two options/);
    const dup = buildDecisionQuestions([
      { id: "a", type: "noul", instructions: "", options: "" },
      { id: "a", type: "noul", instructions: "", options: "" },
    ]);
    expect(dup).toMatch(/used twice/);
    const levels = buildDecisionQuestions([
      { id: "s", type: "score", instructions: "", options: "x" },
    ]);
    expect(levels).toMatch(/2 to 10 levels/);
  });
});

describe("DecidePanel", () => {
  beforeEach(() => vi.clearAllMocks());

  it("offers only decision deployments", () => {
    render(<DecidePanel deployments={[JULIA, CHAT]} decisionNames={new Set(["julia-1-q8_0"])} />);
    expect(screen.getByRole("option", { name: "julia-1-q8_0" })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: "lfm2.5-1.2b" })).not.toBeInTheDocument();
  });

  it("sends the example questions and renders each answer's probabilities", async () => {
    vi.mocked(api.decide).mockResolvedValue({
      answers: {
        route: {
          type: "choice",
          choice: "billing",
          probabilities: { billing: 0.91, support: 0.04, legal: 0.05 },
          confidence: 0.86,
        },
        urgent: { type: "noul", noul: 0.73 },
        frustration: {
          type: "score",
          score: 2.6,
          legend: { "0": "calm", "1": "mildly annoyed", "2": "annoyed", "3": "angry" },
          probabilities: { "0": 0.02, "1": 0.08, "2": 0.18, "3": 0.72 },
          confidence: 0.6,
        },
      },
      usage: { input_tokens: 212, output_tokens: 0 },
    });
    render(<DecidePanel deployments={[JULIA]} decisionNames={new Set(["julia-1-q8_0"])} />);
    await userEvent.setup().click(screen.getByRole("button", { name: /^Decide$/ }));

    await waitFor(() => expect(api.decide).toHaveBeenCalledTimes(1));
    const [model, , questions] = vi.mocked(api.decide).mock.calls[0];
    expect(model).toBe("julia-1-q8_0");
    expect(Object.keys(questions)).toEqual(["route", "urgent", "frustration"]);
    expect(questions.route).toMatchObject({ type: "choice" });

    expect(await screen.findByText("yes 73.0%")).toBeInTheDocument();
    expect(screen.getByText("91.0%")).toBeInTheDocument();
    expect(screen.getByText("angry")).toBeInTheDocument();
    expect(screen.getByText(/212 input tokens/)).toBeInTheDocument();
  });
});
