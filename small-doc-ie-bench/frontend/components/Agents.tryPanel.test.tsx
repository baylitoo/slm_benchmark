import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { TryPanel } from "@/components/Agents";
import * as api from "@/lib/api";
import type { AgentChatResponse, AgentView } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return { ...actual, agentChat: vi.fn() };
});

const AGENT: AgentView = {
  name: "tool-helper",
  kind: "custom",
  options: {},
};

describe("Agents TryPanel — tool-call trace (#262)", () => {
  it("renders each executed tool call with its status, latency, arguments and result", async () => {
    const response: AgentChatResponse = {
      choices: [{ message: { role: "assistant", content: "the sum is 5" } }],
      docie_agent: {
        agent: "tool-helper",
        kind: "custom",
        tool_calls: [
          {
            tool: "calc__add",
            status: "ok",
            latency_ms: 12,
            arguments: '{"a": 2, "b": 3}',
            result: "5",
          },
        ],
      },
    };
    vi.mocked(api.agentChat).mockResolvedValue(response);

    render(<TryPanel agent={AGENT} />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("textbox"), "what is 2+3?");
    await user.click(screen.getByRole("button", { name: /run/i }));

    expect(await screen.findByText("calc__add")).toBeInTheDocument();
    expect(screen.getByText("ok")).toBeInTheDocument();
    expect(screen.getByText("12ms")).toBeInTheDocument();
    expect(screen.getByText('{"a": 2, "b": 3}')).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("the sum is 5")).toBeInTheDocument();
  });

  it("shows no tool-call section when the agent didn't run any tools", async () => {
    vi.mocked(api.agentChat).mockResolvedValue({
      choices: [{ message: { role: "assistant", content: "hi there" } }],
      docie_agent: { agent: "tool-helper", kind: "custom" },
    });

    render(<TryPanel agent={AGENT} />);
    const user = userEvent.setup();
    await user.type(screen.getByRole("textbox"), "hello");
    await user.click(screen.getByRole("button", { name: /run/i }));

    await screen.findByText("hi there");
    expect(screen.queryByText("Tool calls")).not.toBeInTheDocument();
  });
});
