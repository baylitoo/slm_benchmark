import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { ArenaPanel } from "@/components/Playground";
import * as api from "@/lib/api";
import { ApiError } from "@/lib/api";
import type { DeploymentRecord } from "@/lib/api";
import type { PollingState } from "@/lib/usePolling";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    chatCompletionStream: vi.fn(),
  };
});

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
  };
}

const RECORDS = [makeDeployment("lfm2.5-350m"), makeDeployment("nuextract3")];

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

function renderArena(records: DeploymentRecord[] = RECORDS) {
  return render(
    <ArenaPanel deployments={makePolling(records)} selectable={records} />,
  );
}

/** Type a prompt and press Send, then wait for both panels to settle. */
async function sendPrompt(user: ReturnType<typeof userEvent.setup>, text: string) {
  await user.type(
    screen.getByPlaceholderText(/Type a message for both models/),
    text,
  );
  await user.click(screen.getByRole("button", { name: "Send" }));
}

describe("ArenaPanel", () => {
  beforeAll(() => {
    // jsdom has no scrollIntoView; the panel auto-scrolls to the latest turn.
    Element.prototype.scrollIntoView = vi.fn();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    // Each side answers with a model-tagged string, streamed as one token.
    vi.mocked(api.chatCompletionStream).mockImplementation(
      async (model, _messages, onToken) => {
        onToken(`answer from ${model}`);
      },
    );
  });

  it("preselects two different deployments, left and right", () => {
    renderArena();
    const selects = screen.getAllByRole("combobox");
    expect(selects).toHaveLength(2);
    expect(selects[0]).toHaveValue("lfm2.5-350m");
    expect(selects[1]).toHaveValue("nuextract3");
  });

  it("sends the same message list to both deployments and streams both answers", async () => {
    renderArena();
    const user = userEvent.setup();
    await sendPrompt(user, "hello");

    expect(await screen.findByText("answer from lfm2.5-350m")).toBeInTheDocument();
    expect(await screen.findByText("answer from nuextract3")).toBeInTheDocument();

    expect(api.chatCompletionStream).toHaveBeenCalledTimes(2);
    const calls = vi.mocked(api.chatCompletionStream).mock.calls;
    expect(calls[0][0]).toBe("lfm2.5-350m");
    expect(calls[1][0]).toBe("nuextract3");
    // Identical payloads: one prompt, both sides.
    expect(calls[0][1]).toEqual([{ role: "user", content: "hello" }]);
    expect(calls[1][1]).toEqual(calls[0][1]);
  });

  it("shows a per-side elapsed time once a turn ran", async () => {
    renderArena();
    const user = userEvent.setup();
    await sendPrompt(user, "hello");
    await screen.findByText("answer from nuextract3");
    // One "N.N s" readout per side.
    expect(screen.getAllByText(/^\d+\.\d s$/).length).toBe(2);
  });

  it("prepends the system prompt to both sides", async () => {
    renderArena();
    const user = userEvent.setup();
    await user.type(screen.getByPlaceholderText("You are…"), "Be terse.");
    await sendPrompt(user, "hello");
    await screen.findByText("answer from nuextract3");

    const calls = vi.mocked(api.chatCompletionStream).mock.calls;
    expect(calls[0][1]).toEqual([
      { role: "system", content: "Be terse." },
      { role: "user", content: "hello" },
    ]);
    expect(calls[1][1]).toEqual(calls[0][1]);
  });

  it("feeds the LEFT side's answer as assistant history by default", async () => {
    renderArena();
    const user = userEvent.setup();
    await sendPrompt(user, "first");
    await screen.findByText("answer from nuextract3");
    await sendPrompt(user, "second");
    await waitFor(() =>
      expect(api.chatCompletionStream).toHaveBeenCalledTimes(4),
    );

    const history = [
      { role: "user", content: "first" },
      { role: "assistant", content: "answer from lfm2.5-350m" },
      { role: "user", content: "second" },
    ];
    const calls = vi.mocked(api.chatCompletionStream).mock.calls;
    expect(calls[2][1]).toEqual(history);
    expect(calls[3][1]).toEqual(history);
  });

  it("feeds the RIGHT side's answer when the turn's Continue-from control is switched", async () => {
    renderArena();
    const user = userEvent.setup();
    await sendPrompt(user, "first");
    await screen.findByText("answer from nuextract3");

    await user.click(screen.getByRole("button", { name: "Right" }));
    await sendPrompt(user, "second");
    await waitFor(() =>
      expect(api.chatCompletionStream).toHaveBeenCalledTimes(4),
    );

    const calls = vi.mocked(api.chatCompletionStream).mock.calls;
    expect(calls[2][1]).toEqual([
      { role: "user", content: "first" },
      { role: "assistant", content: "answer from nuextract3" },
      { role: "user", content: "second" },
    ]);
  });

  it("keeps one side's answer when the other side fails", async () => {
    vi.mocked(api.chatCompletionStream).mockImplementation(
      async (model, _messages, onToken) => {
        if (model === "nuextract3") throw new ApiError(500, "right side broke");
        onToken(`answer from ${model}`);
      },
    );
    renderArena();
    const user = userEvent.setup();
    await sendPrompt(user, "hello");

    expect(await screen.findByText("answer from lfm2.5-350m")).toBeInTheDocument();
    expect(await screen.findByText("right side broke")).toBeInTheDocument();

    // The failed side is skipped in history: the next turn falls back to the
    // surviving (left) answer — which is also the default side here.
    await sendPrompt(user, "again");
    await waitFor(() =>
      expect(api.chatCompletionStream).toHaveBeenCalledTimes(4),
    );
    const calls = vi.mocked(api.chatCompletionStream).mock.calls;
    expect(calls[2][1]).toEqual([
      { role: "user", content: "hello" },
      { role: "assistant", content: "answer from lfm2.5-350m" },
      { role: "user", content: "again" },
    ]);
  });

  it("shows the empty-model state when nothing is deployed", () => {
    renderArena([]);
    expect(screen.getAllByText("No chat model deployed yet.").length).toBe(2);
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  });
});
