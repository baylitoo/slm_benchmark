import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmbedRerankPanel } from "@/components/Playground";
import * as api from "@/lib/api";
import type { DeploymentRecord } from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    embed: vi.fn(),
    rerank: vi.fn(),
  };
});

function makeDeployment(name: string): DeploymentRecord {
  return {
    spec: { name, launch: { runtime: "llama_cpp", model: `${name}.gguf` } },
    state: "ready",
    endpoint: "http://127.0.0.1:8081",
  };
}

const EMBED_MODEL = makeDeployment("lfm2.5-embedding-350m");
const RERANK_MODEL = makeDeployment("lfm2.5-colbert-350m");
const embeddingNames = new Set(["lfm2.5-embedding-350m"]);
const rerankerNames = new Set(["lfm2.5-colbert-350m"]);

function renderPanel() {
  return render(
    <EmbedRerankPanel
      deployments={[EMBED_MODEL, RERANK_MODEL]}
      embeddingNames={embeddingNames}
      rerankerNames={rerankerNames}
    />,
  );
}

describe("EmbedRerankPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("filters the deployment picker per sub-mode — a hard boundary, not just a label", async () => {
    renderPanel();
    // Embed is the default sub-mode: only the embedding model is offered.
    expect(screen.getByRole("option", { name: /lfm2\.5-embedding-350m/ })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /lfm2\.5-colbert-350m/ })).not.toBeInTheDocument();

    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Rerank" }));

    expect(screen.getByRole("option", { name: /lfm2\.5-colbert-350m/ })).toBeInTheDocument();
    expect(
      screen.queryByRole("option", { name: /lfm2\.5-embedding-350m/ }),
    ).not.toBeInTheDocument();
  });

  it("embeds text A and text B as two INDEPENDENT calls, not a shared batch array", async () => {
    // The bug being fixed: a batched [textA, textB] request's response order
    // isn't guaranteed to match the request order on every backend, so
    // pairing by array position silently paired the wrong (or empty) vector.
    // Two separate calls sidestep any batching ambiguity entirely.
    vi.mocked(api.embed).mockImplementation(async (_model, input) => {
      const text = Array.isArray(input) ? input[0] : input;
      return {
        data: [{ index: 0, embedding: text === "hello" ? [1, 0] : [0, 1] }],
      };
    });
    renderPanel();
    const user = userEvent.setup();
    const [textA, textB] = screen.getAllByRole("textbox");
    await user.clear(textA);
    await user.type(textA, "hello");
    await user.clear(textB);
    await user.type(textB, "world");
    await user.click(screen.getByRole("button", { name: /Embed & compare/ }));

    await waitFor(() => expect(api.embed).toHaveBeenCalledTimes(2));
    expect(api.embed).toHaveBeenCalledWith("lfm2.5-embedding-350m", "hello");
    expect(api.embed).toHaveBeenCalledWith("lfm2.5-embedding-350m", "world");
    // Orthogonal vectors -> cosine similarity 0, and BOTH vectors present
    // (previously an empty second vector made every result read as 0 too,
    // indistinguishable from a real orthogonal pair — dims proves it's real).
    expect(await screen.findByText("2 dims")).toBeInTheDocument();
    expect(screen.getByText(/cosine similarity 0\.0000/)).toBeInTheDocument();
  });

  it("reranks documents and sorts them by relevance score, highest first", async () => {
    vi.mocked(api.rerank).mockResolvedValue({
      results: [
        { index: 0, relevance_score: 0.1 },
        { index: 1, relevance_score: 0.9 },
      ],
    });
    renderPanel();
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Rerank" }));
    const [queryBox, docsBox] = screen.getAllByRole("textbox");
    await user.clear(queryBox);
    await user.type(queryBox, "invoice total?");
    await user.clear(docsBox);
    await user.type(docsBox, "weather forecast{enter}invoice total 5400 EUR");
    // "Rerank" is the accessible name of both the sub-mode toggle AND the
    // submit button; the submit button is the one rendered last.
    await user.click(screen.getAllByRole("button", { name: "Rerank" }).at(-1)!);

    await waitFor(() => expect(api.rerank).toHaveBeenCalledTimes(1));
    const order = (await screen.findAllByText(/^0\.\d{4}$/)).map((el) => el.textContent);
    expect(order).toEqual(["0.9000", "0.1000"]);
  });
});
