import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { chatCompletion, chatCompletionStream, downloadArtifact, getModels } from "./api";
import { setApiKey } from "./apiKey";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

describe("request() auth header", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends X-API-Key when a key is stored", async () => {
    setApiKey("sk-abc123");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, []));
    vi.stubGlobal("fetch", fetchMock);

    await getModels();

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-abc123");
  });

  it("omits X-API-Key when no key is stored", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, []));
    vi.stubGlobal("fetch", fetchMock);

    await getModels();

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).not.toHaveProperty("X-API-Key");
  });

  it("surfaces an actionable message pointing at the top-bar key icon on 401", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(jsonResponse(401, { detail: "A valid API key is required" }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getModels()).rejects.toMatchObject({
      status: 401,
      message: expect.stringContaining("key icon in the top bar"),
    });
  });

  // request() is one of FOUR fetch call sites in this file (chatCompletion/
  // chatCompletionStream go through openaiPost/their own fetch, downloadArtifact
  // has its own raw fetch -- none go through request()) -- each covered
  // separately so a future refactor of any of them can't silently drop the
  // header without a test noticing.
  it("chatCompletion (openaiPost) sends X-API-Key when a key is stored", async () => {
    setApiKey("sk-abc123");
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse(200, { id: "x", choices: [] }));
    vi.stubGlobal("fetch", fetchMock);

    await chatCompletion("some-model", [{ role: "user", content: "hi" }]);

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-abc123");
  });

  it("chatCompletionStream sends X-API-Key when a key is stored", async () => {
    setApiKey("sk-abc123");
    // No body -> the function returns right after the header/status checks,
    // without needing to exercise the SSE frame-decoding loop.
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await chatCompletionStream("some-model", [{ role: "user", content: "hi" }], () => {});

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-abc123");
  });

  it("downloadArtifact sends X-API-Key when a key is stored", async () => {
    setApiKey("sk-abc123");
    const fetchMock = vi
      .fn()
      .mockResolvedValue(new Response(new Blob(["report"]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    // jsdom doesn't implement the Blob URL API -- stub just enough for
    // downloadArtifact's create/revoke pair to run without throwing.
    const revokeObjectURL = vi.fn();
    vi.stubGlobal(
      "URL",
      Object.assign(URL, { createObjectURL: vi.fn(() => "blob:mock"), revokeObjectURL }),
    );

    await downloadArtifact("/v1/studio/artifacts/abc", "report.html");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>)["X-API-Key"]).toBe("sk-abc123");
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:mock");
  });
});
