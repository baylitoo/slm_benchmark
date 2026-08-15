import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getModels } from "./api";
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
});
