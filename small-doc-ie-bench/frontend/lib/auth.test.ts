import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getSession, login, logout, refresh } from "./auth";

function jsonResponse(body: unknown, ok = true, status = 200): Response {
  return {
    ok,
    status,
    json: async () => body,
  } as Response;
}

const PAIR = {
  accessToken: "access-1",
  refreshToken: "refresh-1",
  tokenType: "Bearer",
  expiresIn: 1800,
};

describe("auth session", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-10-08T12:00:00Z"));
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllGlobals();
  });

  it("stores the pair and turns expiresIn seconds into an absolute deadline", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAIR)));

    const session = await login("amine", "secret");

    expect(session.accessToken).toBe("access-1");
    expect(session.login).toBe("amine");
    // 1800s after the frozen clock, not a relative value a later read would
    // mis-measure.
    expect(session.expiresAt).toBe(Date.parse("2026-10-08T12:30:00Z"));
    expect(getSession()?.refreshToken).toBe("refresh-1");
  });

  it("never writes the password anywhere, only the login", async () => {
    const fetchMock = vi.fn(async () => jsonResponse(PAIR));
    vi.stubGlobal("fetch", fetchMock);

    await login("amine", "hunter2");

    expect(window.sessionStorage.getItem("docie:session")).not.toContain("hunter2");
  });

  it("rotates the refresh token the server returns", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAIR)));
    await login("amine", "secret");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({ ...PAIR, accessToken: "access-2", refreshToken: "refresh-2" }),
      ),
    );
    const rotated = await refresh();

    expect(rotated?.accessToken).toBe("access-2");
    // The old refresh token is single-use on the server; keeping it would log
    // the user out on the next renewal.
    expect(getSession()?.refreshToken).toBe("refresh-2");
  });

  it("signs out when the refresh token is rejected", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAIR)));
    await login("amine", "secret");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ message: "expired" }, false, 401)),
    );

    expect(await refresh()).toBeNull();
    expect(getSession()).toBeNull();
  });

  it("clears the session locally even when the revoke call fails", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAIR)));
    await login("amine", "secret");

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("network down");
      }),
    );
    await logout();

    expect(getSession()).toBeNull();
  });

  it("surfaces the server's own message on a failed sign-in", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ message: "Identifiants invalides" }, false, 401)),
    );

    await expect(login("amine", "wrong")).rejects.toThrow("Identifiants invalides");
    expect(getSession()).toBeNull();
  });
});
