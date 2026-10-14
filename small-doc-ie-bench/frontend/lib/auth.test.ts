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

describe("request URLs", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  // The four route names are the only paths this client hardcodes. Host, mount
  // prefix and API version all come from NEXT_PUBLIC_AUTH_BASE_URL, so the
  // request must be exactly base + "/auth/<route>" with nothing injected.
  it("appends only the route name to the configured API root", async () => {
    const fetchMock = vi.fn(async (_url: string) => jsonResponse(PAIR));
    vi.stubGlobal("fetch", fetchMock);

    await login("amine", "secret");

    // AUTH_BASE is unset in tests, so the path stands alone -- which is the
    // point: no version segment is added by this module.
    expect(fetchMock.mock.calls[0][0]).toBe("/auth/login");
  });

  it("uses the same root for refresh and logout", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAIR)));
    await login("amine", "secret");

    const fetchMock = vi.fn(async (_url: string) => jsonResponse(PAIR));
    vi.stubGlobal("fetch", fetchMock);
    await refresh();
    expect(fetchMock.mock.calls[0][0]).toBe("/auth/refresh");

    const logoutMock = vi.fn(async (_url: string) => jsonResponse({}));
    vi.stubGlobal("fetch", logoutMock);
    await logout();
    expect(logoutMock.mock.calls[0][0]).toBe("/auth/logout");
  });
});

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

  it("names the misconfiguration on a 404, instead of a bare status code", async () => {
    // Hitting the bare origin reaches the SPA, not the auth API. The operator
    // needs to be told the base URL is wrong, not handed "HTTP 404".
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({}, false, 404)),
    );

    await expect(login("amine", "secret")).rejects.toThrow(/NEXT_PUBLIC_AUTH_BASE_URL/);
  });

  it("still prefers the server's own message when it sends one", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ message: "Route introuvable" }, false, 404)),
    );

    await expect(login("amine", "secret")).rejects.toThrow("Route introuvable");
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
