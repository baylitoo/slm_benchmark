import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { authHeader, getApiKey, setApiKey, useHasApiKey } from "./apiKey";

describe("apiKey", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns null and an empty header when unset", () => {
    expect(getApiKey()).toBeNull();
    expect(authHeader()).toEqual({});
  });

  it("persists a trimmed key and exposes it as an X-API-Key header", () => {
    setApiKey("  sk-abc123  ");
    expect(getApiKey()).toBe("sk-abc123");
    expect(authHeader()).toEqual({ "X-API-Key": "sk-abc123" });
  });

  it("clears the key on null or blank input", () => {
    setApiKey("sk-abc123");
    setApiKey(null);
    expect(getApiKey()).toBeNull();

    setApiKey("sk-xyz");
    setApiKey("   ");
    expect(getApiKey()).toBeNull();
  });

  it("survives a raw localStorage read (no JSON wrapping)", () => {
    setApiKey("sk-abc123");
    expect(window.localStorage.getItem("docie:api-key")).toBe("sk-abc123");
  });
});

describe("useHasApiKey", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("reflects the current stored state and updates on same-tab writes", () => {
    const { result } = renderHook(() => useHasApiKey());
    expect(result.current).toBe(false);

    act(() => setApiKey("sk-abc123"));
    expect(result.current).toBe(true);

    act(() => setApiKey(null));
    expect(result.current).toBe(false);
  });
});
