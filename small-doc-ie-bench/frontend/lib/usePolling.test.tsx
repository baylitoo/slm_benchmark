import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import { SWRConfig } from "swr";
import { usePolling } from "./usePolling";

// Each test gets its own SWR cache so the useId()-derived keys used by
// separate usePolling instances can never collide across test cases.
function wrapper({ children }: { children: ReactNode }) {
  return (
    <SWRConfig value={{ provider: () => new Map() }}>{children}</SWRConfig>
  );
}

describe("usePolling", () => {
  it("enabled=false disables polling: fetches once then never again on its own", async () => {
    const fn = vi.fn().mockResolvedValue("first");
    const { result } = renderHook(({ enabled }) => usePolling(fn, 10, enabled), {
      initialProps: { enabled: false },
      wrapper,
    });

    // No enabled=true render yet -> no automatic fetch at all.
    await new Promise((r) => setTimeout(r, 50));
    expect(fn).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
    expect(result.current.live).toBe(false);
  });

  it("refresh() forces an immediate refetch even while disabled", async () => {
    const fn = vi.fn().mockResolvedValueOnce("a").mockResolvedValueOnce("b");
    const { result } = renderHook(() => usePolling(fn, 10_000, false), { wrapper });

    expect(fn).not.toHaveBeenCalled();

    await act(async () => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.data).toBe("a"));
    expect(fn).toHaveBeenCalledTimes(1);

    await act(async () => {
      result.current.refresh();
    });

    await waitFor(() => expect(result.current.data).toBe("b"));
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
