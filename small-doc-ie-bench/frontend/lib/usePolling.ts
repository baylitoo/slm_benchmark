"use client";

import { useCallback, useEffect, useId, useRef, useState } from "react";
import useSWR from "swr";

export interface PollingState<T> {
  data: T | null;
  error: unknown;
  /** True only on the very first load (no data yet). */
  loading: boolean;
  /** True while a background refresh is in flight (data already present). */
  refreshing: boolean;
  /** Epoch ms of the last successful fetch, or null. */
  lastUpdated: number | null;
  /** Whether the interval is currently ticking (visible + enabled). */
  live: boolean;
  refresh: () => void;
}

/**
 * Poll `fn` every `intervalMs`, backed by SWR instead of a hand-rolled
 * `setInterval`/`visibilitychange` state machine (see `useAsync`'s docstring
 * for the same lesson applied to one-shot fetches).
 *
 * Unlike `useAsync`, callers here pass a plain function rather than a cache
 * key and don't want cross-component cache sharing -- each call site is its
 * own independent polling loop -- so the SWR key is a stable per-mount id
 * from `useId()` rather than a shared string.
 *
 * - `refreshInterval: enabled ? intervalMs : 0` reproduces the enabled-gate
 *   (SWR treats `0` as "don't poll").
 * - SWR's default `refreshWhenHidden: false` pauses polling on a hidden tab,
 *   same as the old `document.visibilitychange` listener.
 * - `enabled` flipping false -> true (e.g. the caller's section becoming
 *   active again, which does NOT touch tab visibility) forces one immediate
 *   revalidation via `mutate()`, matching the old behavior of fetching right
 *   away on re-activation instead of waiting out the rest of the interval.
 * - `lastUpdated` has no SWR equivalent: tracked locally, stamped in the
 *   `onSuccess` config callback.
 * - `refresh()` is SWR's `mutate()`, which -- because the key is always the
 *   same stable id regardless of `enabled` -- fires immediately even while
 *   disabled, same as the old unconditional `run()`. It's wrapped in
 *   `useCallback` to keep a stable identity across renders: at least one
 *   caller (`Observability.tsx`'s `UsageCard`) depends on `refresh` in a
 *   `useEffect` array to force a refetch when its window filter changes,
 *   and a fresh function identity every render would re-fire that effect
 *   on every render instead of only on real changes.
 */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs = 4000,
  enabled = true,
): PollingState<T> {
  const key = useId();
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);

  // Keep the latest fn without retriggering a re-subscribe.
  const fnRef = useRef(fn);
  fnRef.current = fn;

  const { data, error, isLoading, isValidating, mutate } = useSWR<T>(
    key,
    () => fnRef.current(),
    {
      refreshInterval: enabled ? intervalMs : 0,
      revalidateOnMount: enabled,
      onSuccess: () => setLastUpdated(Date.now()),
    },
  );

  // Force an immediate refetch when re-activated, rather than waiting for
  // the next interval tick -- `revalidateOnMount` only fires once, at the
  // very first render.
  const wasEnabled = useRef(enabled);
  useEffect(() => {
    if (enabled && !wasEnabled.current) void mutate();
    wasEnabled.current = enabled;
  }, [enabled, mutate]);

  const refresh = useCallback(() => {
    void mutate();
  }, [mutate]);

  return {
    data: data ?? null,
    error: error ?? null,
    loading: isLoading,
    refreshing: isValidating && !isLoading,
    lastUpdated,
    live: enabled,
    refresh,
  };
}
