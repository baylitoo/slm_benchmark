"use client";

import { useCallback, useEffect, useRef, useState } from "react";

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
 * Poll `fn` every `intervalMs`. The interval only runs while:
 *   - `enabled` is true (caller-controlled, e.g. the section is active), AND
 *   - the document is visible (auto-pauses on a hidden/background tab).
 *
 * A manual `refresh()` always fires regardless of those gates. The first load
 * sets `loading`; subsequent ticks set `refreshing` and keep stale data on
 * screen to avoid flicker.
 */
export function usePolling<T>(
  fn: () => Promise<T>,
  intervalMs = 4000,
  enabled = true,
): PollingState<T> {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<number | null>(null);
  const [live, setLive] = useState(false);

  // Keep the latest fn without retriggering the effect each render.
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const hasData = useRef(false);
  const inFlight = useRef(false);

  const run = useCallback(async () => {
    if (inFlight.current) return;
    inFlight.current = true;
    if (hasData.current) setRefreshing(true);
    try {
      const result = await fnRef.current();
      setData(result);
      setError(null);
      setLastUpdated(Date.now());
      hasData.current = true;
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
      setRefreshing(false);
      inFlight.current = false;
    }
  }, []);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;

    const start = () => {
      if (timer != null) return;
      setLive(true);
      void run();
      timer = setInterval(() => void run(), intervalMs);
    };
    const stop = () => {
      setLive(false);
      if (timer != null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const sync = () => {
      const visible =
        typeof document === "undefined" || document.visibilityState === "visible";
      if (enabled && visible) start();
      else stop();
    };

    sync();
    document.addEventListener("visibilitychange", sync);
    return () => {
      document.removeEventListener("visibilitychange", sync);
      stop();
    };
  }, [enabled, intervalMs, run]);

  return { data, error, loading, refreshing, lastUpdated, live, refresh: run };
}
