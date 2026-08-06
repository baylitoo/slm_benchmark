"use client";

import useSWR from "swr";

export interface AsyncState<T> {
  data: T | null;
  error: unknown;
  loading: boolean;
  reload: () => void;
}

/**
 * SWR-backed one-shot fetch with a cache key.
 *
 * The old hand-rolled version fetched once on mount and NEVER revalidated:
 * seed a new model and every family dropdown stayed wrong until a full page
 * reload; a benchmark started in another tab never appeared. Backing the same
 * `{data, error, loading, reload}` shape with SWR gives:
 *
 * - revalidate-on-focus (throttled): stale lists heal themselves the moment
 *   the user comes back to the tab;
 * - request dedup by key: two components asking for the same endpoint at boot
 *   (e.g. the Playground's and Deploy's `getFamilies`) share ONE request and
 *   one cache entry;
 * - stale-while-revalidate: `loading` is true only before the FIRST response,
 *   so refreshes never flicker the UI back to skeletons.
 *
 * `reload()` still forces an immediate refetch (used after mutations).
 */
export function useAsync<T>(key: string, fn: () => Promise<T>): AsyncState<T> {
  const { data, error, isLoading, mutate } = useSWR<T>(key, fn, {
    revalidateOnFocus: true,
    focusThrottleInterval: 15_000,
    dedupingInterval: 2_000,
  });
  return {
    data: data ?? null,
    error: error ?? null,
    loading: isLoading,
    reload: () => void mutate(),
  };
}
