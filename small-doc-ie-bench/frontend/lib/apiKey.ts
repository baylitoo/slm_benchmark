"use client";

// Client-side operator credential for the backend's X-API-Key auth (see
// docie_bench.security.TenantQuotaManager.authenticate — hmac.compare_digest
// against Settings.API_KEYS, gated by AUTH_REQUIRED). This Studio has no
// login flow (single operator, see api.ts's REVIEWER_ID comment for the same
// pattern elsewhere): the key is a config value persisted in THIS browser
// only, not a session — localStorage is deliberately enough here.

import { useEffect, useState } from "react";
import { mutate } from "swr";

const STORAGE_KEY = "docie:api-key";
// Fired on same-tab writes so other mounted components (e.g. the TopBar
// indicator dot) notice a change; the native `storage` event only fires in
// OTHER tabs, never the one that made the write.
const CHANGE_EVENT = "docie:api-key-changed";

/** The stored API key, or null if unset / not in a browser context. */
export function getApiKey(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const v = window.localStorage.getItem(STORAGE_KEY);
    return v && v.trim() ? v : null;
  } catch {
    // Storage unavailable (private mode, disabled) — degrade to "no key".
    return null;
  }
}

/** Persist (or clear, when null/blank) the operator's API key. */
export function setApiKey(key: string | null): void {
  if (typeof window === "undefined") return;
  try {
    const trimmed = key?.trim();
    if (trimmed) window.localStorage.setItem(STORAGE_KEY, trimmed);
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore write failures (quota / disabled storage) — nothing to persist to.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
  // Without this, every panel's stale 401 error (rendered before the key was
  // entered) sits there unchanged until the next unrelated SWR trigger
  // (revalidate-on-focus, a manual reload) happens to fire -- the operator
  // gets no feedback that saving actually worked. `mutate(() => true, ...)`
  // is SWR's own documented pattern for "revalidate every cache key" (all
  // `useAsync` calls share the default cache -- no SWRConfig in this app).
  void mutate(() => true, undefined, { revalidate: true });
}

/** `{ "X-API-Key": ... }` when a key is stored, else `{}` — spread into fetch headers. */
export function authHeader(): Record<string, string> {
  const key = getApiKey();
  return key ? { "X-API-Key": key } : {};
}

/** Reactive "is a key stored" flag — updates on same-tab and cross-tab changes. */
export function useHasApiKey(): boolean {
  const [has, setHas] = useState(() => getApiKey() != null);
  useEffect(() => {
    const sync = () => setHas(getApiKey() != null);
    sync();
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return has;
}
