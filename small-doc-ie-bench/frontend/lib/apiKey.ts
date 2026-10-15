"use client";

// The credential the browser sends as X-API-Key, in two flavours.
//
// 1. A STUDIO SESSION token, minted by POST /v1/auth/session once the user
//    has proved a company sign-in (lib/auth.ts). Short-lived and tied to that
//    sign-in, so it lives in sessionStorage next to the company session and
//    dies with the tab — a signed-in user is never asked for a key.
// 2. An OPERATOR key, typed into the top-bar dialog. The fallback for a
//    deployment with no company auth configured (AUTH_BASE unset), and the way
//    an operator uses a key from API_KEYS. A config value for THIS browser
//    rather than a session, so localStorage is right for it.
//
// A session token wins when both exist: it is scoped to the signed-in user, so
// it gives honest per-user attribution in the usage ledger where a shared
// operator key cannot. Both are verified by the same backend check
// (docie_bench.security.TenantQuotaManager.authenticate).

import { useEffect, useState } from "react";
import { mutate } from "swr";

const STORAGE_KEY = "docie:api-key";
// Session-scoped on purpose, mirroring lib/auth.ts's reasoning for the company
// tokens: this is a credential, not a setting, and closing the tab should end it.
const SESSION_TOKEN_KEY = "docie:studio-token";
// Absolute epoch ms when the Studio token stops being accepted. Stored because
// the Studio token can expire BEFORE the company token (its TTL is capped at
// the company access-token lifetime, so a longer-lived company session is
// truncated): renewal has to be scheduled on whichever deadline comes first,
// or the Studio would sit unauthenticated until the company deadline.
const SESSION_EXPIRY_KEY = "docie:studio-token-exp";
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

/** The Studio session token from the current company sign-in, if any. */
export function getStudioToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const v = window.sessionStorage.getItem(SESSION_TOKEN_KEY);
    return v && v.trim() ? v : null;
  } catch {
    return null;
  }
}

/** When the stored Studio token expires, or null when there is none. */
export function getStudioTokenExpiry(): number | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.sessionStorage.getItem(SESSION_EXPIRY_KEY);
    if (!raw) return null;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

/** Store (or clear, when null) the Studio session token and its deadline. */
export function setStudioToken(token: string | null, expiresInSeconds?: number): void {
  if (typeof window === "undefined") return;
  try {
    if (token) {
      window.sessionStorage.setItem(SESSION_TOKEN_KEY, token);
      if (Number.isFinite(expiresInSeconds) && (expiresInSeconds ?? 0) > 0) {
        window.sessionStorage.setItem(
          SESSION_EXPIRY_KEY,
          String(Date.now() + (expiresInSeconds as number) * 1000),
        );
      }
    } else {
      window.sessionStorage.removeItem(SESSION_TOKEN_KEY);
      window.sessionStorage.removeItem(SESSION_EXPIRY_KEY);
    }
  } catch {
    // Storage unavailable: fall back to the operator key, same as before.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
  // Panels that 401'd before sign-in must refetch now that there is a
  // credential; same reasoning as setApiKey below.
  void mutate(() => true, undefined, { revalidate: true });
}

/** `{ "X-API-Key": ... }` when a credential is available, else `{}`.
 *
 * Session token first: it identifies the signed-in user, while an operator key
 * identifies only the deployment. */
export function authHeader(): Record<string, string> {
  const credential = getStudioToken() ?? getApiKey();
  return credential ? { "X-API-Key": credential } : {};
}

/** Reactive "do we have a credential at all" flag — a Studio session counts,
 * so a signed-in user is never told a key is missing. */
export function useHasApiKey(): boolean {
  const [has, setHas] = useState(() => getStudioToken() != null || getApiKey() != null);
  useEffect(() => {
    const sync = () => setHas(getStudioToken() != null || getApiKey() != null);
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
