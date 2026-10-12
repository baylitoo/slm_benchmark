"use client";
// Studio sign-in against the company auth server (app.cynov.com,
// `/rest/v2/auth/*`). This gates the STUDIO UI only: the DocIE backend keeps
// its own X-API-Key mechanism untouched (docie_bench.security
// .TenantQuotaManager.authenticate), so the access token is never sent to it
// and the backend needs no key material to verify anything.
//
// Tokens live in sessionStorage, not localStorage: these are credentials to a
// company-wide system, and a 7-day refresh token in localStorage stays within
// reach of any injected script for its whole life. Tab-scoped is the safer
// default and matches the auth server's own stated convention for browser
// tokens. Closing the tab signs out; swap the constant below for
// window.localStorage if the team prefers persistence.

import { useEffect, useState } from "react";

import { AUTH_BASE } from "./env";

const STORAGE_KEY = "docie:session";
// Same-tab writes don't fire the native `storage` event, so mounted components
// (the gate, the top-bar menu) learn about sign-in/out through this instead.
const CHANGE_EVENT = "docie:session-changed";
// Refresh this long before the access token expires, so a slow network or a
// backgrounded tab doesn't leave the app holding a dead token.
const REFRESH_MARGIN_MS = 60_000;

export interface Session {
  accessToken: string;
  refreshToken: string;
  /** Absolute epoch ms when the access token stops being valid. */
  expiresAt: number;
  /** Login the user typed, for display only — never a security claim. */
  login: string;
}

interface TokenPair {
  accessToken: string;
  refreshToken: string;
  tokenType?: string;
  /** Lifetime of the access token in SECONDS (auth server's contract). */
  expiresIn: number;
}

function store(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    // Storage unavailable (private mode, disabled): the app still works, the
    // user just signs in again on every load.
    return null;
  }
}

export function getSession(): Session | null {
  const s = store();
  if (!s) return null;
  try {
    const raw = s.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Session;
    if (!parsed?.accessToken || !parsed?.refreshToken) return null;
    return parsed;
  } catch {
    return null;
  }
}

function setSession(session: Session | null): void {
  const s = store();
  if (!s) return;
  try {
    if (session) s.setItem(STORAGE_KEY, JSON.stringify(session));
    else s.removeItem(STORAGE_KEY);
  } catch {
    // Ignore: a failed write degrades to "signed out", never to a crash.
  }
  window.dispatchEvent(new Event(CHANGE_EVENT));
}

/** The auth server's error bodies are `{ "message": "..." }` on every route. */
async function readError(resp: Response, fallback: string): Promise<string> {
  try {
    const body = (await resp.json()) as { message?: string };
    return body?.message || fallback;
  } catch {
    return fallback;
  }
}

async function post<T>(path: string, body: unknown, headers?: Record<string, string>): Promise<T> {
  let resp: Response;
  try {
    resp = await fetch(`${AUTH_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(headers ?? {}) },
      body: JSON.stringify(body),
    });
  } catch {
    // A network-level failure here is usually CORS (the auth server must allow
    // this Studio's origin) or the server being unreachable — the browser
    // deliberately hides which, so say both.
    throw new Error(
      "Authentication server unreachable. Check the network, and that it allows this origin.",
    );
  }
  if (!resp.ok) throw new Error(await readError(resp, `Authentication failed (HTTP ${resp.status})`));
  return (await resp.json()) as T;
}

function sessionFrom(pair: TokenPair, login: string): Session {
  // expiresIn is seconds; a missing/odd value falls back to the documented
  // 30-minute access-token lifetime rather than to "never expires".
  const seconds = Number.isFinite(pair.expiresIn) && pair.expiresIn > 0 ? pair.expiresIn : 1800;
  return {
    accessToken: pair.accessToken,
    refreshToken: pair.refreshToken,
    expiresAt: Date.now() + seconds * 1000,
    login,
  };
}

/** Sign in with company credentials. Throws with the server's own message. */
export async function login(loginName: string, password: string): Promise<Session> {
  const pair = await post<TokenPair>("/rest/v2/auth/login", {
    login: loginName,
    password,
  });
  const session = sessionFrom(pair, loginName);
  setSession(session);
  return session;
}

/**
 * Exchange the refresh token for a new pair (the server rotates the refresh
 * token, so the stored one is replaced, never reused).
 */
export async function refresh(): Promise<Session | null> {
  const current = getSession();
  if (!current) return null;
  try {
    const pair = await post<TokenPair>("/rest/v2/auth/refresh", {
      refreshToken: current.refreshToken,
    });
    const session = sessionFrom(pair, current.login);
    setSession(session);
    return session;
  } catch {
    // Expired, revoked, or the account was disabled: the only honest outcome
    // is to sign out and show the login screen.
    setSession(null);
    return null;
  }
}

/** Revoke both tokens server-side, then clear them locally whatever happens. */
export async function logout(): Promise<void> {
  const current = getSession();
  setSession(null);
  if (!current) return;
  try {
    await post(
      "/rest/v2/auth/logout",
      { refreshToken: current.refreshToken },
      { Authorization: `Bearer ${current.accessToken}` },
    );
  } catch {
    // Already signed out locally; a failed revoke must not block the UI.
  }
}

/** First sign-in password change (the auth server's own flow). */
export async function changePassword(
  loginName: string,
  oldPassword: string,
  newPassword: string,
): Promise<void> {
  await post("/rest/v2/auth/change-password", {
    login: loginName,
    oldPassword,
    newPassword,
  });
}

/**
 * The live session, with silent refresh.
 *
 * Re-reads on sign-in/out from this tab and from others, and schedules a
 * refresh a minute before expiry. An already-expired token (a laptop that
 * slept past the 30 minutes) refreshes immediately on mount rather than
 * showing the app with a dead credential.
 */
export function useSession(): { session: Session | null; ready: boolean } {
  const [session, setLocal] = useState<Session | null>(null);
  // Server render and first paint must agree: read storage only after mount.
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const sync = () => setLocal(getSession());
    sync();
    setReady(true);
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  useEffect(() => {
    if (!session) return;
    const due = session.expiresAt - Date.now() - REFRESH_MARGIN_MS;
    if (due <= 0) {
      void refresh();
      return;
    }
    // setTimeout clamps above ~24.8 days; the 30-minute access token is far
    // below that, so no chunking is needed.
    const timer = window.setTimeout(() => void refresh(), due);
    return () => window.clearTimeout(timer);
  }, [session]);

  return { session, ready };
}
