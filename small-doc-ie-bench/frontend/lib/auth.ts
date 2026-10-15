"use client";
// Studio sign-in against the company auth server. This gates the STUDIO UI
// only: the DocIE backend keeps its own X-API-Key mechanism untouched
// (docie_bench.security
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

import { API_BASE, AUTH_BASE } from "./env";
import { getStudioTokenExpiry, setStudioToken } from "./apiKey";

// NEXT_PUBLIC_AUTH_BASE_URL is the auth API's ROOT -- everything up to but not
// including `/auth/...`. On the company deployment that is
// `https://newapp.cynov.com/Facturation-api/rest/v2`: an application prefix
// AND a version segment, neither of which this code may assume.
//
// Only the four route names below are hardcoded, because those are the
// contract this client implements. The host, the mount prefix and the API
// version are all deployment facts and all live in that one variable, so a
// different mount point or a move to another version needs no code change
// here -- and nothing has to be translated: the value is exactly the API root
// an administrator quotes.
//
// Getting it wrong is easy to do and was previously impossible to diagnose:
// the bare origin serves the single-page app, which answers the CORS preflight
// with 405, and a failed preflight makes `fetch` reject with no readable
// response -- indistinguishable from the auth server simply not allowing this
// origin. Hence the explicit 404/405 message in `post` below.
// Exchanging a company session for Studio backend access goes through DocIE's
// POST /v1/auth/session, which verifies the session by calling the auth
// server's own /auth/refresh once and returns BOTH the rotated company pair
// and a Studio token.
//
// That is why renewal is routed through it INSTEAD of calling /auth/refresh
// directly, not in addition to it: the auth server rotates refresh tokens, so
// doing both would double the upstream calls AND invalidate one of the two
// rotated tokens. Steady-state load on the auth server is therefore unchanged
// from before this existed -- one renewal per tab per token lifetime -- plus a
// single exchange at sign-in.
const EXCHANGE_PATH = "/v1/auth/session";

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
  if (!resp.ok) {
    // 404/405 on an auth route almost always means the base URL is pointing at
    // something that is not the auth API -- typically the bare origin, where
    // the single-page app answers instead. Say that, rather than reporting a
    // status code the operator then has to go and interpret.
    const fallback =
      resp.status === 404 || resp.status === 405
        ? `No auth API at ${AUTH_BASE}${path} (HTTP ${resp.status}). ` +
          "NEXT_PUBLIC_AUTH_BASE_URL must be the auth API's root, including " +
          "any application prefix and version segment (e.g. " +
          "https://host/Facturation-api/rest/v2) — not just the origin."
        : `Authentication failed (HTTP ${resp.status})`;
    throw new Error(await readError(resp, fallback));
  }
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
  const pair = await post<TokenPair>("/auth/login", {
    login: loginName,
    password,
  });
  const session = sessionFrom(pair, loginName);
  setSession(session);
  // One exchange here so the Studio has backend access immediately. The
  // password went straight to the company auth server above and never reaches
  // DocIE; only the resulting refresh token is exchanged.
  await exchangeForStudioToken(session);
  return session;
}

/**
 * Trade the company refresh token for a Studio token, and adopt the rotated
 * company pair the exchange returns.
 *
 * Returns the (possibly rotated) session, or null when the company session was
 * rejected. A deployment with the exchange unconfigured answers 503: that is
 * not a failure of the sign-in, so the company session is kept and the Studio
 * falls back to asking for an operator key exactly as it did before.
 */
async function exchangeForStudioToken(current: Session): Promise<Session | null> {
  let resp: Response;
  try {
    resp = await fetch(`${API_BASE}${EXCHANGE_PATH}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refreshToken: current.refreshToken }),
    });
  } catch {
    // DocIE itself unreachable. The company session stands; panels will show
    // their own errors rather than this being reported as a sign-in failure.
    return current;
  }
  if (resp.status === 503) {
    setStudioToken(null);
    return current;
  }
  if (!resp.ok) {
    // The company session was refused upstream: sign out rather than leave a
    // dead session in storage.
    if (resp.status === 401) {
      setStudioToken(null);
      setSession(null);
      return null;
    }
    return current;
  }
  try {
    const body = (await resp.json()) as {
      studioToken?: string;
      expiresIn?: number;
      accessToken?: string;
      refreshToken?: string;
      expiresInCompany?: number;
      login?: string;
    };
    if (body.accessToken && body.refreshToken) {
      // The upstream rotated these; keeping the superseded refresh token would
      // sign the user out at the next renewal.
      const rotated: Session = {
        accessToken: body.accessToken,
        refreshToken: body.refreshToken,
        expiresAt:
          Date.now() +
          (Number.isFinite(body.expiresInCompany) && (body.expiresInCompany ?? 0) > 0
            ? (body.expiresInCompany as number)
            : 1800) *
            1000,
        login: body.login || current.login,
      };
      setSession(rotated);
      setStudioToken(body.studioToken ?? null, body.expiresIn);
      return rotated;
    }
    setStudioToken(body.studioToken ?? null, body.expiresIn);
    return current;
  } catch {
    return current;
  }
}

/**
 * Exchange the refresh token for a new pair (the server rotates the refresh
 * token, so the stored one is replaced, never reused).
 */
export async function refresh(): Promise<Session | null> {
  const current = getSession();
  if (!current) return null;
  // One upstream renewal, not two: the exchange refreshes against the company
  // auth server and hands back both credentials.
  const exchanged = await exchangeForStudioToken(current);
  if (exchanged === null) return null;
  if (exchanged !== current) return exchanged;
  // Exchange unconfigured or DocIE unreachable -- renew directly, as before.
  try {
    const pair = await post<TokenPair>("/auth/refresh", {
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
  setStudioToken(null);
  if (!current) return;
  try {
    await post(
      "/auth/logout",
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
  await post("/auth/change-password", {
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
    // The Studio token's TTL is capped at the company access-token lifetime,
    // so a company session longer than that cap expires LATER than the Studio
    // token. Scheduling on the company deadline alone would leave the Studio
    // unauthenticated in between, so renew on whichever comes first.
    const studioExpiry = getStudioTokenExpiry();
    const deadline =
      studioExpiry !== null ? Math.min(session.expiresAt, studioExpiry) : session.expiresAt;
    const due = deadline - Date.now() - REFRESH_MARGIN_MS;
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
