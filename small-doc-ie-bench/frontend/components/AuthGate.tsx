"use client";
// Gate between the sign-in screen and the Studio.
//
// Deliberately a UI gate, not a security boundary: the DocIE backend is
// protected by its own X-API-Key check (AUTH_REQUIRED + API_KEYS), which this
// does not touch. What it buys is that opening the Studio URL no longer shows
// the console to anyone who finds it — a company sign-in comes first.
//
// AUTH_BASE unset => no auth server configured => render the app as before, so
// a local `next dev` or a single-operator deployment keeps working untouched.

import { AUTH_BASE } from "@/lib/env";
import { useSession } from "@/lib/auth";
import { LoginScreen } from "./LoginScreen";

export function AuthGate({ children }: { children: React.ReactNode }) {
  const { session, ready } = useSession();

  if (!AUTH_BASE) return <>{children}</>;
  // Storage is only readable after mount; rendering either branch before that
  // would flash the wrong one on every load.
  if (!ready) return null;
  if (!session) return <LoginScreen />;
  return <>{children}</>;
}
