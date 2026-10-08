"use client";
// Sign-in screen shown in place of the Studio until there is a session.
// Credentials go straight to the company auth server (lib/auth.ts); this app
// never stores or forwards the password.

import { useState } from "react";
import { LogIn } from "lucide-react";

import { changePassword, login } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";
import { Button, Field, TextInput } from "./ui";

export function LoginScreen() {
  const { t } = useI18n();
  const [loginName, setLoginName] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // The auth server has a dedicated first-sign-in password change; surfacing it
  // here means a new account is not a dead end.
  const [changing, setChanging] = useState(false);
  const [newPassword, setNewPassword] = useState("");
  const [notice, setNotice] = useState<string | null>(null);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      if (changing) {
        await changePassword(loginName, password, newPassword);
        setChanging(false);
        setPassword(newPassword);
        setNewPassword("");
        setNotice("Password changed — sign in with the new one.");
      } else {
        // The session lands in storage; the gate above re-renders on its own.
        await login(loginName, password);
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid min-h-screen place-items-center bg-muted/30 px-4">
      <form
        onSubmit={submit}
        className="w-full max-w-sm space-y-4 rounded-lg border border-border bg-card p-6 shadow-sm"
      >
        <div className="space-y-1">
          <h1 className="text-lg font-semibold text-foreground">DocIE Studio</h1>
          <p className="text-xs text-muted-foreground">
            {t(changing ? "Set a new password for this account." : "Sign in with your company account.")}
          </p>
        </div>

        <Field label="Login">
          <TextInput
            value={loginName}
            onChange={(e) => setLoginName(e.target.value)}
            autoComplete="username"
            autoFocus
            required
          />
        </Field>

        <Field label={changing ? "Current password" : "Password"}>
          <TextInput
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete={changing ? "current-password" : "current-password"}
            required
          />
        </Field>

        {changing && (
          <Field label="New password" hint="At least 8 characters.">
            <TextInput
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              autoComplete="new-password"
              minLength={8}
              required
            />
          </Field>
        )}

        {error && (
          <p role="alert" className="rounded-md bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {error}
          </p>
        )}
        {notice && (
          <p className="rounded-md bg-muted px-3 py-2 text-xs text-muted-foreground">{t(notice)}</p>
        )}

        <Button type="submit" loading={busy} className="w-full">
          <LogIn className="h-4 w-4" />
          {t(changing ? "Change password" : "Sign in")}
        </Button>

        <button
          type="button"
          onClick={() => {
            setChanging((v) => !v);
            setError(null);
            setNotice(null);
          }}
          className="w-full text-center text-xs text-muted-foreground underline decoration-dotted underline-offset-2 hover:text-foreground"
        >
          {t(changing ? "Back to sign in" : "First sign-in? Change your password")}
        </button>
      </form>
    </main>
  );
}
