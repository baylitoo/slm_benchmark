"use client";
// Who is signed in, and the way out. Renders nothing when no auth server is
// configured (AUTH_BASE unset), so a local or single-operator deployment keeps
// the top bar it had before.

import { LogOut } from "lucide-react";

import { AUTH_BASE } from "@/lib/env";
import { logout, useSession } from "@/lib/auth";
import { useI18n } from "@/lib/i18n";

export function SessionMenu() {
  const { t } = useI18n();
  const { session } = useSession();

  if (!AUTH_BASE || !session) return null;

  return (
    <div className="flex items-center gap-1.5">
      <span
        className="hidden max-w-[12ch] truncate text-xs text-muted-foreground sm:inline"
        title={session.login}
      >
        {session.login}
      </span>
      <button
        type="button"
        onClick={() => void logout()}
        aria-label={t("Sign out")}
        title={t("Sign out")}
        className="grid h-9 w-9 place-items-center rounded-md border border-border bg-card text-muted-foreground transition hover:bg-muted hover:text-foreground"
      >
        <LogOut className="h-4 w-4" />
      </button>
    </div>
  );
}
