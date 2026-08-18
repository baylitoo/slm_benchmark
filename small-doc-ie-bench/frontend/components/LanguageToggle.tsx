"use client";

import { Languages } from "lucide-react";
import { useI18n } from "@/lib/i18n";

export function LanguageToggle() {
  const { locale, toggleLocale, t } = useI18n();
  const nextLabel = locale === "en" ? "Switch to French" : "Switch to English";

  return (
    <button
      type="button"
      onClick={toggleLocale}
      aria-label={t(nextLabel)}
      title={t(nextLabel)}
      className="inline-flex h-9 items-center gap-1.5 rounded-md border border-border bg-card px-2.5 text-xs font-semibold text-muted-foreground transition hover:bg-muted hover:text-foreground"
    >
      <Languages className="h-4 w-4" aria-hidden />
      <span>{locale === "en" ? "FR" : "EN"}</span>
    </button>
  );
}
