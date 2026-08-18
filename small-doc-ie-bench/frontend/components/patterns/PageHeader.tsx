"use client";

import type { ReactNode } from "react";
import { useI18n, useTranslatedNode } from "@/lib/i18n";

/** Page title on the left, primary action on the right (LiteLLM pattern). */
export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: ReactNode;
  actions?: ReactNode;
}) {
  const { t } = useI18n();
  const tx = useTranslatedNode();
  return (
    <div className="mb-5 flex items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold text-foreground">{t(title)}</h1>
        {subtitle && <p className="mt-0.5 text-sm text-muted-foreground">{tx(subtitle)}</p>}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
    </div>
  );
}
