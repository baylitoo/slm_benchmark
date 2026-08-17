"use client";

import type { ReactNode } from "react";
import { RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";

/**
 * The "Showing X of Y · Fetch · Pager" line that sits above a table.
 * `onFetch` maps to the existing poller `refresh()`/`reload()`; the pager slot
 * is a <Pager> the page wires to its local page state.
 */
export function ResultLine({
  shown,
  total,
  noun = "results",
  onFetch,
  fetching,
  pager,
}: {
  shown: number;
  total: number;
  noun?: string;
  onFetch?: () => void;
  fetching?: boolean;
  pager?: ReactNode;
}) {
  const { t } = useI18n();
  return (
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground">
      <div className="flex items-center gap-2">
        <span>
          {t("Showing")} <span className="tabular-nums text-foreground">{shown}</span>{" "}
          {t("of")} <span className="tabular-nums text-foreground">{total}</span> {t(noun)}
        </span>
        {onFetch && (
          <button
            type="button"
            onClick={onFetch}
            className="inline-flex items-center gap-1.5 rounded-md border border-border bg-card px-2 py-1 font-medium text-foreground transition hover:bg-muted"
          >
            <RefreshCw className={cn("h-3.5 w-3.5", fetching && "animate-spin")} />
            {t("Fetch")}
          </button>
        )}
      </div>
      {pager}
    </div>
  );
}
