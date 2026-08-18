"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/cn";
import { useI18n } from "@/lib/i18n";

/** Right-aligned pager: "Page x of y" + Prev/Next (disabled at bounds). */
export function Pager({
  page,
  pageCount,
  onPrev,
  onNext,
}: {
  page: number; // 1-indexed
  pageCount: number;
  onPrev: () => void;
  onNext: () => void;
}) {
  const { t } = useI18n();
  const atStart = page <= 1;
  const atEnd = page >= pageCount;
  return (
    <div className="flex items-center gap-1 text-xs text-muted-foreground">
      <span className="tabular-nums">
        {t("Page {page} of {pageCount}", {
          page: Math.min(page, pageCount || 1),
          pageCount: pageCount || 1,
        })}
      </span>
      <button
        type="button"
        onClick={onPrev}
        disabled={atStart}
        aria-label={t("Previous page")}
        className={cn(
          "grid h-7 w-7 place-items-center rounded-md border border-border bg-card transition hover:bg-muted disabled:opacity-40 disabled:hover:bg-card",
        )}
      >
        <ChevronLeft className="h-4 w-4" />
      </button>
      <button
        type="button"
        onClick={onNext}
        disabled={atEnd}
        aria-label={t("Next page")}
        className={cn(
          "grid h-7 w-7 place-items-center rounded-md border border-border bg-card transition hover:bg-muted disabled:opacity-40 disabled:hover:bg-card",
        )}
      >
        <ChevronRight className="h-4 w-4" />
      </button>
    </div>
  );
}
