"use client";

import { Fragment, useMemo, useState, type ReactNode } from "react";
import { ChevronUp, ChevronDown, ChevronsUpDown } from "lucide-react";
import { cn } from "@/lib/cn";
import { ComingSoon, EmptyState, Skeleton } from "../ui";

export interface Column<T> {
  key: string;
  header: string;
  render: (row: T) => ReactNode;
  className?: string;
  /** When provided, the column header becomes a client-side sort toggle. */
  sortAccessor?: (row: T) => string | number | null | undefined;
}

type SortDir = "asc" | "desc";

/**
 * Explicit-column table with header sort arrows, hairline rows, horizontal
 * scroll, and loading / error / empty states. Sorting is client-side over the
 * already-fetched rows (no new fetches). Rows may expand via `renderExpanded`.
 */
export function Table<T>({
  columns,
  rows,
  getRowKey,
  loading,
  error,
  emptyLabel = "No records found",
  emptyDescription,
  emptyIcon,
  renderExpanded,
  expandedKey,
  onRowClick,
}: {
  columns: Column<T>[];
  rows: T[] | null;
  getRowKey: (row: T, index: number) => string;
  loading?: boolean;
  error?: unknown;
  emptyLabel?: string;
  emptyDescription?: string;
  emptyIcon?: ReactNode;
  /** Optional expandable detail row. */
  renderExpanded?: (row: T) => ReactNode;
  /** Row key currently expanded (controlled by the parent). */
  expandedKey?: string | null;
  onRowClick?: (row: T) => void;
}) {
  const [sortKey, setSortKey] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const sorted = useMemo(() => {
    if (!rows) return rows;
    const col = columns.find((c) => c.key === sortKey);
    if (!col || !col.sortAccessor) return rows;
    const acc = col.sortAccessor;
    const copy = [...rows];
    copy.sort((a, b) => {
      const va = acc(a);
      const vb = acc(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      let cmp: number;
      if (typeof va === "number" && typeof vb === "number") cmp = va - vb;
      else cmp = String(va).localeCompare(String(vb));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [rows, columns, sortKey, sortDir]);

  function toggleSort(col: Column<T>) {
    if (!col.sortAccessor) return;
    if (sortKey === col.key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(col.key);
      setSortDir("asc");
    }
  }

  if (loading && !rows) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <Skeleton key={i} className="h-11 w-full" />
        ))}
      </div>
    );
  }
  if (error && !rows) {
    return <ComingSoon error={error} />;
  }
  if (!sorted || sorted.length === 0) {
    return <EmptyState title={emptyLabel} description={emptyDescription} icon={emptyIcon} />;
  }

  return (
    <div className="scroll-thin overflow-x-auto rounded-md border border-border">
      <table className="w-full text-left text-sm">
        <thead className="bg-muted/60 text-[11px] uppercase tracking-wider text-muted-foreground">
          <tr>
            {columns.map((col) => {
              const isSorted = sortKey === col.key;
              return (
                <th
                  key={col.key}
                  className={cn("whitespace-nowrap px-3 py-2.5 font-medium", col.className)}
                >
                  {col.sortAccessor ? (
                    <button
                      type="button"
                      onClick={() => toggleSort(col)}
                      className="inline-flex items-center gap-1 uppercase hover:text-foreground"
                    >
                      {col.header}
                      {isSorted ? (
                        sortDir === "asc" ? (
                          <ChevronUp className="h-3 w-3" />
                        ) : (
                          <ChevronDown className="h-3 w-3" />
                        )
                      ) : (
                        <ChevronsUpDown className="h-3 w-3 opacity-40" />
                      )}
                    </button>
                  ) : (
                    col.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const key = getRowKey(row, i);
            const expanded = expandedKey != null && expandedKey === key;
            return (
              <Fragment key={key}>
                <tr
                  onClick={onRowClick ? () => onRowClick(row) : undefined}
                  className={cn(
                    "border-t border-border transition-colors hover:bg-muted/40",
                    onRowClick && "cursor-pointer",
                  )}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={cn("px-3 py-2.5 align-middle text-foreground/90", col.className)}
                    >
                      {col.render(row)}
                    </td>
                  ))}
                </tr>
                {expanded && renderExpanded && (
                  <tr className="border-t border-border bg-muted/20">
                    <td colSpan={columns.length} className="px-3 py-3">
                      {renderExpanded(row)}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
