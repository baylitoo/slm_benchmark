"use client";

import { ComingSoon, EmptyState, Skeleton, TableShell, THead, Th, Td } from "./ui";
import { JsonView } from "./JsonView";

/**
 * Renders a list of records as a table when possible, with graceful loading /
 * coming-soon / empty states. Object rows derive their columns from keys.
 */
export function DataTable({
  rows,
  loading,
  error,
  emptyLabel = "No records yet.",
  emptyDescription,
}: {
  rows: unknown[] | null;
  loading: boolean;
  error: unknown;
  emptyLabel?: string;
  emptyDescription?: string;
}) {
  if (loading && !rows) {
    return (
      <div className="space-y-2">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={i} className="h-10 w-full" />
        ))}
      </div>
    );
  }
  if (error && !rows) {
    return <ComingSoon error={error} />;
  }
  if (!rows || rows.length === 0) {
    return <EmptyState title={emptyLabel} description={emptyDescription} />;
  }

  const allObjects = rows.every((r) => r && typeof r === "object" && !Array.isArray(r));
  if (!allObjects) {
    return <JsonView value={rows} />;
  }

  const objs = rows as Record<string, unknown>[];
  const columns = Array.from(new Set(objs.flatMap((r) => Object.keys(r))));

  return (
    <TableShell>
      <THead>
        <tr>
          {columns.map((c) => (
            <Th key={c}>{c}</Th>
          ))}
        </tr>
      </THead>
      <tbody>
        {objs.map((row, i) => (
          <tr
            key={i}
            className="border-t border-border transition-colors hover:bg-muted/40"
          >
            {columns.map((c) => (
              <Td
                key={c}
                className="max-w-[24rem] truncate whitespace-nowrap"
                title={cellTitle(row[c])}
              >
                {renderCell(row[c])}
              </Td>
            ))}
          </tr>
        ))}
      </tbody>
    </TableShell>
  );
}

function renderCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function cellTitle(value: unknown): string | undefined {
  if (value === null || value === undefined) return undefined;
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
