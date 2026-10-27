"use client";

import { cn } from "@/lib/cn";
import { useTranslatedNode } from "@/lib/i18n";

// ---------------------------------------------------------------------------
// Stat tiles and a capacity meter.
//
// The form follows the data's job rather than taste: a count is a number, and a
// measure against a ceiling is a meter. Neither is a chart, and reaching for one
// would add axes and a legend to communicate two numbers.
//
// Shared rather than local to one page because every section lands on the same
// question -- how much is running, how much room is left -- and they should
// answer it in the same shape.
// ---------------------------------------------------------------------------

/** How a capacity reading is doing. Named by SEVERITY, not by colour. */
export type MeterTone = "ok" | "warn" | "err";

const METER_FILL: Record<MeterTone, string> = {
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  err: "bg-rose-500",
};

/**
 * Headroom, by fraction used.
 *
 * Thresholds rather than a gradient: an operator acts at a boundary ("nearly
 * full"), not on a continuum, and a smoothly shifting hue gives no moment to
 * act on.
 */
export function meterTone(fractionUsed: number): MeterTone {
  if (fractionUsed >= 0.9) return "err";
  if (fractionUsed >= 0.75) return "warn";
  return "ok";
}

/**
 * A horizontal capacity bar.
 *
 * The numbers live beside it as text, always. Colour alone cannot carry the
 * reading -- it is unavailable to a colourblind reader, and a bar that is
 * "quite full" is not a quantity. The fill is a status colour, reserved for
 * exactly this and never used to tell two series apart.
 */
export function Meter({
  fraction,
  tone,
  className,
  label,
}: {
  /** 0..1, clamped. */
  fraction: number;
  tone?: MeterTone;
  className?: string;
  /** Describes the reading for a screen reader, e.g. "memory used". */
  label?: string;
}) {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(fraction) ? fraction : 0));
  const resolved = tone ?? meterTone(clamped);
  const pct = Math.round(clamped * 100);
  return (
    <div
      role="meter"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={label}
      className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}
    >
      <div
        // Rounded data-end, anchored to the baseline at the left. A zero-width
        // fill stays invisible rather than rendering a stub that reads as "a
        // little", which is a different claim from "none".
        className={cn("h-full rounded-full transition-all duration-500", METER_FILL[resolved])}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

/**
 * One reading: a quiet label, a value with weight, and optional supporting
 * detail beneath.
 *
 * The label is sentence case at a readable size. Uppercase micro-caps at 11px
 * -- the pattern this replaces -- cost legibility to buy a look, and on a
 * muted foreground they were close to unreadable.
 */
export function StatTile({
  label,
  value,
  detail,
  meter,
  className,
}: {
  label: React.ReactNode;
  value: React.ReactNode;
  detail?: React.ReactNode;
  meter?: React.ReactNode;
  className?: string;
}) {
  const tx = useTranslatedNode();
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-card px-4 py-3",
        className,
      )}
    >
      <p className="text-xs font-medium text-muted-foreground">{tx(label)}</p>
      <div className="mt-1 text-xl font-semibold leading-7 tabular-nums text-foreground">
        {value}
      </div>
      {meter && <div className="mt-2.5">{meter}</div>}
      {detail && (
        <p className="mt-1.5 text-xs leading-5 text-muted-foreground">{tx(detail)}</p>
      )}
    </div>
  );
}

/** A row of tiles that reflows rather than scrolling on a narrow viewport. */
export function StatRow({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid gap-3 sm:grid-cols-2 lg:grid-cols-3",
        className,
      )}
    >
      {children}
    </div>
  );
}
