// Theme-aware presentational primitives shared across the app.
// Dependency-light: Tailwind + lucide-react + a `cn()` helper.

import { forwardRef } from "react";
import { Loader2, Inbox, Clock3 } from "lucide-react";
import { ApiUnavailable } from "@/lib/api";
import { cn } from "@/lib/cn";

// ---------------------------------------------------------------------------
// Card
// ---------------------------------------------------------------------------

export function Card({
  title,
  subtitle,
  children,
  actions,
  icon,
  className,
  bodyClassName,
}: {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  children?: React.ReactNode;
  actions?: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section
      className={cn(
        "rounded-xl border border-border bg-card shadow-card",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            {icon && (
              <span className="mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted text-muted-foreground">
                {icon}
              </span>
            )}
            <div className="min-w-0">
              {title && (
                <h2 className="truncate text-sm font-semibold tracking-tightish text-foreground">
                  {title}
                </h2>
              )}
              {subtitle && (
                <p className="mt-0.5 text-[13px] text-muted-foreground">{subtitle}</p>
              )}
            </div>
          </div>
          {actions && <div className="flex shrink-0 items-center gap-2">{actions}</div>}
        </header>
      )}
      <div className={cn("p-5", bodyClassName)}>{children}</div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Button
// ---------------------------------------------------------------------------

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
type ButtonSize = "sm" | "md";

const BTN_BASE =
  "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-lg font-medium tracking-tightish transition-[background,box-shadow,transform] duration-150 ease-swift active:translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring/60 focus-visible:ring-offset-2 focus-visible:ring-offset-background disabled:pointer-events-none disabled:opacity-50";

const BTN_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-accent-foreground shadow-xs hover:brightness-[1.07] active:brightness-95",
  secondary:
    "border border-border bg-card text-foreground hover:bg-muted",
  ghost: "text-muted-foreground hover:bg-muted hover:text-foreground",
  danger:
    "bg-rose-500 text-white shadow-xs hover:bg-rose-600 dark:bg-rose-600 dark:hover:bg-rose-500",
};

const BTN_SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs rounded-lg",
  md: "h-9 px-3.5 text-sm rounded-lg",
};

export const Button = forwardRef<
  HTMLButtonElement,
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: ButtonVariant;
    size?: ButtonSize;
    loading?: boolean;
  }
>(function Button(
  { children, className, variant = "primary", size = "md", loading, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(BTN_BASE, BTN_VARIANTS[variant], BTN_SIZES[size], className)}
      {...props}
    >
      {loading && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
});

export function IconButton({
  label,
  className,
  children,
  ...props
}: React.ButtonHTMLAttributes<HTMLButtonElement> & { label: string }) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      className={cn(
        "grid h-8 w-8 place-items-center rounded-lg border border-border bg-transparent text-muted-foreground transition hover:bg-muted hover:text-foreground disabled:opacity-50",
        className,
      )}
      {...props}
    >
      {children}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Form fields
// ---------------------------------------------------------------------------

export function Field({
  label,
  hint,
  htmlFor,
  required,
  children,
  className,
}: {
  label: string;
  hint?: string;
  htmlFor?: string;
  required?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("block", className)}>
      <label
        htmlFor={htmlFor}
        className="mb-1.5 flex items-center gap-1 text-xs font-medium text-muted-foreground"
      >
        {label}
        {required && <span className="text-rose-500">*</span>}
      </label>
      {children}
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

const INPUT_BASE =
  "w-full rounded-lg border border-input bg-background px-3 text-sm text-foreground placeholder:text-muted-foreground/70 transition focus:border-accent focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:ring-offset-0 disabled:opacity-50";

export const TextInput = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function TextInput({ className, ...props }, ref) {
  return (
    <input ref={ref} className={cn(INPUT_BASE, "h-9", className)} {...props} />
  );
});

export const TextArea = forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(function TextArea({ className, ...props }, ref) {
  return (
    <textarea
      ref={ref}
      className={cn(INPUT_BASE, "resize-y py-2 leading-relaxed", className)}
      {...props}
    />
  );
});

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...props }, ref) {
  return (
    <select
      ref={ref}
      className={cn(INPUT_BASE, "h-9 cursor-pointer appearance-none pr-8", className)}
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%23a1a1aa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 0.6rem center",
      }}
      {...props}
    >
      {children}
    </select>
  );
});

// ---------------------------------------------------------------------------
// Badge / status
// ---------------------------------------------------------------------------

export type BadgeTone = "neutral" | "ok" | "warn" | "err" | "info";

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: "bg-muted text-muted-foreground border-border",
  ok: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400",
  warn: "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400",
  err: "bg-rose-500/10 text-rose-600 border-rose-500/20 dark:text-rose-400",
  info: "bg-accent/10 text-accent border-accent/20",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: React.ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

const DOT_TONES: Record<BadgeTone, string> = {
  neutral: "bg-muted-foreground",
  ok: "bg-emerald-500",
  warn: "bg-amber-500",
  err: "bg-rose-500",
  info: "bg-accent",
};

export function StatusDot({
  tone = "neutral",
  pulse,
  className,
}: {
  tone?: BadgeTone;
  pulse?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-block h-2 w-2 rounded-full",
        DOT_TONES[tone],
        pulse && "animate-pulse-dot",
        className,
      )}
    />
  );
}

// ---------------------------------------------------------------------------
// Skeleton / spinner / empty / coming-soon
// ---------------------------------------------------------------------------

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md bg-muted",
        "after:absolute after:inset-0 after:animate-shimmer after:bg-gradient-to-r after:from-transparent after:via-foreground/[0.04] after:to-transparent",
        className,
      )}
    />
  );
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("h-4 w-4 animate-spin", className)} />;
}

export function EmptyState({
  title,
  description,
  icon,
}: {
  title: string;
  description?: string;
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-muted/40 px-6 py-10 text-center">
      <span className="mb-3 grid h-11 w-11 place-items-center rounded-full border border-border bg-card text-muted-foreground">
        {icon ?? <Inbox className="h-5 w-5" />}
      </span>
      <p className="text-sm font-medium text-foreground">{title}</p>
      {description && (
        <p className="mx-auto mt-1 max-w-sm text-xs text-muted-foreground">
          {description}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// SegmentedControl — a small tablist-style toggle (text/file, table/ports…).
// Stateless: value + onChange live in the parent.
// ---------------------------------------------------------------------------

export function SegmentedControl<T extends string>({
  options,
  value,
  onChange,
  className,
  ariaLabel,
}: {
  options: { value: T; label: React.ReactNode; icon?: React.ReactNode }[];
  value: T;
  onChange: (value: T) => void;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <div
      role="tablist"
      aria-label={ariaLabel}
      className={cn(
        "inline-flex rounded-lg border border-border bg-muted p-0.5 text-sm",
        className,
      )}
    >
      {options.map((opt) => {
        const isActive = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(opt.value)}
            className={cn(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 transition-[background,color] duration-150 ease-swift",
              isActive
                ? "bg-card text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Table shell — shared chrome for the app's data tables (presentational only).
// ---------------------------------------------------------------------------

export function TableShell({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "scroll-thin overflow-auto rounded-xl border border-border",
        className,
      )}
    >
      <table className="w-full text-left text-sm">{children}</table>
    </div>
  );
}

export function THead({ children }: { children: React.ReactNode }) {
  return (
    <thead className="bg-muted/50 text-xs uppercase tracking-wide text-muted-foreground">
      {children}
    </thead>
  );
}

export function Th({
  children,
  className,
}: {
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <th
      className={cn("whitespace-nowrap px-3 py-2.5 font-medium", className)}
    >
      {children}
    </th>
  );
}

export function Td({
  children,
  className,
  title,
}: {
  children?: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <td
      title={title}
      className={cn("px-3 py-2.5 text-foreground/90", className)}
    >
      {children}
    </td>
  );
}

/** Friendly placeholder for endpoints that aren't built yet (404/501). */
export function ComingSoon({ error }: { error?: unknown }) {
  const isUnavailable = error instanceof ApiUnavailable;
  return (
    <EmptyState
      icon={<Clock3 className="h-5 w-5" />}
      title={isUnavailable ? "Coming soon" : "Couldn't load"}
      description={
        isUnavailable
          ? "The backend route for this view isn't available yet. The UI is ready and will light up automatically once it ships."
          : error instanceof Error
            ? error.message
            : "Not available."
      }
    />
  );
}
