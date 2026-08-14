// Theme-aware presentational primitives shared across the app.
// Dependency-light: Tailwind + lucide-react + a `cn()` helper.

import { forwardRef, useEffect, useRef } from "react";
import {
  Loader2,
  Inbox,
  Clock3,
  AlertCircle,
  AlertTriangle,
  Info,
  X,
} from "lucide-react";
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
        "rounded-lg border border-border bg-card shadow-card",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-border px-5 py-4">
          <div className="flex min-w-0 items-start gap-3">
            {icon && (
              <span className="mt-0.5 grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border bg-muted text-accent">
                {icon}
              </span>
            )}
            <div className="min-w-0">
              {title && (
                <h2 className="truncate text-sm font-semibold text-foreground">
                  {title}
                </h2>
              )}
              {subtitle && (
                <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
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
  "inline-flex select-none items-center justify-center gap-2 whitespace-nowrap rounded-md font-medium transition focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50";

const BTN_VARIANTS: Record<ButtonVariant, string> = {
  primary:
    "bg-accent text-accent-foreground shadow-sm hover:bg-accent/90 active:bg-accent",
  secondary:
    "border border-border bg-card text-foreground hover:bg-muted",
  ghost: "text-muted-foreground hover:bg-muted hover:text-foreground",
  danger:
    "bg-rose-500 text-white shadow-sm hover:bg-rose-500/90 dark:bg-rose-600 dark:hover:bg-rose-600/90",
};

const BTN_SIZES: Record<ButtonSize, string> = {
  sm: "h-8 px-3 text-xs",
  md: "h-10 px-4 text-sm",
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
      {loading && <Loader2 className="h-4 w-4 animate-spin" />}
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
        "grid h-9 w-9 place-items-center rounded-md border border-border bg-card text-muted-foreground transition hover:bg-muted hover:text-foreground disabled:opacity-50",
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
        className="mb-1.5 flex items-center gap-1 text-xs font-medium text-foreground"
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
  "w-full rounded-md border border-input bg-card px-3 text-sm text-foreground placeholder:text-muted-foreground/70 transition focus:border-accent focus-visible:ring-2 focus-visible:ring-ring/40 focus-visible:ring-offset-0 disabled:opacity-50";

export const TextInput = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function TextInput({ className, ...props }, ref) {
  return (
    <input ref={ref} className={cn(INPUT_BASE, "h-10", className)} {...props} />
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
      className={cn(INPUT_BASE, "h-10 cursor-pointer appearance-none pr-8", className)}
      style={{
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='none' stroke='%2394a3b8' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E\")",
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
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
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
        "after:absolute after:inset-0 after:animate-shimmer after:bg-gradient-to-r after:from-transparent after:via-foreground/5 after:to-transparent",
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
    <div className="flex flex-col items-center justify-center rounded-md border border-dashed border-border bg-muted/20 px-6 py-12 text-center">
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

/** Friendly placeholder for endpoints that aren't built yet (404/501). */
export function ComingSoon({ error }: { error?: unknown }) {
  const isUnavailable = error instanceof ApiUnavailable;
  return (
    <EmptyState
      icon={<Clock3 className="h-5 w-5" />}
      title={isUnavailable ? "Coming soon" : "Couldn't load"}
      description={
        isUnavailable
          ? "This view isn't available on this server."
          : error instanceof Error
            ? error.message
            : "Not available."
      }
    />
  );
}

// ---------------------------------------------------------------------------
// Alert — the inline error/warning/info banner every page was hand-rolling.
// ---------------------------------------------------------------------------

export type AlertTone = "err" | "warn" | "info";

const ALERT_TONES: Record<AlertTone, string> = {
  err: "border-rose-500/30 bg-rose-500/10 text-rose-600 dark:text-rose-400",
  warn: "border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400",
  info: "border-accent/20 bg-accent/10 text-accent",
};

const ALERT_ICONS: Record<AlertTone, typeof AlertCircle> = {
  err: AlertCircle,
  warn: AlertTriangle,
  info: Info,
};

export function Alert({
  tone = "err",
  children,
  className,
}: {
  tone?: AlertTone;
  children: React.ReactNode;
  className?: string;
}) {
  const Icon = ALERT_ICONS[tone];
  return (
    <p
      role={tone === "err" ? "alert" : undefined}
      className={cn(
        "flex items-start gap-2 rounded-lg border px-3 py-2 text-sm",
        ALERT_TONES[tone],
        className,
      )}
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" />
      <span className="min-w-0">{children}</span>
    </p>
  );
}

// ---------------------------------------------------------------------------
// Segmented — the pill-tab row used for mode/sort/type switches.
// ---------------------------------------------------------------------------

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  className,
}: {
  value: T;
  onChange: (value: T) => void;
  options: {
    value: T;
    label: React.ReactNode;
    icon?: React.ComponentType<{ className?: string }>;
  }[];
  className?: string;
}) {
  return (
    <div
      className={cn(
        "inline-flex rounded-md border border-border bg-card p-0.5 text-xs",
        className,
      )}
    >
      {options.map((opt) => {
        const Icon = opt.icon;
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            aria-pressed={active}
            className={cn(
              "inline-flex items-center gap-1.5 rounded px-2.5 py-1 transition",
              active
                ? "bg-muted text-foreground"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {Icon && <Icon className="h-3.5 w-3.5" />}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Checkbox — matches the visual weight of the raw `<input>`s it replaces
// (h-3.5 text-xs, native browser control) so migrating a call site is a
// no-visual-diff swap; adds accent-color tinting as the one deliberate polish.
// ---------------------------------------------------------------------------

export const Checkbox = forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement> & {
    label: React.ReactNode;
    hint?: string;
  }
>(function Checkbox({ label, hint, className, ...props }, ref) {
  return (
    <label
      className={cn(
        "flex cursor-pointer items-center gap-2 text-xs text-foreground/90",
        className,
      )}
    >
      <input
        ref={ref}
        type="checkbox"
        className="h-3.5 w-3.5 accent-accent"
        {...props}
      />
      <span>
        {label}
        {hint && <span className="block text-muted-foreground">{hint}</span>}
      </span>
    </label>
  );
});

// ---------------------------------------------------------------------------
// Dialog — centered modal (generalizes Catalog's SeedDialog).
// ---------------------------------------------------------------------------

export function Dialog({
  open,
  onClose,
  title,
  subtitle,
  children,
  footer,
  className,
}: {
  open: boolean;
  onClose: () => void;
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  children: React.ReactNode;
  footer?: React.ReactNode;
  className?: string;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} aria-hidden />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={typeof title === "string" ? title : undefined}
        className={cn(
          "relative z-10 w-full max-w-md rounded-lg border border-border bg-card p-5 shadow-elevated",
          className,
        )}
      >
        <div className="mb-3 flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h3 className="text-sm font-semibold text-foreground">{title}</h3>
            {subtitle && (
              <p className="mt-0.5 truncate text-xs text-muted-foreground">
                {subtitle}
              </p>
            )}
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-m-1 grid h-8 w-8 shrink-0 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
        {footer && (
          <div className="mt-4 flex items-center justify-end gap-2">{footer}</div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sheet — right-hand slide-over (generalizes Deploy.tsx's SlideOverPanel).
//
// Persistently mounted; slides off-screen when closed so children (a form + an
// in-flight result panel) keep their state instead of unmounting. `inert`
// keeps the off-screen subtree out of the tab order and hidden from screen
// readers without tearing it down; focus moves into the panel on open and
// back to the trigger element on close, standard dialog behavior.
// ---------------------------------------------------------------------------

export function Sheet({
  open,
  onClose,
  children,
}: {
  open: boolean;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);
  const openerRef = useRef<HTMLElement | null>(null);
  const wasOpen = useRef(open);

  useEffect(() => {
    if (open && !wasOpen.current) {
      openerRef.current = document.activeElement as HTMLElement | null;
      closeRef.current?.focus();
    } else if (!open && wasOpen.current) {
      openerRef.current?.focus();
      openerRef.current = null;
    }
    wasOpen.current = open;
  }, [open]);

  return (
    <>
      <div
        className={cn(
          "fixed inset-0 z-40 bg-black/40 transition-opacity duration-200",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        onClick={onClose}
        aria-hidden
      />
      <aside
        inert={!open}
        aria-hidden={!open}
        className={cn(
          "fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col bg-background shadow-elevated transition-transform duration-200",
          open ? "translate-x-0" : "translate-x-full",
        )}
      >
        <div className="flex items-center justify-end border-b border-border px-3 py-2">
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close panel"
            className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition hover:bg-muted hover:text-foreground"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="scroll-thin flex-1 overflow-y-auto p-4">{children}</div>
      </aside>
    </>
  );
}
