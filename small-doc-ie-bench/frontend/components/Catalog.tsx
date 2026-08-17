"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Rocket,
  Info,
  AlertCircle,
  Flame,
  Download,
  Heart,
  Clock,
} from "lucide-react";
import {
  HF_PARAMETER_RANGES,
  searchCatalog,
  seedHf,
  formatBytes,
  type CatalogCard,
  type CatalogSort,
  type ModelFamily,
} from "@/lib/api";
import { useAsync } from "@/lib/useAsync";
import { cn } from "@/lib/cn";
import { Alert, Badge, Button, Checkbox, Dialog, Field, Segmented, Select, TextInput } from "./ui";
import { useToast } from "./Toast";
import { Table, type Column } from "./patterns/Table";
import { Toolbar } from "./patterns/Toolbar";
import { ResultLine } from "./patterns/ResultLine";
import { errText, FamilyOptions } from "./deploy/shared";
import { defaultStoreName } from "./deploy/seed/ExistingModelsDialog";

// ---------------------------------------------------------------------------
// Support tier — verdict + resolved family → a display badge. Mirrors the
// catalog artifact's tiers. EVERY field the tier reads is defensively guarded
// (the backend may still be emitting the plain search shape, so verdict/family
// arrive undefined): an unrated card falls through to "Unrated", never crashes.
// ---------------------------------------------------------------------------

type Tier = "ready" | "override" | "fallback" | "gap" | "unrated";

const TRANSFORMERS_FAMILIES = new Set(["transformers", "transformers_trust_remote_code"]);

const TIER_META: Record<Tier, { label: string; rank: number; className: string; hint: string }> = {
  ready: {
    label: "Ready",
    rank: 0,
    className: "bg-emerald-500/10 text-emerald-600 border-emerald-500/20 dark:text-emerald-400",
    hint: "Supported out of the box — a native GGUF / encoder serving family.",
  },
  override: {
    label: "Override",
    rank: 1,
    className: "bg-amber-500/10 text-amber-600 border-amber-500/20 dark:text-amber-400",
    hint: "Pick a family before seeding — no confirmed family for this architecture yet.",
  },
  fallback: {
    label: "Fallback",
    rank: 2,
    className: "bg-violet-500/10 text-violet-600 border-violet-500/20 dark:text-violet-400",
    hint: "Last-resort transformers runtime (no GGUF path) — heavier RAM, slower.",
  },
  gap: {
    label: "Gap",
    rank: 3,
    className: "bg-rose-500/10 text-rose-600 border-rose-500/20 dark:text-rose-400",
    hint: "Unsupported — no serving path resolves for this architecture.",
  },
  unrated: {
    label: "Unrated",
    rank: 4,
    className: "bg-muted text-muted-foreground border-border",
    hint: "No pre-verdict yet — open the per-repo inspect to confirm support.",
  },
};

function tierOf(card: CatalogCard): Tier {
  // A transformers last-resort family is a Fallback even when the verdict reads
  // "supported" — it's the no-GGUF path, so it must not read as green Ready.
  if (card.family && TRANSFORMERS_FAMILIES.has(card.family)) return "fallback";
  switch (card.verdict) {
    case "supported":
      return "ready";
    case "needs_family":
      return "override";
    case "unsupported":
      return "gap";
    default:
      return "unrated";
  }
}

function TierBadge({ tier }: { tier: Tier }) {
  const meta = TIER_META[tier];
  return (
    <span
      title={meta.hint}
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium",
        meta.className,
      )}
    >
      {meta.label}
    </span>
  );
}

/**
 * "Fits this node" badge, judged against the deploy budget (free − safety
 * margin). Defensive: only true/false render — `null`/`undefined` (unknown, no
 * node snapshot) renders NOTHING, so a missing judgment never reads as "fits".
 */
function FitBadge({ card }: { card: CatalogCard }) {
  if (card.fits_node == null) return null;
  const budget =
    card.node_available_bytes != null
      ? `node has ~${formatBytes(card.node_available_bytes)} free (after margin)`
      : undefined;
  return (
    <Badge tone={card.fits_node ? "ok" : "err"} className={budget ? "cursor-help" : undefined}>
      <span title={budget}>{card.fits_node ? "Fits" : "Too big"}</span>
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Formatting helpers (all null/undefined tolerant — render a dash).
// ---------------------------------------------------------------------------

function Dash({ title }: { title?: string }) {
  return (
    <span className="text-muted-foreground" title={title}>
      —
    </span>
  );
}

/** 1234 -> "1.2K", 1_200_000 -> "1.2M", 3_400_000_000 -> "3.4B". */
function humanCount(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return "—";
  const abs = Math.abs(n);
  if (abs < 1000) return String(n);
  const units: [number, string][] = [
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "K"],
  ];
  for (const [divisor, suffix] of units) {
    if (abs >= divisor) {
      const v = n / divisor;
      const label = v >= 100 ? String(Math.round(v)) : v.toFixed(1).replace(/\.0$/, "");
      return `${label}${suffix}`;
    }
  }
  return String(n);
}

/** "updated 3d ago" style relative label from an ISO stamp. */
function agoLabel(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "—";
  const secs = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (secs < 60) return `${secs}s ago`;
  const mins = Math.round(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 48) return `${hrs}h ago`;
  const days = Math.round(hrs / 24);
  if (days < 365) return `${days}d ago`;
  return `${Math.round(days / 365)}y ago`;
}

// ---------------------------------------------------------------------------
// Facets.
// ---------------------------------------------------------------------------

const SORTS: { id: CatalogSort; label: string; icon: typeof Flame }[] = [
  { id: "trending", label: "Trending", icon: Flame },
  { id: "downloads", label: "Popular", icon: Download },
  { id: "recent", label: "New", icon: Clock },
  { id: "likes", label: "Most-liked", icon: Heart },
];

// value === "" means "All". `text-generation` etc. are the Hub pipeline tags.
const PIPELINES: { value: string; label: string }[] = [
  { value: "", label: "All tasks" },
  { value: "text-generation", label: "Chat" },
  { value: "image-text-to-text", label: "Vision" },
  { value: "feature-extraction", label: "Embeddings" },
  { value: "token-classification", label: "NER" },
];

const LIMIT = 30;
const DEBOUNCE_MS = 350;

// ---------------------------------------------------------------------------
// Catalog view — an Azure-Foundry-like Hub browser. Rendered as a `deploy`
// sub-view (Deploy owns the PageHeader), mirroring the Sizing tab: this returns
// a plain content column with no header of its own.
// ---------------------------------------------------------------------------

export function Catalog({
  families,
  existingStoreNames,
  onSeeded,
}: {
  families: ModelFamily[] | null;
  existingStoreNames: string[];
  onSeeded?: () => void;
}) {
  const [query, setQuery] = useState("");
  const [debounced, setDebounced] = useState("");
  const [sort, setSort] = useState<CatalogSort>("trending");
  const [pipeline, setPipeline] = useState("");
  const [ggufOnly, setGgufOnly] = useState(true);
  const [author, setAuthor] = useState("");
  const [parameterRange, setParameterRange] = useState("");
  const [familyFilter, setFamilyFilter] = useState(""); // client-side, "" = all
  const [seedCard, setSeedCard] = useState<CatalogCard | null>(null);

  // Debounce the free-text query so each keystroke doesn't fire a request.
  useEffect(() => {
    const id = setTimeout(() => setDebounced(query.trim()), DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [query]);

  // Key encodes every server-side param INCLUDING the (possibly empty) query,
  // so the first paint fires a real request: empty query + a sort = top models.
  const key =
    `catalog:${sort}:${pipeline}:${ggufOnly}:${author.trim()}:` +
    `${parameterRange}:${debounced}:${LIMIT}`;
  const { data, error, loading, reload } = useAsync<CatalogCard[]>(key, () =>
    searchCatalog({
      query: debounced,
      sort,
      pipeline_tag: pipeline || null,
      author: author.trim() || null,
      num_parameters: parameterRange || null,
      gguf_only: ggufOnly,
      limit: LIMIT,
    }),
  );

  // Family filter is client-side over the returned cards.
  const familyOptions = useMemo(() => {
    const set = new Set<string>();
    for (const c of data ?? []) if (c.family) set.add(c.family);
    return [...set].sort();
  }, [data]);

  const rows = useMemo(() => {
    if (!data) return null;
    if (!familyFilter) return data;
    return data.filter((c) => c.family === familyFilter);
  }, [data, familyFilter]);

  const total = data?.length ?? 0;
  const shown = rows?.length ?? 0;

  const filtersDirty =
    query !== "" ||
    sort !== "trending" ||
    pipeline !== "" ||
    !ggufOnly ||
    author !== "" ||
    parameterRange !== "" ||
    familyFilter !== "";

  function resetFilters() {
    setQuery("");
    setSort("trending");
    setPipeline("");
    setGgufOnly(true);
    setAuthor("");
    setParameterRange("");
    setFamilyFilter("");
  }

  const columns: Column<CatalogCard>[] = [
    {
      key: "repo",
      header: "Model",
      className: "max-w-[22rem]",
      sortAccessor: (c) => c.id,
      render: (c) => (
        <div className="min-w-0">
          <span className="block truncate font-mono text-xs text-foreground" title={c.id}>
            {c.id}
          </span>
          {c.runtime_note && (
            <span
              className="mt-0.5 inline-flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400"
              title={c.runtime_note}
            >
              <AlertCircle className="h-3 w-3 shrink-0" />
              <span className="truncate">{c.runtime_note}</span>
            </span>
          )}
        </div>
      ),
    },
    {
      key: "tier",
      header: "Tier",
      sortAccessor: (c) => TIER_META[tierOf(c)].rank,
      render: (c) => (
        <div className="flex flex-wrap items-center gap-1.5">
          <TierBadge tier={tierOf(c)} />
          <FitBadge card={c} />
        </div>
      ),
    },
    {
      key: "family",
      header: "Family",
      sortAccessor: (c) => c.family ?? "",
      render: (c) =>
        c.family ? (
          <span className="font-mono text-xs text-foreground/90">{c.family}</span>
        ) : (
          <Dash title={c.reason || "No family resolved yet."} />
        ),
    },
    {
      key: "params",
      header: "Params",
      sortAccessor: (c) => c.params ?? 0,
      render: (c) => (
        <div className="leading-tight">
          <span className="font-mono tabular-nums text-xs text-foreground">
            {c.param_label ?? "—"}
          </span>
          {c.size_est_bytes != null && (
            <span
              className="block text-[10px] text-muted-foreground"
              title="Rough on-disk / RAM estimate"
            >
              ~{formatBytes(c.size_est_bytes)}
            </span>
          )}
        </div>
      ),
    },
    {
      key: "downloads",
      header: "Downloads",
      sortAccessor: (c) => c.downloads ?? c.downloads_all_time ?? 0,
      render: (c) => (
        <span
          className="font-mono tabular-nums text-xs"
          title={
            c.downloads_all_time != null
              ? `${c.downloads_all_time.toLocaleString()} all-time`
              : undefined
          }
        >
          {humanCount(c.downloads ?? c.downloads_all_time)}
        </span>
      ),
    },
    {
      key: "likes",
      header: "Likes",
      sortAccessor: (c) => c.likes ?? 0,
      render: (c) => <span className="font-mono tabular-nums text-xs">{humanCount(c.likes)}</span>,
    },
    {
      key: "updated",
      header: "Updated",
      sortAccessor: (c) => (c.last_modified ? Date.parse(c.last_modified) || 0 : 0),
      render: (c) => (
        <span className="whitespace-nowrap text-xs text-muted-foreground">
          {agoLabel(c.last_modified)}
        </span>
      ),
    },
    {
      key: "license",
      header: "License",
      render: (c) => (
        <div className="flex items-center gap-1.5">
          {c.gated && <Badge tone="warn">gated</Badge>}
          {c.license ? (
            <span className="text-xs text-muted-foreground">{c.license}</span>
          ) : (
            !c.gated && <Dash />
          )}
        </div>
      ),
    },
    {
      key: "actions",
      header: "",
      className: "text-right",
      render: (c) => {
        const tier = tierOf(c);
        const disabled = tier === "gap";
        return (
          <Button
            size="sm"
            variant="secondary"
            disabled={disabled}
            title={
              disabled
                ? "No serving path resolves for this model — nothing to seed."
                : "Seed this model into the store"
            }
            onClick={() => setSeedCard(c)}
          >
            <Rocket className="h-3.5 w-3.5" />
            Seed
          </Button>
        );
      },
    },
  ];

  return (
    <div className="space-y-4">
      {/* Pre-check disclaimer */}
      <p className="flex items-start gap-2 rounded-md border border-border bg-muted/30 px-3 py-2 text-xs text-muted-foreground">
        <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" />
        <span>
          The tier is a fast pre-check read from each model&apos;s architecture. The per-repo
          inspect confirms support before anything is downloaded.
        </span>
      </p>

      {/* Sort tabs + search */}
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          value={sort}
          onChange={setSort}
          options={SORTS.map((s) => ({ value: s.id, label: s.label, icon: s.icon }))}
        />
        <div className="ml-auto w-full sm:w-72">
          <TextInput
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search the Hub (e.g. lfm2, nuextract, gliner)…"
            className="h-9 text-sm"
            aria-label="Search models"
          />
        </div>
      </div>

      {/* Facets */}
      <Toolbar onReset={resetFilters} resetDisabled={!filtersDirty}>
        <Select
          value={pipeline}
          onChange={(e) => setPipeline(e.target.value)}
          className="h-8 w-40 text-xs"
          aria-label="Task"
        >
          {PIPELINES.map((p) => (
            <option key={p.value} value={p.value}>
              {p.label}
            </option>
          ))}
        </Select>
        <Select
          value={familyFilter}
          onChange={(e) => setFamilyFilter(e.target.value)}
          className="h-8 w-44 text-xs"
          aria-label="Family"
          disabled={familyOptions.length === 0}
        >
          <option value="">All families</option>
          {familyOptions.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </Select>
        <Select
          value={parameterRange}
          onChange={(e) => setParameterRange(e.target.value)}
          className="h-8 w-40 text-xs"
          aria-label="Model parameters"
        >
          {HF_PARAMETER_RANGES.map((range) => (
            <option key={range.value} value={range.value}>
              {range.label}
            </option>
          ))}
        </Select>
        <TextInput
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder="Author (e.g. LiquidAI)"
          className="h-8 w-44 text-xs"
          aria-label="Author"
        />
        <Checkbox
          checked={ggufOnly}
          onChange={(e) => setGgufOnly(e.target.checked)}
          label="GGUF only"
          className="rounded-md border border-border bg-card px-2.5 py-1.5 text-muted-foreground"
        />
      </Toolbar>

      <ResultLine
        shown={shown}
        total={total}
        noun="models"
        onFetch={reload}
        fetching={loading}
      />

      <Table<CatalogCard>
        columns={columns}
        rows={rows}
        getRowKey={(c, i) => c.id || `card-${i}`}
        loading={loading}
        error={error}
        emptyLabel="No models found"
        emptyDescription="Try a different search, task, or sort — or clear the filters."
      />

      {seedCard && (
        <SeedDialog
          card={seedCard}
          families={families}
          alreadyExists={existingStoreNames.includes(defaultStoreName(seedCard.id))}
          onClose={() => setSeedCard(null)}
          onSeeded={onSeeded}
        />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Seed dialog — a small confirm that reuses the existing seed mutation
// (`seedHf`). Family is prefilled from the card's resolved family; for an
// "Override" (needs_family) card the operator picks one before seeding.
// ---------------------------------------------------------------------------

function SeedDialog({
  card,
  families,
  alreadyExists,
  onClose,
  onSeeded,
}: {
  card: CatalogCard;
  families: ModelFamily[] | null;
  alreadyExists: boolean;
  onClose: () => void;
  onSeeded?: () => void;
}) {
  const { toast } = useToast();
  const [family, setFamily] = useState(card.family ?? "");
  const [quant, setQuant] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    if (!family) {
      toast({ title: "Pick a family first", tone: "error" });
      return;
    }
    setBusy(true);
    try {
      await seedHf({
        repo: card.id,
        quant: quant.trim() || null,
        quant_prefer: true,
        family,
      });
      toast({
        title: alreadyExists ? "Using existing model" : "Download started",
        description: `${card.id} — continues in the background (appears under Models when done).`,
        tone: "success",
      });
      onSeeded?.();
      onClose();
    } catch (err) {
      toast({ title: "Seed failed", description: errText(err, "Seed failed."), tone: "error" });
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open
      onClose={onClose}
      title="Seed model"
      subtitle={
        <span className="font-mono" title={card.id}>
          {card.id}
        </span>
      }
      footer={
        <>
          <Button variant="secondary" size="sm" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button size="sm" loading={busy} disabled={!family} onClick={() => void submit()}>
            <Rocket className="h-3.5 w-3.5" />
            {alreadyExists ? "Use existing model" : "Download & seed"}
          </Button>
        </>
      }
    >
      <div className="mb-3 flex flex-wrap items-center gap-1.5">
        <TierBadge tier={tierOf(card)} />
        <FitBadge card={card} />
        {card.architecture && <Badge tone="neutral">arch: {card.architecture}</Badge>}
        {card.param_label && <Badge tone="neutral">{card.param_label}</Badge>}
      </div>

      {card.reason && <p className="mb-3 text-xs text-muted-foreground">{card.reason}</p>}

      {card.runtime_note && (
        <Alert tone="warn" className="mb-3">
          {card.runtime_note}
        </Alert>
      )}

      {alreadyExists && (
        <Alert tone="warn" className="mb-3">
          <strong>{defaultStoreName(card.id)}</strong> is already in the model store. Continuing
          will reuse it; no files will be downloaded or replaced.
        </Alert>
      )}

      <div className="space-y-3">
        <Field
          label="Family"
          hint={
            tierOf(card) === "override"
              ? "No confirmed family for this architecture — pick one to try."
              : "Suggested from the architecture — override if needed."
          }
        >
          <Select value={family} onChange={(e) => setFamily(e.target.value)}>
            {!family && <option value="">Select a family…</option>}
            <FamilyOptions families={families} />
          </Select>
        </Field>
        <Field label="Quantization" hint="Optional — a preference; the server falls back per repo.">
          <TextInput
            value={quant}
            onChange={(e) => setQuant(e.target.value)}
            placeholder="e.g. Q4_K_M"
          />
        </Field>
      </div>
    </Dialog>
  );
}
