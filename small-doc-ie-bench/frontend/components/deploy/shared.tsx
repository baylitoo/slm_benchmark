"use client";

import { type ModelFamily } from "@/lib/api";
import { toUserMessage } from "@/lib/errors";
import { type BadgeTone } from "../ui";

export const POLL_MS = 4000;
export const PAGE_SIZE = 10;

// ---------------------------------------------------------------------------
// Deployment lifecycle → badge tone.
// ---------------------------------------------------------------------------

export function stateTone(state?: string | null): BadgeTone {
  switch ((state ?? "").toLowerCase()) {
    case "ready":
    case "running":
    case "serving":
      return "ok";
    case "failed":
    case "error":
      return "err";
    case "starting":
    case "downloading":
    case "degraded":
      return "warn";
    default:
      return "neutral";
  }
}

// Server-derived failure kind → short badge label. Mirrors
// docie_bench.serving.failure.FAILURE_LABELS; unknown kinds pass through so a
// new backend category still renders (as its raw slug) with no UI change.
const FAILURE_LABELS: Record<string, string> = {
  oom: "OOM",
  "insufficient-memory": "Won't fit",
  "port-conflict": "Port in use",
  "spawn-error": "Won't start",
  crashed: "Crashed",
  unhealthy: "Unhealthy",
};

export function failureLabel(kind?: string | null): string | null {
  if (!kind) return null;
  return FAILURE_LABELS[kind] ?? kind;
}

export function failureTone(kind?: string | null): BadgeTone {
  if (!kind) return "neutral";
  return kind === "unhealthy" ? "warn" : "err";
}

export function errText(err: unknown, fallback: string): string {
  return toUserMessage(err, { fallback });
}

// The generic capability tag for a family, derived purely from its flags — so
// any family the backend adds reads sensibly with no per-name UI change.
function familyTypeTag(f: ModelFamily): string {
  if (f.analyzer) return "encoder / analyzer";
  if (f.embedding) return "embedding · vectors (RAG)";
  // Transformers/AutoModel is the last-resort path (no GGUF) — label it as such
  // before the generic vision/chat tags so a manual pick reads honestly.
  if (f.transformers_runtime)
    return f.trust_remote_code
      ? "transformers · last resort · trusts remote code"
      : "transformers · last resort (~2-3x RAM)";
  if (f.vision || f.needs_mmproj) return "vision";
  return "chat / extraction";
}

function familyOptionLabel(f: ModelFamily): string {
  return `${f.name} — ${familyTypeTag(f)}`;
}

/** Family <option> list from the families API, with descriptive labels. */
export function FamilyOptions({ families }: { families: ModelFamily[] | null }) {
  if (!families || families.length === 0) {
    return <option value="openai_chat">openai_chat — chat / extraction</option>;
  }
  return (
    <>
      {families.map((f) => (
        <option key={f.name} value={f.name}>
          {f.name} · {familyOptionLabel(f)}
        </option>
      ))}
    </>
  );
}

function familyOptionsOf(families: ModelFamily[] | null): string[] {
  return families && families.length > 0 ? families.map((f) => f.name) : ["openai_chat"];
}
