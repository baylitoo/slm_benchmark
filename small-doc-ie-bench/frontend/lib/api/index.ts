// Typed client for the DocIE Studio backend.
//
// Every call is tolerant of endpoints that don't exist yet: a 404/501 is
// surfaced as a structured `ApiUnavailable` so the UI can render a friendly
// "coming soon" state instead of crashing.
//
// Split from one ~1950-line module into this package for maintainability --
// pure move, no behavior changed. One submodule per domain:
//
//   - core.ts           -- error classes, the request() fetch wrapper, and
//                           the low-level helpers (readBody/detailOf/
//                           unauthorizedError) call sites that bypass
//                           request() (downloadArtifact, the OpenAI-shaped
//                           chat calls) still need for consistent error
//                           formatting.
//   - extract.ts         -- extraction trigger, realtime token, run-status
//                            polling fallback, document rasterization.
//   - serving.ts          -- model manifests, runtimes, deployments, ports,
//                            deploy trigger, store catalog, family contracts.
//   - lifecycle.ts        -- load/unload/pin/delete/repair + log tail.
//   - sizing.ts           -- node RAM snapshot, fit table, what-if, scale.
//   - observability.ts    -- review-metrics/OCR-cache/activity tiles.
//   - datasets.ts         -- registered-dataset listing + validation.
//   - modelProfiles.ts    -- configs/models.yaml profile CRUD.
//   - benchmark.ts        -- benchmark trigger, run index, artifacts,
//                            comparisons.
//   - schemas.ts          -- static schema listing + named dynamic-schema
//                            and routing-policy registries.
//   - batch.ts            -- batch extraction (N docs, one durable job).
//   - seeding.ts          -- Ollama/HF seeding, HF Hub search/inspect,
//                            catalog browser.
//   - review.ts           -- human review queue workflow.
//   - agents.ts           -- agent CRUD + OpenAI-compatible chat calls.
//   - helpers.ts          -- deployment-derived classification, embed/
//                            rerank, formatBytes.
//   - browser.ts          -- browser-only helpers with no server round-trip.
//
// Every name that lived at this module's top level before the split is
// re-exported below, so nothing outside `lib/api/` needs to know it was
// ever one file -- `import { X } from "@/lib/api"` resolves to this barrel
// unchanged for every existing call site.

export * from "./core";
export * from "./extract";
export * from "./serving";
export * from "./lifecycle";
export * from "./sizing";
export * from "./observability";
export * from "./datasets";
export * from "./modelProfiles";
export * from "./benchmark";
export * from "./schemas";
export * from "./seeding";
export * from "./batch";
export * from "./review";
export * from "./agents";
export * from "./helpers";
export * from "./browser";
