# Changelog

All notable changes to this project are documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
grouped by capability area rather than by commit — this is a first `1.0.0`
release for a project that grew incrementally without prior version tags, so
this entry summarizes the full feature set as it stands today rather than a
diff against a previous release.

## [1.0.0]

### Benchmark & evaluation core

- CPU-only benchmark harness for structured document extraction with small
  LLMs: reproducible, resumable runs with full manifest/environment capture,
  dependency and git-state snapshots, and deterministic regression comparison
  against a baseline with statistical deltas, confidence intervals, sign
  tests, root-cause attribution, and budget/judge-calibration verdicts.
- Versioned dataset registry with integrity/duplicate/leakage validation,
  label-provenance tracking (asserted vs. derived fields), and declarative
  ground-truth mapping.
- Dynamic, typed extraction schemas (string/date/number/money/list/object)
  compiled into working pydantic models and NuExtract templates at request
  time, plus a persistence layer for saving and reusing named schemas.
- Evidence-grounding metrics, an LLM-judge evaluation mode with calibration
  gating against human labels, and honest hallucination/containment
  semantics.
- Multi-stage routing pipeline (fallback/escalation across model profiles by
  configurable rule + budget), with a named, persisted routing-policy
  registry as the discoverable alternative to a raw config file path.
- Human review and active-learning workflow: a priority-ranked queue,
  optimistic-concurrency corrections, arithmetic-mismatch suggestions, and a
  durable event/audit trail.
- OCR lab with a persistent, content-addressed cache shared across runs.

### Model serving platform

- Unified OpenAI-compatible gateway in front of local model runtimes
  (llama.cpp/Ollama-style GGUF serving, a transformers/AutoModel last-resort
  runtime for checkpoints with no GGUF, and an encoder runtime for
  analyzer/guardrail models), plus embedding and reranker endpoints.
- A canonical GGUF model store with per-family capability contracts, seeding
  directly from the Hugging Face Hub (live progress, resumable downloads) or
  a local Ollama install, and Hub browse-and-deploy with a pre-flight
  architecture → family support verdict before any download.
- A durable control plane: a reconciler publishing observed placement state
  (phase, RSS, measured throughput/TTFT — always measured, never estimated),
  dynamic load/unload lifecycle with idle-TTL eviction and auto-reload on
  request, per-deployment failure classification (OOM, port conflict, crash,
  ...), in-place repair (port reallocation without delete/recreate), and
  scaling a store model to N load-balanced, addressable replicas.
- A capacity/sizing engine: a live RAM fit-table per store model, a
  hypothetical-mix "what-if" planner, and node RAM/reclaimable-memory
  snapshots.
- Vision model support end to end (page rasterization, vision-routed
  extraction profiles, a VLM-as-OCR pipeline stage).

### Agents platform

- Preconfigured agents over the OpenAI-compatible surface: a PII security
  proxy, an OCR agent, and a staged document-extraction builder
  (OCR / OCR→LLM / vision→structured), each backed by any deployed model or
  models.yaml profile.
- GLiNER2-based guardrails (PII detection + safety moderation) as a
  one-checkpoint deployable analyzer, with a live PII report on every
  proxied request.

### DocIE Studio (web UI)

- A Next.js control surface covering Playground (chat/vision/extraction
  against any deployment), Deploy (catalog browse-to-seed, model store,
  deployments with live phase/throughput, sizing, and now a Downloads tab
  tracking seed jobs durably with live progress and failure logs), Agents
  (templates, instances, create/edit in place), Benchmark (run + results,
  with a Comparison Lab surfacing the regression-comparison engine's deltas
  and budget verdicts), Review (the human-review queue), and Observability
  (activity, OCR-cache, and review-queue tiles alongside the metrics
  dashboard).
- Authoring model profiles (pipeline and OCR-only) and dataset validation
  directly from the Studio, without hand-editing server-side config files.

### Security & operations

- Multi-tenant API-key authentication (`AUTH_REQUIRED=true`) with per-tenant
  request quotas, hardened against the LAN-exposure gaps of an early
  single-operator default (no baked-in Postgres/observability credentials,
  loopback-bound admin ports, authenticated artifact downloads).
- Durable, tenant-scoped indexes for benchmark runs, extraction audits, and
  seed-download jobs — reachable from any replica, not tied to worker-local
  filesystem state.

### Codebase

- The Studio's backend API (`studio_api.py`) and frontend API client
  (`lib/api.ts`) were each split from a single large file into a package
  organized by domain, for maintainability — pure moves, no behavior
  changes, each pinned by a test asserting the exact route/export surface is
  unchanged.
