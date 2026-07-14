# DocIE — Server Architecture

How the framework runs on a server: what each container does, where state
lives, how an extraction travels through the system, and how the serving
control plane keeps records, processes, and RAM in sync.

Deep dives: [serving-control-plane.md](serving-control-plane.md) (reconciler /
resources / sizing / lifecycle design), [docie-studio.md](docie-studio.md)
(Studio jobs & artifacts), [serving-factory.md](serving-factory.md) (model
store & families), [THREAT_MODEL.md](THREAT_MODEL.md).

---

## 1. Big picture

DocIE is a **dataset-, solution-, and runtime-agnostic document
information-extraction platform**: a benchmark harness and a serving layer
behind one OpenAI-compatible surface, managed from a web control board
(DocIE Studio).

```
                       browser (localhost:3000)
                               │
                        ┌──────▼──────┐
                        │     web     │  Next.js Studio UI (LiteLLM-style board:
                        │             │  Playground / Deploy / Sizing / Benchmark / Obs)
                        └──────┬──────┘
                        REST   │  + realtime tokens
                        ┌──────▼──────┐         ┌───────────┐
                        │     api     │────────▶│  inngest   │  event bus + run store
                        │  (FastAPI)  │  events │ (self-host)│  (dashboard :8288)
                        └──────┬──────┘         └─────┬─────┘
              reads observed   │                      │ delivers by registered app
              state, fires     │            ┌─────────┴──────────┐
              lifecycle events │            │                    │
                        ┌──────▼──────┐  ┌──▼───────────┐  ┌─────▼─────┐
                        │  postgres   │  │   serving    │  │  worker   │
                        │ catalog +   │  │ (1 replica)  │  │ (scale N) │
                        │ observed +  │  │ owns model   │  │ extract / │
                        │ studio runs │  │ processes +  │  │ benchmark │
                        └─────────────┘  │ reconciler   │  │ jobs      │
                                         └──────┬───────┘  └─────┬─────┘
                                                │ spawns          │ HTTP to the
                                         ┌──────▼───────┐        │ advertised
                                         │ llama-server │◀───────┘ endpoint
                                         │ (one per     │  http://serving:<port>/v1
                                         │  deployment) │
                                         └──────────────┘
```

Supporting services: `redis` (Inngest run state), and optional compose
profiles — `bench` (CLI-in-a-container), `llm-llamacpp` (standalone GGUF
server), `prometheus` + `grafana` (observability).

## 2. The containers and what they own

| Service | Replicas | Role | Can it touch model processes? |
|---|---|---|---|
| **api** | 1 | REST surface (`/v1/extract/*`, `/v1/serving/*`, `/v1/studio/*`), auth, rate limits, fires Inngest events, serves artifacts | No — reads shared state only |
| **serving** | **exactly 1** | The control plane. Registers the lifecycle Inngest app (`docie-serving`: deploy / delete / load / unload / seed). Spawns and supervises `llama-server` processes, runs the **reconciler** loop, allocates ports, publishes observed state | **Yes — the only one** |
| **worker** | scale freely | Registers the workload app (`docie-studio`: extract, benchmark, GC). Calls deployed models **over HTTP**; never spawns a runtime | No |
| **web** | 1 | Next.js Studio board | No |
| **inngest** | 1 | Self-hosted event bus; durable runs, retries, realtime channels | — |
| **postgres** | 1 | Model catalog, placements (observed state), node snapshot, Studio run/artifact index | — |
| **redis** | 1 | Inngest queue/run state | — |

Two design rules fall out of container isolation:

1. **Only the `serving` container can see or signal model processes** (they are
   its children, in its PID namespace). Everything the api/web show about a
   deployment therefore comes from *published observed state*, not from
   probing.
2. **Event routing is by Inngest app registration**, not service name:
   lifecycle functions exist only in the `serving` app, workloads only in the
   `worker` app, so a `serving/*` event can never land on a scaled worker.
   `serving` must stay single-replica (a lease file makes a second reconciler
   refuse to start).

## 3. Where state lives (three homes, one writer each)

| State | Home | Writer | Consumed by |
|---|---|---|---|
| **Desired** — what should run, plus lifecycle controls (`activation`, `pinned`, `last_served`) | `deployments.json` on the shared `serving-state` volume | `serving` only | reconciler, resolver (DB-optional routing) |
| **Observed** — what *is* running: `phase` (hot/loading/cold/evicted/failed), pid + create_time, RSS, health, last_error | Postgres `model_placement` | reconciler only | api, Studio board |
| **Resources** — node total/free RAM, per-process RSS, reclaimable | Postgres `serving_node` (single row, refreshed each cycle) | reconciler only | Sizing engine/tab |

Other shared volumes: `artifact-store` (benchmark reports/predictions,
written by workers, served back by the api by id), `ocr-cache`
(liteparse/OCR artifacts), and the model store
(`<serving-home>/models/<name>/model.gguf` + optional `mmproj.gguf` +
`index.json` — see [serving-factory.md](serving-factory.md)).

Postgres is **not** required for routing: the resolver reads the on-disk
store index and `deployments.json` first, so extraction keeps working with no
`DATABASE_URL` (the board's observed surface is what degrades).

## 4. The serving control plane

The staleness killer. A background loop in the `serving` container
(~10s cycle, single writer, guarded by a lease) that reconciles records
with reality:

- **Liveness by `/health`**, not pid — with a pid `create_time` guard against
  PID reuse. A deployment that reached `READY` and stops answering is declared
  dead within 2–3 misses (fast death); a still-loading GGUF keeps a long
  tolerance (slow load). Dead ≠ frozen `ready` ever again.
- **Gated restarts** — a crashed deployment is only respawned if restart
  budget remains **and** a live RAM fit-check passes (no OOM crash-loops);
  the budget is forgiven after a sustained healthy streak.
- **Publishing** — each cycle upserts observed placements + the node snapshot
  to Postgres. The board reads *that*, so what you see is what the node runs.
- **Real delete** — `DELETE /v1/serving/deployments/{name}` kills the process,
  removes the record, frees the port, clears the placement row.

**Resources & sizing.** The tracker reads cgroup-v2 first
(`memory.max`/`memory.current`, minus reclaimable page cache so unloaded
mmap'd GGUFs don't count as used), falling back to VM stats with an explicit
`source` flag. Each model's footprint = `max(calibrated steady-state RSS,
predicted)` where predicted = GGUF size + KV cache(ctx, n_parallel) +
overhead, mmproj included for vision. The **Sizing tab** turns that into
"N more instances of X fit now" with a safety margin, a capacity bar, and
what-if planning — same math as the admission gate.

**Dynamic load/unload.** Deployments have phases
`hot → (idle TTL) → evicted/cold → (on demand) → loading → hot`. Idle models
are unloaded (process killed, record + port kept). A request to a cold
*managed* deployment triggers a reload and **waits** (one load per deployment
under concurrency — no thundering herd, no 502; cold-start TTFT is the
accepted price). Under memory pressure the reconciler evicts LRU unpinned
deployments with storm guards (min-hot-time, per-cycle rate limit,
fit-before-evict). Pinned deployments are never evicted; manually stopped
ones are never auto-reloaded.

## 5. Life of an extraction

1. Browser → Playground: pick a **deployment** (live list), upload a PDF.
2. `web → api`: `POST /v1/studio/extract` → api fires `doc/extract.requested`
   (Inngest) and returns a realtime channel.
3. A **worker** picks up the event. The resolver maps the deployment name →
   the record's advertised endpoint (`http://serving:<port>/v1`) and the
   family's profile (template delivery, response-format style, vision flag,
   generation params). Cold managed deployment → fire load, wait.
4. **Ingestion** ([liteparse](https://github.com/run-llama/liteparse), PDFium):
   text-only models get the spatial text layer (OCR fallback for scans);
   vision models get rasterized page images (`screenshot()`); NuExtract3/
   LFM2.5-VL take the image path with their mmproj projector loaded.
5. **Model call** over the OpenAI-compatible endpoint with the negotiated
   structured-output style (`json_schema → json_object → none+repair` ladder,
   probed per runtime) — the worker stamps `last_served` for the lifecycle.
6. **Post-processing**: schema validation, normalization, evidence grounding
   against OCR spans (hallucination accounting), usage capture.
7. Result streams back over the realtime channel (polling fallback); audit
   row persisted.

Benchmarks follow the same path per document, then write `report.html` /
`predictions.jsonl` to the `artifact-store` volume with a Postgres index row —
served back by the api regardless of which worker produced them.

## 6. Memory ceilings on Docker Desktop (read before trusting any number)

The RAM the tracker reports is the **lowest applicable ceiling**, not the
host's DIMMs. On Windows/Docker Desktop there are three stacked layers:

```
host RAM (e.g. 16 GB)
  └── WSL2 VM         — default = 50% of host  → set %UserProfile%\.wslconfig:
                         [wsl2] memory=12GB, then `wsl --shutdown`
        └── serving container mem_limit — default 8g → DOCIE_SERVING_MEM_LIMIT in .env
              └── what the tracker reports (source=cgroup)
```

A 16 GB machine with defaults shows **8 GB total** — that is *correct* for
what the serving node can actually use. Raise both layers to give it more.
`source=vm` in the snapshot means no cgroup limit was set and numbers are
soft (the VM's view). On a bare-metal Linux server none of this applies:
set `DOCIE_SERVING_MEM_LIMIT` (or run unlimited) and cgroup numbers are exact.

## 7. Running it

```bash
cp .env.example .env        # set INNGEST keys; AUTH_REQUIRED/API_KEYS for anything networked
docker compose up -d --build
# Studio → http://localhost:3000   Inngest dashboard → http://localhost:8288
# +observability: docker compose --profile observability up -d   (Grafana :3001)
```

Model onboarding: **seed** (Deploy → Add model: from a local Ollama pull or a
GGUF path, picking a *family* — the family contract decides `--jinja`,
mmproj, template delivery, generation defaults) → **deploy** (port
auto-allocated, admission fit-checked) → it appears in the Playground
deployment picker and the Sizing tab.

Scale rule of thumb: `docker compose up -d --scale worker=3` is safe
(workers are stateless HTTP callers); **never scale `serving`** (the lease
will refuse the second reconciler, but don't rely on it).

### Ops quick checks

```bash
curl -s localhost:8080/v1/serving/deployments | jq   # observed state (phase, RSS, errors)
curl -s localhost:8080/v1/serving/resources   | jq   # node RAM + source flag
curl -s localhost:8080/v1/serving/sizing      | jq   # what fits now
docker compose logs -f serving                        # reconciler cycles + runtime stderr tails
```

Dead runtime → board flips `failed` within ~2 cycles. Delete → row and port
gone. If numbers look wrong, check §6 before filing a bug.
