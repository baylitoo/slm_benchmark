# DocIE Studio

A litellm-style web app + async backend on top of the benchmark framework. One
`docker compose up` brings up the whole stack; the framework's core operations
(document extraction, benchmark runs, model deploy) run as durable
[Inngest](https://www.inngest.com/) functions on a background **worker**, and a
Next.js UI lets you drive and observe them.

It is **additive**: the existing CLI, dashboards, and the hand-rolled
`orchestrator/` queue are untouched. Inngest is a *new* event-driven path.

## Architecture

```
 browser ──► web (Next.js :3000) ──► api (FastAPI :8080) ──inngest.send(event)──► inngest server (:8288 / :8289)
                  ▲  realtime hook            │  /v1/studio/*                              │ dispatch (Connect WS)
                  └──────────── realtime ◄────┴── token / runs proxy                       ▼
                                                                                   worker (docie-worker)
                                                                                   runs the functions, calls
                                                                                   ExtractionService / run_benchmark
```

- **worker** holds the Inngest functions and dials OUT to the Connect gateway
  (`:8289`) — no inbound HTTP, no public reachability needed. Scale with
  `docker compose up -d --scale worker=3`.
- **api** only *sends* events and proxies status/realtime tokens back to the UI.
- **inngest** server reuses the project Postgres (a dedicated `inngest` DB, see
  `infra/postgres/init.sql`) plus a small Redis for run state.

| Service | Port | Purpose |
|---|---|---|
| `web` | 3000 | DocIE Studio UI |
| `api` | 8080 | FastAPI: `/v1/studio/*`, extraction, benchmark, reviews |
| `inngest` | 8288 / 8289 | Dashboard+API / Connect gateway |
| `grafana` | 3001 | Dashboards (observability profile) |
| `prometheus` | 9090 | Metrics (observability profile) |
| `postgres` | 5432 | App DB + `inngest` DB |
| `redis` | — | Inngest run state |

## One-command setup

```bash
cp .env.example .env
# In .env set two hex keys (even length): `openssl rand -hex 32`
#   INNGEST_EVENT_KEY=...
#   INNGEST_SIGNING_KEY=...
#   INNGEST_DEV=0
# And point OPENAI_COMPAT_BASE_URL at an LLM reachable from the worker container.

make studio            # == docker compose up -d --build postgres redis inngest worker api web
```

Then open:
- **Studio UI** → http://localhost:3000
- **Inngest dashboard** → http://localhost:8288 (the `worker` shows under *Apps*)

> **Gotcha:** `infra/postgres/init.sql` only runs on a *fresh* Postgres volume.
> If you already have a `postgres-data` volume, create the DB once:
> `docker compose exec postgres psql -U docie -d docie -c "CREATE DATABASE inngest;"`

## The tabs

1. **Playground** — paste text or upload a file (PDF/image, sent as base64),
   pick a schema + model profile, run an extraction, and watch the result stream
   in live (realtime hook, with polling fallback). Fully wired.
2. **Deploy** — litellm-style model/runtime/deployment tables + a one-click
   deploy action. *(Backend `/v1/serving/*` + `model/deploy.requested` in
   progress; the UI degrades gracefully until they land.)*
3. **Benchmark** — trigger a dataset benchmark run and view metrics. Trigger is
   wired (`POST /v1/studio/benchmark`); the results list depends on
   `/v1/serving/benchmarks`.
4. **Observability** — embeds Grafana (`:3001`) and links the Inngest dashboard
   and raw Prometheus metrics (`/metrics`).

## API contract (`/v1/studio`)

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/studio/extract` | fire `doc/extract.requested`; returns `{event_ids, channel, topics}` |
| POST | `/v1/studio/benchmark` | fire `benchmark/run.requested` (needs `dataset`) |
| GET | `/v1/studio/realtime-token?channel=&topics=` | mint a realtime subscription token (501 if unavailable) |
| GET | `/v1/studio/runs/{event_id}` | polling fallback: proxy Inngest run status |

Events carry JSON; documents travel as `content_b64` (base64 bytes) or raw
`text`. Functions publish best-effort realtime topics (`status`, `progress`,
`result`, `error`) on the per-request `channel`.

## Local dev loop (no Docker)

Fastest way to iterate on functions / confirm the worker connects:

```bash
pip install -e .
npx inngest-cli@latest dev          # terminal A: in-memory dev server (:8288)
make worker-dev                     # terminal B: INNGEST_DEV=1 docie-worker
# terminal C: fire an event straight at the dev server
curl -s -X POST http://localhost:8288/e/dev_key -H "Content-Type: application/json" \
  -d '{"name":"doc/extract.requested","data":{"text":"INVOICE\nTotal: 120.00 EUR","schema_name":"invoice"}}'
```

The worker should log a connection and appear under *Apps* at
http://localhost:8288. A *successful* extraction also needs `OPENAI_COMPAT_*`
pointing at a running LLM.

Frontend dev: `cd frontend && npm install && npm run dev` (see `frontend/README.md`).

## Smoke test (stack up)

```bash
make studio-smoke   # POST /v1/studio/extract, prints {event_ids, channel, topics}
# then, with the returned event id:
curl -s http://localhost:8080/v1/studio/runs/<EVENT_ID> | jq .
```

## Troubleshooting

- **Worker stuck `Reconnecting`** — don't override the gateway in dev; let the
  SDK discover `:8289`. In Docker, `INNGEST_CONNECT_GATEWAY_URL=ws://inngest:8289/v0/connect`.
- **`ImportError` on `inngest.connect` / `inngest.experimental.realtime`** — bump
  the `inngest` package; the realtime token route 501s and the UI falls back to
  polling, so the core loop still works.
- **`In cloud mode but no signing key`** — set `INNGEST_DEV=1` for local dev, or
  provide `INNGEST_SIGNING_KEY`/`INNGEST_EVENT_KEY` for the self-hosted server.
- **Port 3000 clash** — the studio UI owns 3000; Grafana was moved to 3001.
  Override with `STUDIO_PORT` / `GRAFANA_PORT`.

## Make targets

`make studio` · `make studio-down` · `make studio-logs` · `make studio-smoke` · `make worker-dev`
