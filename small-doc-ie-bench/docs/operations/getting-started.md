# Getting started

One `docker compose up` brings up the whole platform on one machine. This page
gets it running, serves a first model, and covers the checks for when it does
not behave.

## Bring the stack up

```bash
cp .env.example .env
```

Three things in `.env` have no usable default:

| Variable | |
|---|---|
| `POSTGRES_PASSWORD` | required; `DATABASE_URL` references it |
| `INNGEST_EVENT_KEY`, `INNGEST_SIGNING_KEY` | two even-length hex strings, e.g. `openssl rand -hex 32` |
| `INNGEST_DEV` | `0` for the self-hosted server |

Then:

```bash
docker compose up -d --build
```

| Service | Port | |
|---|---|---|
| `web` | 3000 | the Studio |
| `api` | 8080 | the HTTP API |
| `inngest` | 8288 / 8289 | dashboard and API / Connect gateway |
| `postgres` | 5432, loopback only | application and Inngest databases |
| `grafana`, `prometheus` | 3001, 9090, loopback only | `--profile observability` |

`serving`, `worker` and `redis` publish nothing. Open the Studio at
`http://localhost:3000`; the Inngest dashboard at `http://localhost:8288` should
list both apps, `docie-studio` and `docie-serving`, under *Apps*.

```{warning}
`make studio` starts `postgres redis inngest worker api web` and **not**
`serving`. A stack started that way has nowhere to deploy a model.
```

Postgres, Prometheus and Grafana bind to `127.0.0.1` because Postgres holds
real credentials, Prometheus has no authentication, and Grafana allows
anonymous viewing for the Studio's embedded dashboard. The end of
`.env.example` explains how to publish one of them further if you need to.

### An existing Postgres volume

`infra/postgres/init.sql` and `POSTGRES_PASSWORD` take effect only when the
volume is first initialised. On an older volume, create the Inngest database
and set the password in place:

```bash
docker compose exec postgres psql -U docie -d docie -c "CREATE DATABASE inngest;"
docker compose exec postgres psql -U docie -d docie -c "ALTER USER docie WITH PASSWORD '<password>';"
```

## Serve a first model

Everything here happens in the Studio; {doc}`studio` covers each section.

1. **Serving → Models → Add model**, Hugging Face tab. Inspect
   `LiquidAI/LFM2.5-350M-GGUF`, pick the `Q4_K_M` quant and family `lfm2`, then
   download and seed it.
2. **Serving → Deployments → Deploy a model**, pick it, leave the runtime on
   Auto. The row moves from `loading` to `hot` once a health probe passes.
3. **Playground → Extract**, paste
   `data/sample_dataset/files/invoice-001.txt`, schema `invoice`, your
   deployment. Structured JSON comes back in about ten seconds on a laptop CPU.

## Check it

```bash
curl -s localhost:8080/v1/serving/deployments | jq   # phase, RSS, last error
curl -s localhost:8080/v1/serving/resources   | jq   # node memory and its source
curl -s localhost:8080/v1/serving/sizing      | jq   # what fits now
docker compose logs -f serving                        # reconciler cycles, runtime stderr
```

Add `-H "X-API-Key: ..."` once authentication is on. A dead runtime shows
`failed` within a couple of reconcile cycles. If the memory figures look wrong,
read the *Memory* section of {doc}`../architecture/serving` before anything
else.

## Scaling on one machine

`docker compose up -d --scale worker=3` is safe: workers are stateless and
reach models over HTTP. **Never scale `serving`.** It holds the model processes
and the desired-state file, and a second one refuses to start rather than fight
the first. More serving capacity means another node — see
{doc}`fleet-runbook`.

## Developing without Docker

To work on job functions, run Inngest's in-memory dev server and a local
worker:

```bash
pip install -e .
npx inngest-cli@latest dev      # terminal A, :8288
make worker-dev                 # terminal B, INNGEST_DEV=1 docie-worker
curl -s -X POST http://localhost:8288/e/dev_key -H "Content-Type: application/json" \
  -d '{"name":"doc/extract.requested","data":{"text":"INVOICE\nTotal: 120.00 EUR","schema_name":"invoice"}}'
```

The worker should appear under *Apps*. A successful extraction also needs
`OPENAI_COMPAT_BASE_URL` pointing at a running model. For the frontend,
`cd frontend && npm install && npm run dev`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Worker stuck `Reconnecting` | In Docker, `INNGEST_CONNECT_GATEWAY_URL=ws://inngest:8289/v0/connect`. In local dev, leave it unset and let the SDK discover the gateway |
| `In cloud mode but no signing key` | `INNGEST_DEV=1` locally, or set both Inngest keys for the self-hosted server |
| Realtime token route answers `501` | the installed `inngest` package lacks realtime; the Studio falls back to polling |
| Port 3000 or 3001 taken | `STUDIO_PORT`, `GRAFANA_PORT` |
| Deployment `failed` | its logs, from the row in **Deployments**, have the reason; **Repair** redeploys on a fresh port and resets the restart budget |
| Deployment `evicted` | idle unload or memory pressure; the next request reloads it |
