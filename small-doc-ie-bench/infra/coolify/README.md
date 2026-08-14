# Deploying to Coolify

This deploys the full stack (`docker-compose.yml`) as a Coolify "Docker
Compose" resource, with `docker-compose.override.yml` in this folder
narrowing what's reachable from the public internet:

- **Public:** `web` (Studio UI), `api` (OpenAI-compatible + Studio API)
- **Internal-only** (compose network, unreachable from outside): `postgres`,
  `inngest`, `redis`, `serving`, `prometheus`, `grafana`

`serving` was already internal-only in the base compose file (it's the sole
holder of the unauth llama-server processes) — nothing to change there.

## 1. Domains: use Coolify's auto-assigned ones

No custom domain/DNS needed. With no wildcard domain configured on the
server, Coolify generates a free `*.sslip.io` domain per resource from the
server's IP the moment you assign a domain to a service in the UI — that's
what we're using here, for both `api` and `web`.

**Order matters**: `api`'s domain has to exist before `web` is *built*, not
just before it's deployed — `NEXT_PUBLIC_API_BASE` gets baked into the web
image at build time (see §3), so a build that runs before `api` has an
assigned domain bakes in the wrong (or missing) value and needs a rebuild to
fix. So: add the compose resource, assign `api`'s domain first, copy the
generated `https://<something>.sslip.io` URL from the UI, *then* set
`NEXT_PUBLIC_API_BASE` to that value before the first build of `web`.

## 2. Create the Coolify resource

1. New Resource → Docker Compose → point at this repo (`baylitoo/slm_benchmark`), the branch you want to deploy, and `docker-compose.yml` at the repo root.
2. In the Compose settings, add `infra/coolify/docker-compose.override.yml` as an additional compose file (Coolify supports multiple compose files per resource) — this is what clears the `ports:` on postgres/inngest/prometheus/grafana/api/web so Coolify's proxy fronts api/web instead of the raw host port.
3. Assign `api` a domain first (Coolify generates the `*.sslip.io` URL) and copy it — you need this value for §3 before building `web`. Then assign `web` its own domain the same way. Leave `postgres`/`inngest`/`prometheus`/`grafana` with no domain — that's what keeps them private.
4. **Do not** enable the `local-llm` or `tools` compose profiles — production model serving goes through the `serving` control plane (deploy models via the Studio UI / `POST /v1/serving/deploy`), not the static `llm-llamacpp` container.

## 3. Environment variables

Set these in Coolify's environment editor for the app (not committed — this
repo's `.env` stays out of git). Generate secrets yourself, don't reuse the
`.env.example` local-dev defaults on a public box:

```bash
# run these locally or on the server, paste the output into Coolify
openssl rand -hex 32   # -> INNGEST_EVENT_KEY
openssl rand -hex 32   # -> INNGEST_SIGNING_KEY
openssl rand -hex 24   # -> POSTGRES_PASSWORD (also update DATABASE_URL to match)
```

| Variable | Value | Why |
|---|---|---|
| `AUTH_REQUIRED` | `false` | The Studio web UI has no login flow and never sends an API key on its own requests — turning this on 401s the UI itself. Anonymous callers are still rate-limited (`ANONYMOUS_RATE_LIMIT_REQUESTS`, default 600/60s per IP, `ANONYMOUS_MAX_CONCURRENT_REQUESTS`, default 16). Revisit if this becomes a real multi-tenant API. |
| `STUDIO_CORS_ORIGINS` | `web`'s generated `https://<something>.sslip.io` URL | Unset defaults to localhost-only — the deployed Studio's browser calls to `api`'s domain get CORS-blocked without this. |
| `DATABASE_URL` | `postgresql+psycopg://docie:<your-postgres-password>@postgres:5432/docie` | Internal service name, not a public host. |
| `POSTGRES_PASSWORD` | matches the password in `DATABASE_URL` above | docker-compose.yml's `postgres` service env. |
| `INNGEST_EVENT_KEY` / `INNGEST_SIGNING_KEY` | generated above | Required — the inngest service `${VAR:?}`-fails to start without them. |
| `INNGEST_DEV` | `0` | Production self-hosted mode (this is already the compose default). |
| `NEXT_PUBLIC_API_BASE` | `api`'s generated `https://<something>.sslip.io` URL (see §1 — assign `api`'s domain BEFORE this build) | **Baked into the web image at build time.** This is the #1 thing people miss — leaving the `http://localhost:8080` default means every visitor's browser tries to call their own machine. Wrong value = rebuild required to fix, not just redeploy. |
| `NEXT_PUBLIC_GRAFANA_URL`, `NEXT_PUBLIC_INNGEST_URL` | leave default | These are Observability-tab quick links only. Grafana/Inngest aren't public in this setup, so the links won't resolve — cosmetic, not a functional break. Set them (and give those services a domain) later if you decide to expose observability too. |
| `NEXT_PUBLIC_INNGEST_BASE_URL` | leave default | Same reasoning — Studio's realtime progress bars (seed/deploy/benchmark) silently fall back to polling when this can't reach a public Inngest server. Still works, just coarser-grained progress updates. |

Everything else in `.env.example` (schema/model defaults, OCR cache, sizing
margin, idle-TTL) keeps its documented default unless you have a specific
reason to change it — see the comments in that file.

## 4. Size for the actual host, and remember it's shared

This machine already runs a fair number of other Coolify apps (livekit,
rabbitmq, etc.) — check `free -h` and `df -h` on the box, and give this stack
headroom, not the whole machine:

- `DOCIE_SERVING_MEM_LIMIT` (default `8g`): the cap the `serving` container's
  cgroup enforces on GGUF runtimes (weights + KV cache + ~0.5GB slab per
  deployment). This is a plain Linux host, not Docker Desktop/WSL2, so the
  cgroup limit is authoritative directly — size it to what you're actually
  willing to dedicate to model serving, leaving enough for postgres/api/
  worker/grafana/prometheus/inngest and everything else already running.
- `LLAMA_CPP_N_THREADS` (default `12`): run `nproc` on the box and don't
  oversubscribe past the real core count, especially on a shared host.
- No GPU on this machine, per your setup — nothing else to configure there;
  llama-server runs CPU-only.

## 5. Deploy and smoke-test

After Coolify builds and starts the stack, using the two `*.sslip.io` URLs
from §1:

```bash
curl -s https://<api-generated-domain>/healthz
curl -s https://<api-generated-domain>/v1/models
```

Then open `https://<web-generated-domain>` and confirm the Studio loads and
can reach the API (Observability tab should show live data, not "couldn't
load" errors).

## Known limitations of this setup (by design, not bugs)

- API is unauthenticated (rate-limited only) — see §3.
- Grafana, Prometheus, and the Inngest dashboard are not publicly reachable —
  SSH-tunnel or use Coolify's own terminal/log viewer if you need to look at
  them (`docker compose exec` / Coolify's UI logs).
- Studio realtime progress bars degrade to polling (no public Inngest
  WebSocket endpoint to subscribe to).
