# DocIE Studio (frontend)

A litellm-style web console for the DocIE benchmark backend. Built with
Next.js (App Router, TypeScript) and Tailwind CSS, with a modern SaaS-style
shell: sidebar navigation, cards, dark mode by default with a light toggle,
loading skeletons, empty states, and toast notifications.

Five sections, routed as `/{section}/{view?}` (deep links, refresh and the
browser back button all work). All sections stay mounted across navigation —
the shell lives in an app-router layout — so a running job survives moving
around the app:

1. **Playground** (`/playground`) — extraction, chat, vision and embeddings
   panels over live deployments. Uses Inngest **realtime**
   (`@inngest/realtime`) when the backend can mint a token, and transparently
   **falls back to polling** `GET /v1/studio/runs/{event_id}` when realtime is
   unavailable.
2. **Serving** (`/deploy/models`, `/deploy/deployments`, `/deploy/ports`,
   `/deploy/sizing`) — the model store, browse-and-deploy from Hugging Face,
   deployments with lifecycle actions (load/unload/pin/repair/scale), port
   administration, and the RAM sizing/what-if tab. Models and deployments
   **auto-refresh** on an interval; see "Serving pages" below.
3. **Agents** (`/agents/catalog`, `/agents/instances`, `/agents/create`) —
   preconfigured agents (security proxy, document extraction) exposed as
   OpenAI-compatible endpoints.
4. **Benchmark** (`/benchmark/run`, `/benchmark/results`) — start a benchmark
   run (a `dataset` is required) and browse past runs + their metrics.
5. **Observability** (`/observability`) — embeds Grafana and links to Inngest
   and raw Prometheus metrics (driven by the env vars below).

The API client treats a **bare** `404`/`501` as "not available on this
backend" and the UI degrades gracefully; a `404` carrying the
`X-Docie-Error` header is a domain answer ("deployment 'x' not found") and
its detail is shown verbatim.

## Serving pages

- **Available models** come from `GET /v1/serving/store` (the on-disk model
  store: GGUFs, encoder and transformers snapshots). If the route isn't
  enabled a friendly "no models in the store yet" / "not available" state is
  shown.
- **Runtime picker** is scoped to the chosen model (its store entry's
  compatible backends). Picking a runtime is optional — "Auto" lets the
  server choose; selecting a backend serves it explicitly.
- **Add model** seeds the store from Hugging Face (search, by-repo, or a
  curated collection), from a local Ollama reference, or as an encoder
  snapshot; progress streams over the returned `channel` (realtime, else
  polling with the seed-progress sidecar).
- **Deploy** posts to `POST /v1/studio/deploy` and streams progress the same
  way. The **Deployments** table (`GET /v1/serving/deployments`) reflects new
  deployments on its next auto-refresh and carries the lifecycle actions.
- **Auto-refresh**: the store and deployments lists poll every ~4s with a
  visible "Live · Xs ago" indicator and a manual refresh button. Polling
  auto-pauses when the browser tab is hidden **and** when the section isn't
  active. One-shot lists (families, agents, runs, schemas) are SWR-cached and
  revalidate when the window regains focus.

## Tech / dependencies

Runtime deps added for the redesign (all small, React 19-compatible):

| Package          | Why                                              |
| ---------------- | ------------------------------------------------ |
| `lucide-react`   | Icon set.                                        |
| `next-themes`    | Dark/light theme toggle (dark by default).       |
| `clsx`           | Conditional class names.                         |
| `tailwind-merge` | Resolve Tailwind class conflicts (`cn()` helper).|
| `swr`            | Cache/dedup/focus-revalidate for one-shot reads. |

Theme tokens are CSS variables (see `app/globals.css`) mapped to semantic
Tailwind colors (`background`, `card`, `border`, `muted`, `accent`, …). Dark
mode uses Tailwind's `class` strategy toggled by `next-themes`.

## Environment variables

All are **public** and inlined into the client bundle **at build time**
(`next build`) — they are *not* read at runtime. See the Docker note below.
**No new env vars were added** for the redesign.

| Variable                  | Default                  | Purpose                                                  |
| ------------------------- | ------------------------ | ------------------------------------------------------- |
| `NEXT_PUBLIC_API_BASE`    | `http://localhost:8080`  | FastAPI backend base URL (studio endpoints, `/metrics`). |
| `NEXT_PUBLIC_GRAFANA_URL` | `http://localhost:3000`  | Grafana URL embedded/linked on the Observability tab.    |
| `NEXT_PUBLIC_INNGEST_URL` | `http://localhost:8288`  | Inngest dashboard URL linked on Observability.           |

> The Inngest dashboard default matches compose's `INNGEST_HOST_PORT` (8288).
> If you publish Grafana or Inngest on non-default host ports, override the URL
> vars at build time (compose `build.args`) so the Observability links resolve.

Copy `.env.example` to `.env.local` for local development:

```bash
cp .env.example .env.local
```

## Develop

```bash
npm install
npm run dev          # http://localhost:3000
```

Other scripts:

```bash
npm run build        # production build (standalone output)
npm run start        # serve the production build
npm run typecheck    # tsc --noEmit
```

> The backend has CORS enabled, so the browser talks to `NEXT_PUBLIC_API_BASE`
> directly. If you run the frontend on `:3000` and Grafana also uses `:3000`,
> point one elsewhere (e.g. `next dev -p 3001`, or set `NEXT_PUBLIC_GRAFANA_URL`).

## Docker

```bash
docker build \
  --build-arg NEXT_PUBLIC_API_BASE=http://localhost:8080 \
  --build-arg NEXT_PUBLIC_GRAFANA_URL=http://localhost:3001 \
  --build-arg NEXT_PUBLIC_INNGEST_URL=http://localhost:8288 \
  -t docie-studio ./frontend

docker run -p 3000:3000 docie-studio
```

**Important:** because `NEXT_PUBLIC_*` is baked at build time, pass them as
`--build-arg` (mirrored under `build.args` in docker-compose). Setting them only
as runtime `environment` will not reach the browser bundle. The Dockerfile's
`ARG`/`ENV` list already covers the three variables above — no changes were
needed for the redesign since no new build args were introduced.
