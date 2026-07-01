# DocIE Studio (frontend)

A litellm-style web console for the DocIE benchmark backend. Built with
Next.js (App Router, TypeScript) and Tailwind CSS, with a modern SaaS-style
shell: sidebar navigation, cards, dark mode by default with a light toggle,
loading skeletons, empty states, and toast notifications.

Four sections (all stay mounted; only the active one is shown so a running job
survives navigation):

1. **Playground** — paste text or upload a PDF/image, run an extraction, and
   watch live progress. Uses Inngest **realtime** (`@inngest/realtime`) when the
   backend can mint a token, and transparently **falls back to polling**
   `GET /v1/studio/runs/{event_id}` when realtime is unavailable (HTTP 501).
2. **Deploy** — pick a model from the GGUF model store, choose a runtime scoped
   to that model, and serve it; plus an "Add model" form to seed the store from
   a local Ollama reference. Models and deployments **auto-refresh** on an
   interval. (See "Deploy tab" below.)
3. **Benchmark** — start a benchmark run (a `dataset` is required) and browse
   past runs + their metrics.
4. **Observability** — embeds Grafana and links to Inngest and raw Prometheus
   metrics (all driven by the env vars below so they can point anywhere).

The API client treats `404`/`501` as "coming soon" and the UI degrades
gracefully — endpoints that aren't enabled on a given backend never crash the UI.

## Deploy tab

- **Available models** come from `GET /v1/serving/store` (the queryable GGUF
  model-store catalog), shown as an accessible, selectable radio-group list. If
  the catalog isn't enabled the route returns `501` (or `404` on older builds)
  and a friendly "no models in the store yet" / "coming soon" state is shown.
- **Runtime picker** is scoped to the chosen model: it uses that store entry's
  `available_backends` array directly (e.g. `llama-server`, `ollama`). Picking a
  runtime is optional — "Auto" deploys the bare store entry and lets the server
  choose; selecting a backend serves it explicitly.
- **Add model (seed)** posts to `POST /v1/studio/seed-ollama`
  `{ reference, name, family }` and streams progress; the family list comes from
  `GET /v1/serving/families`.
- **Deploy** posts to `POST /v1/studio/deploy`
  `{ model, runtime?, name?, port?, context_length? }` and streams progress via
  the returned `channel` (realtime, else polling). The **Deployments** table
  (`GET /v1/serving/deployments`) reflects the new deployment on its next
  auto-refresh.
- **Auto-refresh**: the model store and deployments lists poll every ~4s with a
  visible "Live · Xs ago" indicator and a manual refresh button. Polling
  auto-pauses when the browser tab is hidden **and** when Deploy isn't the
  active section.

## Tech / dependencies

Runtime deps added for the redesign (all small, React 19-compatible):

| Package          | Why                                              |
| ---------------- | ------------------------------------------------ |
| `lucide-react`   | Icon set.                                        |
| `next-themes`    | Dark/light theme toggle (dark by default).       |
| `clsx`           | Conditional class names.                         |
| `tailwind-merge` | Resolve Tailwind class conflicts (`cn()` helper).|

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

> In this deployment Grafana is actually published on host port **3001** and the
> Inngest dashboard on **8290**. Override the two URL vars accordingly at build
> time (e.g. compose `build.args`) so the Observability links resolve correctly.

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
  --build-arg NEXT_PUBLIC_INNGEST_URL=http://localhost:8290 \
  -t docie-studio ./frontend

docker run -p 3000:3000 docie-studio
```

**Important:** because `NEXT_PUBLIC_*` is baked at build time, pass them as
`--build-arg` (mirrored under `build.args` in docker-compose). Setting them only
as runtime `environment` will not reach the browser bundle. The Dockerfile's
`ARG`/`ENV` list already covers the three variables above — no changes were
needed for the redesign since no new build args were introduced.
