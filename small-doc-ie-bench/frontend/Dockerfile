# syntax=docker/dockerfile:1

# DocIE Studio frontend — multi-stage build (next build -> next start).
#
# NOTE: NEXT_PUBLIC_* values are inlined into the client bundle at BUILD TIME,
# not read at runtime. Pass them as --build-arg so they reach the browser.
# In docker-compose, mirror them under `build.args` (not just `environment`).

# ---------- deps ----------
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci

# ---------- builder ----------
FROM node:20-alpine AS builder
WORKDIR /app

# Public config baked into the bundle. Override per environment at build time.
ARG NEXT_PUBLIC_API_BASE=http://localhost:8080
ARG NEXT_PUBLIC_GRAFANA_URL=http://localhost:3000
# The Inngest dashboard LINK (Observability). Distinct from the realtime base
# below: this is just where "open the Inngest dashboard" points.
ARG NEXT_PUBLIC_INNGEST_URL=http://localhost:8288
# The Inngest API the @inngest/realtime BROWSER hook connects to for live
# progress. The lib reads exactly this name (getEnvVar("INNGEST_BASE_URL") ->
# process.env.NEXT_PUBLIC_INNGEST_BASE_URL); unset, it defaults to Inngest Cloud
# (api.inngest.com) and a self-hosted-minted token's socket is refused -> the UI
# silently falls back to /runs polling. Point it at the BROWSER-reachable
# host-published Inngest API port (INNGEST_HOST_PORT, default 8288) — never the
# internal docker name (ws://inngest:8289), which the browser cannot resolve.
ARG NEXT_PUBLIC_INNGEST_BASE_URL=http://localhost:8288
ENV NEXT_PUBLIC_API_BASE=$NEXT_PUBLIC_API_BASE \
    NEXT_PUBLIC_GRAFANA_URL=$NEXT_PUBLIC_GRAFANA_URL \
    NEXT_PUBLIC_INNGEST_URL=$NEXT_PUBLIC_INNGEST_URL \
    NEXT_PUBLIC_INNGEST_BASE_URL=$NEXT_PUBLIC_INNGEST_BASE_URL \
    NEXT_TELEMETRY_DISABLED=1

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ---------- runner ----------
FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

RUN addgroup --system --gid 1001 nodejs \
  && adduser --system --uid 1001 nextjs

# Standalone output contains a minimal server + only required node_modules.
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs
EXPOSE 3000
CMD ["node", "server.js"]
