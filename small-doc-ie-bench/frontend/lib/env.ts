// Public runtime config. NEXT_PUBLIC_* values are inlined at build time, so
// these are resolved when the bundle is compiled (see Dockerfile build args).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8080";

// docker-compose maps Grafana to host port 3001, not 3000 — 3000 is the
// Studio's own web UI (see docker-compose.yml's build-arg default and its
// "leave 3000 for the DocIE Studio web UI" comment). This fallback only
// matters outside compose (e.g. `next dev`); keep it in sync with that.
export const GRAFANA_URL =
  process.env.NEXT_PUBLIC_GRAFANA_URL?.replace(/\/$/, "") || "http://localhost:3001";

export const INNGEST_URL =
  process.env.NEXT_PUBLIC_INNGEST_URL?.replace(/\/$/, "") || "http://localhost:8288";

// The Inngest API the @inngest/realtime browser hook connects to for live
// progress. The lib reads process.env.NEXT_PUBLIC_INNGEST_BASE_URL directly
// (getEnvVar("INNGEST_BASE_URL")); this export both documents it and guarantees
// Next inlines the token even though the runtime read happens inside the lib.
// Unset => the lib defaults to Inngest Cloud (api.inngest.com), so a self-hosted
// token's socket is refused and the UI falls back to /runs polling.
export const INNGEST_REALTIME_BASE_URL =
  process.env.NEXT_PUBLIC_INNGEST_BASE_URL?.replace(/\/$/, "") || "http://localhost:8288";

export const METRICS_URL = `${API_BASE}/metrics`;

// The docie dashboard's uid is pinned in infra/grafana/dashboards/docie.json
// ("uid": "docie") specifically so this URL is stable across a fresh Grafana
// volume (an unpinned dashboard gets a random uid at provision time). `kiosk`
// hides Grafana's own nav/header so the iframe reads as an embedded panel, not
// a second app-inside-an-app. Requires GF_AUTH_ANONYMOUS_ENABLED (else a login
// wall) and GF_SECURITY_ALLOW_EMBEDDING (else X-Frame-Options blocks the
// frame outright) — both set on the grafana service in docker-compose.yml.
export const GRAFANA_DASHBOARD_URL = `${GRAFANA_URL}/d/docie/small-document-ie-benchmark?orgId=1&kiosk`;
