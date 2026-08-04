// Public runtime config. NEXT_PUBLIC_* values are inlined at build time, so
// these are resolved when the bundle is compiled (see Dockerfile build args).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8080";

export const GRAFANA_URL =
  process.env.NEXT_PUBLIC_GRAFANA_URL?.replace(/\/$/, "") || "http://localhost:3000";

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
