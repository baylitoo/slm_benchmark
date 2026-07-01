// Public runtime config. NEXT_PUBLIC_* values are inlined at build time, so
// these are resolved when the bundle is compiled (see Dockerfile build args).

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE?.replace(/\/$/, "") || "http://localhost:8080";

export const GRAFANA_URL =
  process.env.NEXT_PUBLIC_GRAFANA_URL?.replace(/\/$/, "") || "http://localhost:3000";

export const INNGEST_URL =
  process.env.NEXT_PUBLIC_INNGEST_URL?.replace(/\/$/, "") || "http://localhost:8288";

export const METRICS_URL = `${API_BASE}/metrics`;
