import { ApiError, ApiUnavailable } from "./api";

/**
 * THE error-to-copy normalizer. Every toast/inline error goes through here so
 * one backend condition reads the same on every page (previously seven
 * hand-rolled variants disagreed — Agents showed a raw HTTP string where
 * Deploy showed a friendly sentence for the same failure).
 *
 * `unavailable` is the feature-specific sentence for a backend that does not
 * ship the route at all (`ApiUnavailable` — a BARE 404/501). Domain errors
 * ("deployment 'x' not found") arrive as `ApiError` carrying the server's own
 * detail and are surfaced verbatim.
 */
export function toUserMessage(
  err: unknown,
  opts: { fallback?: string; unavailable?: string } = {},
): string {
  const fallback = opts.fallback ?? "Something went wrong.";
  if (err instanceof ApiUnavailable) {
    return opts.unavailable ?? "This feature isn't available on this server.";
  }
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message || fallback;
  return fallback;
}
