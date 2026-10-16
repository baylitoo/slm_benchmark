# Authentication and tenancy

Three kinds of credential reach this platform, and they are not
interchangeable. Each exists because the other two cannot do its job.

| Credential | Held by | Grants | Lives in |
|---|---|---|---|
| **Env API key** | an operator | everything, including key management | `API_KEYS` |
| **Studio-issued key** | an application | the `/v1` inference surface | `studio_api_keys` table |
| **Studio session token** | a signed-in person | the Studio, and their own keys | the browser, for the tab's life |

All three arrive in the same `X-API-Key` header and are resolved by one
function, `TenantQuotaManager.authenticate` in `src/docie_bench/security.py`.

## The ladder

`authenticate` tries them in a deliberate order:

1. **A session token**, recognised by its `docie_sess_` prefix. It is
   self-describing, so there is no reason to compare it against operator keys,
   and no way for one to be mistaken for the other. A token carrying the prefix
   but failing verification is a bad credential — it falls through to `401`
   rather than being retried as a key.
2. **Env keys** from `API_KEYS`, compared with `hmac.compare_digest`. This is
   the **operator** tier.
3. **Studio-issued keys**, resolved by one indexed lookup in the catalog. The
   **client** tier.

Env keys are checked before database keys on purpose: a catalog row must never
be able to shadow a configured operator key.

The result is a `TenantContext` carrying `tenant_id`, `authenticated`, and two
flags — `operator` and `studio_user` — that decide what the caller may manage.

## Why a person's credential is not an API key

The Studio is gated by the company auth server, but a browser cannot hold a
durable API key: anything in frontend JavaScript is readable by everyone using
that page. So a signed-in user's browser is given a **short-lived HMAC-signed
session token** instead, minted by `src/docie_bench/studio_session.py`.

Verifying the company's own access token directly is not available to us. The
auth server publishes no JWKS and no introspection endpoint, and holding its
shared signing secret would let this service forge a token for any employee.
What it does provide is `/auth/refresh`, and that is enough: the browser signs
in directly — its password never reaches this platform — then hands us its
refresh token. We exchange it upstream, and a valid token pair coming back
proves the session is live.

That exchange happens at `POST /v1/auth/session`, which is deliberately
unauthenticated, because it is the endpoint that *establishes* authentication.
It is mounted outside the Studio router so a later refactor cannot apply that
router's dependency to it.

Two properties fall out of the design and are worth knowing:

- **Renewal replaces the browser's own refresh call** rather than adding to it.
  The exchange refreshes upstream once and returns both the rotated company
  pair and a fresh session token, so load on the company auth server is
  unchanged and the rotated refresh token is never stranded.
- **A session token cannot outlive the session that proved it.** Its lifetime
  is derived from the company token's own `expiresIn` and capped. Because that
  cap can make it expire first, the browser renews on whichever of the two
  deadlines comes sooner.

With no session secret configured the exchange returns `503` and the Studio
asks for an operator key, exactly as it did before any of this existed.

## Who may manage keys

Issuing keys is part of what the Studio is for, so a signed-in user may create
and revoke them — but only their own.

| Caller | Create | List, reveal, revoke |
|---|---|---|
| Session token (`studio_user`) | yes, owner forced to their login | their own keys only |
| Env key (`operator`) | yes, any tenant | everything |
| Studio-issued key | no | no |

Two decisions in there are load-bearing:

**The owner is the verified login, not the request body.** A caller may send a
`tenant_id`, and it is honoured only for an operator. Otherwise any signed-in
user could mint a key owned by somebody else.

**Another user's key answers `404`, not `403`.** A `403` confirms the key id
exists, which is precisely what someone probing for other people's keys wants
to learn.

A Studio-issued key may manage nothing at all. If a leaked key could mint a
successor, revoking it would accomplish nothing.

## Key storage

Each issued key is stored three ways, each for one job:

- `prefix` — plaintext and indexed, so authentication is one indexed lookup
  rather than a scan over every key.
- `digest` — SHA-256, compared in constant time on the hot path. Not bcrypt or
  argon2: the secret is 32 bytes of `secrets` output, so there is no dictionary
  to slow down, and a slow KDF on the authentication path only hands callers a
  CPU cost and attackers a denial-of-service lever.
- `secret_ciphertext` — AES-GCM, because an operator must be able to recover a
  key after the one-time reveal.

That last one is a deliberate trade-off, not an oversight. A database dump
**plus** `API_KEY_ENCRYPTION_KEY` exposes live keys, which a pure hash would
not. The consequences are wired in accordingly: the encryption key lives only
in the environment, a reveal is logged at `WARNING`, and the reveal route is a
`POST` so a live credential never lands in browser history or a proxy log.

Revocation is a tombstone rather than a delete — usage rows reference the key
id, and a revoked key's history has to stay attributable.

## Quotas, and where they stop working

`TenantQuotaManager` also bounds traffic: a per-window request budget, a
separate and larger budget for `GET`/`HEAD` (the Studio polls continuously, and
counting those against inference would make the default quota self-deny), a
concurrency cap, and a per-IP auth-failure bucket checked *before* a bad key can
reach the tenant limiter.

**These counters are in-process dictionaries.** With several API replicas each
enforces its own budget, so a quota is effectively multiplied by the replica
count. This is the same class of problem as the process-local round-robin
counter in the placement resolver, and it is why per-key rate limits have not
shipped: they would not mean what they say. The fix is a shared store, and
Redis is already in the compose file.

## Behind a proxy

Every request arrives from the reverse proxy's address unless
`FORWARDED_ALLOW_IPS` names that proxy's network. Without it the per-IP
auth-failure bucket is effectively global — one misconfigured client can lock
out every other caller behind the same proxy. This has happened in production.
