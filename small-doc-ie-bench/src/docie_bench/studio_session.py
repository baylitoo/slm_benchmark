"""Studio session tokens: company sign-in alone authorizes the Studio.

The Studio is gated by the company auth server (frontend/lib/auth.ts), but the
browser still had to carry an operator ``X-API-Key`` for every backend call --
a second, unrelated credential typed by hand into a console whose own job
includes issuing keys. This module removes that prompt.

HOW THE COMPANY SESSION IS VERIFIED, and why it is done this way. Accepting the
company access token at face value is not an option: anything that trusts an
unverified JWT lets a caller mint ``{"sub": "admin"}`` for themselves, which is
strictly worse than requiring a key. Verifying it cryptographically needs key
material from the auth server, and that server exposes none -- probed on the
live deployment:

* ``/auth/me``, ``/auth/userinfo``, ``/auth/profile``, ``/auth/introspect``,
  ``/auth/validate`` -> 404 (while ``GET /auth/login`` -> 405, so the probe
  distinguishes "absent" from "wrong method"). No introspection endpoint.
* ``/.well-known/jwks.json``, ``/oauth2/jwks``, ``/.well-known/openid-
  configuration`` -> 404 or the SPA's HTML. No JWKS, no OIDC discovery.

That leaves a shared HS256 secret, which would let DocIE forge a token for ANY
employee for as long as the secret lives -- far too much authority to hold for
the sake of dropping a prompt.

So verification uses the only oracle the auth server definitely provides: its
own ``/auth/refresh``. The browser signs in directly (its password never
reaches DocIE, unchanged by this module) and hands us the refresh token; we
exchange it server-side, and a valid token pair coming back proves the session
is genuine. The rotated pair is handed straight back to the browser, because
the server rotates refresh tokens and keeping the old one would log the user
out on their next renewal.

What we issue is deliberately NOT an ``ApiKeyRecord``: those are credentials
for applications, created and revoked by an operator. A Studio session is a
stateless, short-lived, HMAC-signed assertion -- no row to clean up when a tab
closes, and no catalog dependency on the sign-in path.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Any

from docie_bench.settings import get_settings

logger = logging.getLogger("docie_bench.studio_session")

# Distinct prefix so a session token is never mistaken for an operator key in a
# log, a bug report, or the authenticate() ladder.
TOKEN_PREFIX = "docie_sess_"  # noqa: S105 - a scheme prefix, not a secret

# Matches the company access token's documented 30-minute lifetime, and is the
# CEILING for a minted session: a Studio token must never outlive the company
# session it was derived from, or DocIE access would survive a sign-out or an
# expired company session. The exchange derives the actual TTL from the
# upstream `expiresIn` and clamps it here, so a shorter company session
# shortens the Studio session with it.
DEFAULT_TTL_SECONDS = 1800

# Floor on a derived TTL. An auth server reporting something tiny (or zero)
# must not produce a token that is already dead on arrival; the browser would
# spin re-exchanging it.
MIN_TTL_SECONDS = 60


def ttl_for(upstream_expires_in: Any) -> int:
    """TTL for a session minted against a company token lasting
    ``upstream_expires_in`` seconds, clamped to [MIN, DEFAULT].

    Never longer than the company token: the whole point of the exchange is
    that company sign-in is what authorizes the Studio, so Studio access has to
    end no later than the session that proved it.
    """
    if isinstance(upstream_expires_in, bool) or not isinstance(upstream_expires_in, (int, float)):
        return DEFAULT_TTL_SECONDS
    if upstream_expires_in <= 0:
        return MIN_TTL_SECONDS
    return max(MIN_TTL_SECONDS, min(DEFAULT_TTL_SECONDS, int(upstream_expires_in)))


class StudioSessionUnavailableError(RuntimeError):
    """Session exchange is not configured (no signing secret, or no auth base)."""


def _signing_key() -> bytes:
    configured = get_settings().studio_session_secret.get_secret_value().strip()
    if not configured:
        raise StudioSessionUnavailableError(
            "STUDIO_SESSION_SECRET is not set; the Studio cannot exchange a company "
            "session for backend access and will keep asking for an API key"
        )
    return hashlib.sha256(configured.encode("utf-8")).digest()


def session_exchange_configured() -> bool:
    """Whether the exchange can work, so a surface can say so up front."""
    settings = get_settings()
    return bool(
        settings.studio_session_secret.get_secret_value().strip()
        and settings.company_auth_base_url.strip()
    )


def _b64(raw: bytes) -> str:
    # urlsafe + stripped padding: the token travels in an X-API-Key header, and
    # "=" is legal there but noisy in logs.
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def mint(login: str, *, ttl_seconds: int = DEFAULT_TTL_SECONDS, now: float | None = None) -> str:
    """A signed ``docie_sess_<payload>.<signature>`` for ``login``."""
    issued = int(now if now is not None else time.time())
    payload = json.dumps(
        {"login": login, "exp": issued + ttl_seconds}, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    encoded = _b64(payload)
    signature = _b64(hmac.new(_signing_key(), encoded.encode("ascii"), hashlib.sha256).digest())
    return f"{TOKEN_PREFIX}{encoded}.{signature}"


def verify(token: str, *, now: float | None = None) -> str | None:
    """The login this token asserts, or None when it is not a valid session.

    Returns None rather than raising for every rejection -- a bad token is a
    401 like any other bad credential, and the caller should not have to tell
    "malformed" from "expired" to produce the same response. Never raises on
    a missing secret either: with the exchange unconfigured, no token can have
    been issued, so every candidate is simply invalid.
    """
    if not token.startswith(TOKEN_PREFIX):
        return None
    body = token[len(TOKEN_PREFIX) :]
    encoded, _, signature = body.partition(".")
    if not encoded or not signature:
        return None
    try:
        key = _signing_key()
    except StudioSessionUnavailableError:
        return None
    expected = _b64(hmac.new(key, encoded.encode("ascii"), hashlib.sha256).digest())
    # Constant-time: a wrong signature must cost the same as a right one.
    if not hmac.compare_digest(expected, signature):
        return None
    try:
        claims: dict[str, Any] = json.loads(_unb64(encoded))
    except (ValueError, TypeError):
        return None
    login = claims.get("login")
    expires = claims.get("exp")
    if not isinstance(login, str) or not login or not isinstance(expires, int):
        return None
    if (now if now is not None else time.time()) >= expires:
        return None
    return login


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "MIN_TTL_SECONDS",
    "StudioSessionUnavailableError",
    "TOKEN_PREFIX",
    "mint",
    "session_exchange_configured",
    "ttl_for",
    "verify",
]
