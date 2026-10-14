"""Studio-issued API keys: lifecycle, authentication lookup, and migrations.

Replaces nothing -- ``security.parse_api_keys`` and the ``API_KEYS`` env var
keep working untouched. Those become the OPERATOR tier (they may manage keys);
rows in this table are the CLIENT tier (they may call the serving surfaces).
That split keeps the bootstrap intact: the Studio already authenticates with
an operator-typed key from the browser, so nothing here grants a capability
the operator did not already hold.

Written for the multi-node fleet (#527) from the start: authentication is one
indexed lookup against the shared catalog with NO in-process cache, so every
API replica and every serving node resolves a key identically and a
revocation takes effect everywhere immediately. The cost is one small query
per authenticated request, which is the same order as the per-request
``deployments.json`` read the resolver already does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import logging
import secrets
import uuid
from typing import Any

from sqlalchemy import Table, case, func, select, update
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from docie_bench.storage.db import session_scope
from docie_bench.studio.key_crypto import decrypt_secret, encrypt_secret
from docie_bench.studio.models import ApiKeyRecord, UsageRecord, isoformat, utcnow

logger = logging.getLogger("docie_bench.studio.api_key_store")


class ApiKeyStorageUnavailableError(RuntimeError):
    """No catalog is configured, so a key cannot be issued.

    Distinct from ``KeyEncryptionUnavailableError``: that one means the cipher
    is unconfigured, this one means there is nowhere to persist the row. Both
    surface as 503 with a message naming the missing variable."""

# Arbitrary-but-stable advisory-lock keys, distinct from every other
# migration's -- see seed_store.py/usage_store.py for the same pattern.
_API_KEY_LOCK_KEY = 0x0D0C1E0C
_USAGE_API_KEY_COLUMN_LOCK_KEY = 0x0D0C1E0D

# Every issued key reads `docie_<prefix>_<secret>`. The fixed scheme prefix
# makes a leaked key greppable in logs and recognizable in a bug report; the
# per-key `prefix` is what the UI shows after the one-time reveal.
KEY_SCHEME = "docie"
_PREFIX_BYTES = 6  # 12 hex characters
_SECRET_BYTES = 32

# How stale ``last_used_at`` may get before a successful authentication
# rewrites it. Bounds the write amplification of the stamp: without it a key
# under load would UPDATE its own row on every single request.
_LAST_USED_RESOLUTION = dt.timedelta(minutes=5)


def ensure_api_key_table(engine: Engine) -> bool:
    """Race-safe forward migration: create ``studio_api_keys`` if missing.

    Same shape and reasoning as ``ensure_seed_run_table``: ``CREATE TABLE IF
    NOT EXISTS`` under ``pg_advisory_xact_lock`` on PostgreSQL, plus an
    explicit ``CreateIndex(..., if_not_exists=True)`` loop so the unique index
    on ``prefix`` is not silently skipped by whichever process loses the race
    to create the table.
    """
    from sqlalchemy.schema import CreateIndex, CreateTable

    table = ApiKeyRecord.__table__
    assert isinstance(table, Table)  # narrow FromClause for the compiler
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"), {"key": _API_KEY_LOCK_KEY}
            )
        existed = sa_inspect(connection).has_table("studio_api_keys")
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in table.indexes:
            connection.execute(CreateIndex(index, if_not_exists=True))
    return not existed


def ensure_usage_api_key_column(engine: Engine) -> bool:
    """Forward migration: add ``usage_records.api_key_id`` to an existing table.

    ``create_all`` never alters an existing table, so a database that predates
    per-key attribution would make the first ``record_usage`` throw
    ``UndefinedColumn``. Nullable, so no table rewrite and no lock pain.
    Mirrors ``catalog.ensure_placement_observed_columns`` exactly, including
    the sqlite inspect-then-ALTER fallback for tests. Returns whether the
    column was added.
    """
    inspector = sa_inspect(engine)
    if not inspector.has_table("usage_records"):
        return False  # create_all will create it complete
    with engine.begin() as connection:
        if engine.dialect.name == "postgresql":
            connection.execute(
                sa_text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _USAGE_API_KEY_COLUMN_LOCK_KEY},
            )
            existing = {
                row[0]
                for row in connection.execute(
                    sa_text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'usage_records'"
                    )
                )
            }
            connection.execute(
                sa_text("ALTER TABLE usage_records ADD COLUMN IF NOT EXISTS api_key_id VARCHAR(36)")
            )
            connection.execute(
                sa_text(
                    "CREATE INDEX IF NOT EXISTS ix_usage_records_api_key_id "
                    "ON usage_records (api_key_id)"
                )
            )
            return "api_key_id" not in existing
        columns = {column["name"] for column in inspector.get_columns("usage_records")}
        if "api_key_id" in columns:
            return False
        connection.execute(sa_text("ALTER TABLE usage_records ADD COLUMN api_key_id VARCHAR(36)"))
        return True


def _digest(secret: str) -> str:
    """SHA-256 of the secret, hex.

    Not bcrypt/argon2 on purpose: the secret is 32 bytes of ``secrets``
    output, so there is no dictionary to slow down, and a KDF on the
    authentication path would hand every caller a CPU cost and an attacker a
    cheap denial-of-service lever.
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def split_key(api_key: str) -> tuple[str, str] | None:
    """``docie_<prefix>_<secret>`` -> ``(prefix, secret)``, or None.

    ``split`` with maxsplit=2 so a secret containing the urlsafe-base64 ``_``
    character survives intact -- the naive three-way split would truncate
    roughly half of all generated keys.
    """
    parts = api_key.strip().split("_", 2)
    if len(parts) != 3:
        return None
    scheme, prefix, secret = parts
    if scheme != KEY_SCHEME or not prefix or not secret:
        return None
    return prefix, secret


def _view(record: ApiKeyRecord) -> dict[str, Any]:
    """The masked representation -- never the secret, never the ciphertext."""
    return {
        "id": record.id,
        "prefix": f"{KEY_SCHEME}_{record.prefix}",
        "label": record.label,
        "tenant_id": record.tenant_id,
        "created_at": isoformat(record.created_at),
        "last_used_at": isoformat(record.last_used_at) if record.last_used_at else None,
        "revoked_at": isoformat(record.revoked_at) if record.revoked_at else None,
        "revoked": record.revoked_at is not None,
    }


def create_key(*, label: str, tenant_id: str) -> tuple[dict[str, Any], str]:
    """Issue a key. Returns ``(masked_view, full_key)``.

    The full key is returned EXACTLY here and nowhere else in the normal flow
    -- every later read goes through :func:`_view`, which has no access to it.
    An operator who loses it uses :func:`reveal_key`.

    Raises ``KeyEncryptionUnavailableError`` (from ``encrypt_secret``) when
    ``API_KEY_ENCRYPTION_KEY`` is unset, rather than storing the secret in
    cleartext.
    """
    prefix = secrets.token_hex(_PREFIX_BYTES)
    secret = secrets.token_urlsafe(_SECRET_BYTES)
    full_key = f"{KEY_SCHEME}_{prefix}_{secret}"
    # Encrypt BEFORE opening the session: a missing encryption key must fail
    # without leaving a half-written row behind.
    ciphertext = encrypt_secret(secret)
    record = ApiKeyRecord(
        id=str(uuid.uuid4()),
        prefix=prefix,
        digest=_digest(secret),
        secret_ciphertext=ciphertext,
        label=label.strip()[:200],
        tenant_id=tenant_id.strip()[:128],
    )
    with session_scope() as session:
        if session is None:
            # Loudly, not silently: handing back a key that was never stored
            # would give the operator a credential that authenticates nothing.
            raise ApiKeyStorageUnavailableError(
                "DATABASE_URL is not configured; Studio-issued API keys need the catalog"
            )
        session.add(record)
        session.flush()
        view = _view(record)
    return view, full_key


def list_keys(*, tenant_id: str | None = None) -> list[dict[str, Any]]:
    """Masked views, newest first. Revoked keys are included (tombstones)."""
    statement = select(ApiKeyRecord).order_by(ApiKeyRecord.created_at.desc())
    if tenant_id is not None:
        statement = statement.where(ApiKeyRecord.tenant_id == tenant_id)
    with session_scope() as session:
        if session is None:
            return []
        return [_view(row) for row in session.execute(statement).scalars()]


def revoke_key(key_id: str) -> dict[str, Any] | None:
    """Tombstone the key. Idempotent: an already-revoked key keeps its original
    ``revoked_at`` rather than having it moved forward."""
    with session_scope() as session:
        if session is None:
            return None
        record = session.get(ApiKeyRecord, key_id)
        if record is None:
            return None
        if record.revoked_at is None:
            record.revoked_at = utcnow()
        session.flush()
        return _view(record)


def reveal_key(key_id: str) -> str | None:
    """The full key, decrypted -- the operator escape hatch after the one-time
    reveal. Returns None when no such key exists.

    Logged at WARNING with the key id and no secret: recovering a credential is
    exactly the kind of action that should leave a trace someone can find
    later.
    """
    with session_scope() as session:
        if session is None:
            return None
        record = session.get(ApiKeyRecord, key_id)
        if record is None:
            return None
        prefix, ciphertext = record.prefix, record.secret_ciphertext
    secret = decrypt_secret(ciphertext)
    logger.warning("api key revealed", extra={"docie_api_key_id": key_id})
    return f"{KEY_SCHEME}_{prefix}_{secret}"


def lookup_by_secret(api_key: str) -> tuple[str, str] | None:
    """``(tenant_id, api_key_id)`` for a live key, else None.

    The authentication hot path. One indexed lookup on ``prefix``, then a
    constant-time digest comparison -- so a wrong secret costs the same as a
    right one, and a caller cannot learn anything from response timing. A
    revoked key returns None, which is why there is no cache: revocation has
    to bite on the next request, on every replica.

    Never raises: a database outage degrades to "this key does not
    authenticate" and the caller falls through to the env-configured keys and
    then to 401, rather than turning a catalog blip into a 500 on every
    request.
    """
    parsed = split_key(api_key)
    if parsed is None:
        return None
    prefix, secret = parsed
    expected = _digest(secret)
    try:
        with session_scope() as session:
            if session is None:
                return None
            record = session.execute(
                select(ApiKeyRecord).where(ApiKeyRecord.prefix == prefix)
            ).scalar_one_or_none()
            if record is None or record.revoked_at is not None:
                return None
            if not hmac.compare_digest(record.digest, expected):
                return None
            # Stamp last_used here rather than in a second call: the row is
            # already loaded, and the coarse threshold keeps a busy key from
            # issuing one UPDATE per request just to move a timestamp a few
            # milliseconds. The column exists to spot DORMANT keys, so minute
            # resolution is more than enough.
            now = utcnow()
            if record.last_used_at is None or (now - record.last_used_at) > _LAST_USED_RESOLUTION:
                session.execute(
                    update(ApiKeyRecord)
                    .where(ApiKeyRecord.id == record.id)
                    .values(last_used_at=now)
                )
            return record.tenant_id, record.id
    except SQLAlchemyError:
        logger.warning("api key lookup unavailable", exc_info=True)
        return None


def usage_by_key(*, window_hours: int, tenant_id: str | None = None) -> list[dict[str, Any]]:
    """Per-key aggregates over the last ``window_hours``.

    Joined onto the key rows so a key that has never been used still appears
    (with zeros) -- a usage view that silently omits unused keys makes it look
    as though they do not exist. Rows whose ``api_key_id`` is NULL (env keys,
    anonymous traffic) are not attributable to a key and are excluded here;
    ``usage_store.usage_summary`` still counts them per tenant.
    """
    since = utcnow() - dt.timedelta(hours=window_hours)
    # SUM over a CASE rather than COUNT(...) FILTER: the aggregate has to
    # render on sqlite (tests) as well as PostgreSQL.
    error_count = func.sum(case((UsageRecord.status != "ok", 1), else_=0))
    statement = (
        select(
            ApiKeyRecord.id,
            func.count(UsageRecord.id).label("requests"),
            func.coalesce(func.sum(UsageRecord.prompt_tokens), 0).label("prompt_tokens"),
            func.coalesce(func.sum(UsageRecord.completion_tokens), 0).label("completion_tokens"),
            func.coalesce(error_count, 0).label("errors"),
            func.coalesce(func.sum(UsageRecord.latency_ms), 0).label("latency_ms_total"),
        )
        .select_from(ApiKeyRecord)
        .outerjoin(
            UsageRecord,
            (UsageRecord.api_key_id == ApiKeyRecord.id) & (UsageRecord.created_at >= since),
        )
        .group_by(ApiKeyRecord.id)
    )
    if tenant_id is not None:
        statement = statement.where(ApiKeyRecord.tenant_id == tenant_id)
    with session_scope() as session:
        if session is None:
            return []
        rows = session.execute(statement).all()
        views = {
            record.id: _view(record)
            for record in session.execute(select(ApiKeyRecord)).scalars()
        }
    out: list[dict[str, Any]] = []
    for row in rows:
        view = dict(views.get(row.id, {"id": row.id}))
        requests = int(row.requests or 0)
        view.update(
            {
                "requests": requests,
                "prompt_tokens": int(row.prompt_tokens or 0),
                "completion_tokens": int(row.completion_tokens or 0),
                "errors": int(row.errors or 0),
                # Mean, not a percentile: percentiles need the raw rows, and
                # usage_store.percentile already owns that for the per-deployment
                # view. Null rather than 0 when nothing ran, so the UI can say
                # "no traffic" instead of claiming a 0 ms average.
                "avg_latency_ms": (
                    round(int(row.latency_ms_total or 0) / requests) if requests else None
                ),
            }
        )
        out.append(view)
    out.sort(key=lambda row: row["requests"], reverse=True)
    return out


__all__ = [
    "ApiKeyStorageUnavailableError",
    "KEY_SCHEME",
    "create_key",
    "ensure_api_key_table",
    "ensure_usage_api_key_column",
    "list_keys",
    "lookup_by_secret",
    "reveal_key",
    "revoke_key",
    "split_key",
    "usage_by_key",
]
