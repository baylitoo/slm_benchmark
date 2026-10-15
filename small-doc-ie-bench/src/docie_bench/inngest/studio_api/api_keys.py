"""Studio-issued API key management and per-key usage.

Who may do what here, and why:

* A **company-signed-in user** (``studio_user``) may create and revoke keys,
  but only their OWN -- issuing keys is part of what the Studio is for, so
  gating it on a separately-typed operator key would defeat the point. Every
  read and write is filtered to ``tenant_id == their login``, and the owner of
  a new key is forced to that login rather than taken from the request body:
  otherwise anyone could mint a key owned by somebody else.
* An **operator** (an ``API_KEYS`` env key) may do all of it, unscoped. They
  provision on behalf of others, so they see every key.
* A **Studio-issued key** may do NONE of it. If a leaked key could mint
  another, revoking it would accomplish nothing -- it could already have
  issued itself a successor that survives the revocation.

Another user's key reads as 404, never 403: a 403 would confirm the key
exists, which is exactly what a caller probing for other people's keys wants
to learn.

This is deliberately flat beyond that -- no roles, no read-only tier. See the
roles/permissions issue for the real model.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.security import TenantContext, TenantDependency

router = APIRouter()


class ApiKeyCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    # Honoured only for an OPERATOR, who provisions on behalf of others. A
    # company-signed-in caller has this overridden with their own verified
    # login (see _owner_for), so a browser cannot claim a tenant it has not
    # proved.
    tenant_id: str | None = Field(default=None, max_length=128)


def _require_key_management(tenant: TenantContext) -> None:
    """403 unless the caller may manage keys at all."""
    if not (tenant.operator or tenant.studio_user):
        raise HTTPException(
            status_code=403,
            detail=(
                "Managing API keys requires a company sign-in or an operator key "
                "(one configured in API_KEYS); a Studio-issued API key cannot "
                "create or revoke keys"
            ),
        )


def _scope_for(tenant: TenantContext) -> str | None:
    """The tenant a caller may see, or None for an unscoped operator."""
    return None if tenant.operator else tenant.tenant_id


def _owner_for(tenant: TenantContext, requested: str | None) -> str:
    """Who a newly created key belongs to.

    An operator may name any tenant. Anyone else gets their own verified
    identity regardless of what the body asked for -- the request body is not
    evidence of anything.
    """
    if tenant.operator and requested and requested.strip():
        return requested.strip()
    return tenant.tenant_id


def _owned_or_404(tenant: TenantContext, key_id: str) -> None:
    """404 unless the caller owns ``key_id`` (operators own everything).

    Checked before any mutation. 404 rather than 403 so the response does not
    confirm that someone else's key id exists.
    """
    scope = _scope_for(tenant)
    if scope is None:
        return
    from docie_bench.studio.api_key_store import list_keys

    if not any(row["id"] == key_id for row in list_keys(tenant_id=scope)):
        raise HTTPException(status_code=404, detail=f"no API key {key_id!r}")


@router.get("/api-keys")
async def list_api_keys_route(tenant: TenantDependency) -> dict[str, Any]:
    """Masked views of every issued key, newest first, revoked ones included.

    Never returns a secret. Missing DATABASE_URL degrades to an empty listing,
    the same contract as every other Studio listing route.
    """
    _require_key_management(tenant)
    from docie_bench.studio.api_key_store import list_keys
    from docie_bench.studio.key_crypto import encryption_configured

    return {
        "keys": list_keys(tenant_id=_scope_for(tenant)),
        "encryption_configured": encryption_configured(),
        # So the UI can say "your keys" vs "all keys" honestly rather than
        # implying a signed-in user is seeing the whole fleet.
        "scoped_to": _scope_for(tenant),
    }


@router.post("/api-keys", status_code=201)
async def create_api_key_route(
    payload: ApiKeyCreateRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Issue a key and return its full value ONCE.

    ``key`` is present in this response and in no other: every later read is
    masked to the prefix. An operator who loses it uses the reveal route.
    """
    _require_key_management(tenant)
    from docie_bench.studio.api_key_store import (
        ApiKeyStorageUnavailableError,
        create_key,
    )
    from docie_bench.studio.key_crypto import KeyEncryptionUnavailableError

    try:
        view, full_key = create_key(
            label=payload.label, tenant_id=_owner_for(tenant, payload.tenant_id)
        )
    except (KeyEncryptionUnavailableError, ApiKeyStorageUnavailableError) as exc:
        # 503, not 500: the deployment is missing configuration, and the
        # operator can fix it. The message names the variable.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {**view, "key": full_key}


@router.post("/api-keys/{key_id}/reveal")
async def reveal_api_key_route(key_id: str, tenant: TenantDependency) -> dict[str, Any]:
    """The full key again, decrypted from storage.

    POST rather than GET deliberately: this returns a live credential, and a
    GET would land in browser history, proxy logs and prefetchers. The store
    logs the reveal at WARNING with the key id.
    """
    _require_key_management(tenant)
    _owned_or_404(tenant, key_id)
    from docie_bench.studio.api_key_store import reveal_key
    from docie_bench.studio.key_crypto import KeyEncryptionUnavailableError

    try:
        full_key = reveal_key(key_id)
    except KeyEncryptionUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if full_key is None:
        raise HTTPException(status_code=404, detail=f"no API key {key_id!r}")
    return {"id": key_id, "key": full_key}


@router.delete("/api-keys/{key_id}")
async def revoke_api_key_route(key_id: str, tenant: TenantDependency) -> dict[str, Any]:
    """Revoke a key. A tombstone, not a delete -- usage rows reference its id,
    and a revoked key's history must stay attributable. Idempotent."""
    _require_key_management(tenant)
    _owned_or_404(tenant, key_id)
    from docie_bench.studio.api_key_store import revoke_key

    view = revoke_key(key_id)
    if view is None:
        raise HTTPException(status_code=404, detail=f"no API key {key_id!r}")
    return view


@router.get("/api-keys/usage")
async def api_key_usage_route(
    tenant: TenantDependency, window: str = "24h"
) -> dict[str, Any]:
    """Per-key aggregates over a bounded window.

    Keys with no traffic appear with zeros rather than being omitted, so the
    view answers "is this key being used" and not only "who is busiest".
    Requests made with an env key or anonymously are not attributable to a key
    row and are excluded here; ``GET /usage`` still counts them per tenant.
    """
    _require_key_management(tenant)
    from docie_bench.studio.api_key_store import usage_by_key
    from docie_bench.studio.usage_store import WINDOW_HOURS

    hours = WINDOW_HOURS.get(window)
    if hours is None:
        raise HTTPException(
            status_code=400,
            detail=f"window must be one of {', '.join(sorted(WINDOW_HOURS))} (got {window!r})",
        )
    return {
        "window": window,
        "keys": usage_by_key(window_hours=hours, tenant_id=_scope_for(tenant)),
    }
