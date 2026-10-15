"""Studio-issued API key management and per-key usage.

Every route here requires the OPERATOR tier (an ``API_KEYS`` env key). A
Studio-issued key must not be able to mint another one: otherwise revoking a
leaked key accomplishes nothing, because it could already have issued itself a
successor that survives the revocation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from docie_bench.security import TenantContext, TenantDependency

router = APIRouter()


class ApiKeyCreateRequest(BaseModel):
    label: str = Field(min_length=1, max_length=200)
    # Explicit, not derived from the Studio session: the backend never
    # verifies the company JWT (it gates the UI only), so a tenant claimed by
    # the browser would be an unverified assertion. See issue #540.
    tenant_id: str = Field(min_length=1, max_length=128)


def _require_operator(tenant: TenantContext) -> None:
    """403 unless the caller authenticated with an env-configured key."""
    if not tenant.operator:
        raise HTTPException(
            status_code=403,
            detail=(
                "Managing API keys requires an operator key (one configured in "
                "API_KEYS); a Studio-issued key cannot create or revoke keys"
            ),
        )


@router.get("/api-keys")
async def list_api_keys_route(tenant: TenantDependency) -> dict[str, Any]:
    """Masked views of every issued key, newest first, revoked ones included.

    Never returns a secret. Missing DATABASE_URL degrades to an empty listing,
    the same contract as every other Studio listing route.
    """
    _require_operator(tenant)
    from docie_bench.studio.api_key_store import list_keys
    from docie_bench.studio.key_crypto import encryption_configured

    return {"keys": list_keys(), "encryption_configured": encryption_configured()}


@router.post("/api-keys", status_code=201)
async def create_api_key_route(
    payload: ApiKeyCreateRequest, tenant: TenantDependency
) -> dict[str, Any]:
    """Issue a key and return its full value ONCE.

    ``key`` is present in this response and in no other: every later read is
    masked to the prefix. An operator who loses it uses the reveal route.
    """
    _require_operator(tenant)
    from docie_bench.studio.api_key_store import (
        ApiKeyStorageUnavailableError,
        create_key,
    )
    from docie_bench.studio.key_crypto import KeyEncryptionUnavailableError

    try:
        view, full_key = create_key(label=payload.label, tenant_id=payload.tenant_id)
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
    _require_operator(tenant)
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
    _require_operator(tenant)
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
    _require_operator(tenant)
    from docie_bench.studio.api_key_store import usage_by_key
    from docie_bench.studio.usage_store import WINDOW_HOURS

    hours = WINDOW_HOURS.get(window)
    if hours is None:
        raise HTTPException(
            status_code=400,
            detail=f"window must be one of {', '.join(sorted(WINDOW_HOURS))} (got {window!r})",
        )
    return {"window": window, "keys": usage_by_key(window_hours=hours)}
