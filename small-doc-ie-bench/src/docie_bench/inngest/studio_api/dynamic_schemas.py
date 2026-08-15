"""Named ``DynamicSchemaSpec`` CRUD routes.

Route glue only -- persistence lives in ``docie_bench.studio.dynamic_schemas``
(a different module; this one is the FastAPI-facing counterpart, distinct
package path so there's no actual name collision).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from docie_bench.schemas.dynamic import DynamicSchemaSpec

from . import _shared

router = APIRouter()


@router.get("/schemas/dynamic")
async def list_dynamic_schemas_route() -> list[dict[str, Any]]:
    """Named ``DynamicSchemaSpec``s saved via the Studio's schema builder --
    what a "Schema name" field can reference beyond ``GET /schemas``' static
    ``SCHEMA_REGISTRY`` names. Missing DATABASE_URL degrades to an empty list,
    same contract as ``GET /datasets``/``GET /model-profiles``.
    """
    from docie_bench.studio.dynamic_schemas import list_dynamic_schemas

    return list_dynamic_schemas()


@router.get("/schemas/dynamic/{name}")
async def get_dynamic_schema_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.dynamic_schemas import get_dynamic_schema

    schema = get_dynamic_schema(name)
    if schema is None:
        raise HTTPException(
            status_code=404, detail=f"schema {name!r} not found", headers=_shared._DOMAIN_404
        )
    return schema


@router.post("/schemas/dynamic", status_code=201)
async def create_dynamic_schema_route(payload: DynamicSchemaSpec) -> dict[str, Any]:
    """Save a ``DynamicSchemaSpec`` under its own ``document_type`` so it can be
    referenced by name afterward, instead of an extraction request needing the
    full field list inline every time.

    ``DynamicSchemaSpec`` is already fully validated (schemas.dynamic) and
    already compiles into a working pydantic model + NuExtract template at
    extraction time -- this route is the missing "define once, reuse by name"
    persistence layer, not new extraction logic. Create-only, matching the
    pipeline/ocr profile routes above: 409 on an existing document_type, no
    in-place update (delete + recreate is cheap here since this is a real
    table, not a text file to splice).
    """
    from docie_bench.studio.dynamic_schemas import (
        DynamicSchemaConflictError,
        DynamicSchemaUnavailableError,
        create_dynamic_schema,
    )

    try:
        return create_dynamic_schema(payload)
    except DynamicSchemaConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DynamicSchemaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.delete("/schemas/dynamic/{name}")
async def delete_dynamic_schema_route(name: str) -> dict[str, Any]:
    from docie_bench.studio.dynamic_schemas import (
        DynamicSchemaNotFoundError,
        DynamicSchemaUnavailableError,
        delete_dynamic_schema,
    )

    try:
        delete_dynamic_schema(name)
    except DynamicSchemaNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc), headers=_shared._DOMAIN_404) from exc
    except DynamicSchemaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"deleted": name}
