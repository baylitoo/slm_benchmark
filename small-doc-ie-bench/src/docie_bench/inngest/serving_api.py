"""Read-only serving endpoints for the DocIE Studio Deploy tab.

Thin wrappers over ``ControlPlane`` (the same facade the ``docie`` CLI drives),
reading the shared serving home (``DOCIE_SERVING_HOME``, a named volume mounted
on both ``api`` and ``worker``). Quick reads live here as plain HTTP; the
long-running *deploy* action is an Inngest function (see ``functions.py`` /
``studio_api.py``).

Caveat: ``deployment_status`` checks process liveness in the *calling* process's
namespace. The runtime is spawned in the ``worker`` container, so liveness read
from the ``api`` container is approximate; static lists (models/runtimes/
deployments) are exact since they come from the shared on-disk state.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import APIRouter, HTTPException

from docie_bench.serving.control_plane import ControlPlane
from docie_bench.settings import get_settings

router = APIRouter(prefix="/v1/serving", tags=["serving"])


@lru_cache(maxsize=1)
def _control_plane() -> ControlPlane:
    return ControlPlane.from_defaults()


@router.get("/models")
async def list_models() -> Any:
    return await _control_plane().list_models()


@router.get("/runtimes")
async def list_runtimes() -> Any:
    return await _control_plane().list_runtimes()


@router.get("/deployments")
async def list_deployments() -> Any:
    return await _control_plane().list_deployments()


@router.get("/deployments/{name}")
async def deployment_status(name: str) -> Any:
    try:
        return await _control_plane().deployment_status(name)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/store")
async def list_store() -> Any:
    """The local GGUF model store (queryable Postgres catalog the Studio reads).

    Each entry includes its family and the backends that can serve it faithfully.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        return ModelCatalog().list()
    except CatalogUnavailableError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.get("/families")
async def list_families() -> Any:
    """Known model families and how each must be served."""
    from docie_bench.serving.model_store import FAMILIES

    return [
        {
            "name": fam.name,
            "vision": fam.vision,
            "needs_mmproj": fam.needs_mmproj,
            "ollama_faithful": fam.ollama_faithful,
            "template_delivery": str(fam.template_delivery),
        }
        for fam in FAMILIES.values()
    ]


@router.get("/benchmarks")
async def list_benchmarks() -> list[dict[str, Any]]:
    """List completed benchmark runs (no ControlPlane method — read runs_dir)."""
    runs_dir = get_settings().runs_dir
    results: list[dict[str, Any]] = []
    if not runs_dir.exists():
        return results
    for entry in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
        if not entry.is_dir():
            continue
        record: dict[str, Any] = {"run": entry.name, "path": str(entry)}
        metrics_path = entry / "metrics.json"
        if metrics_path.exists():
            try:
                record["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record["metrics"] = None
        results.append(record)
    return results


__all__ = ["router"]
