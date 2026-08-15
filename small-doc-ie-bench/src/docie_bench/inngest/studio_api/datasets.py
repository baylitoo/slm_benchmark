"""Registered-dataset listing + validation routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


@router.get("/datasets")
async def list_datasets() -> list[dict[str, Any]]:
    """Registered benchmark datasets (``data/datasets.yaml``) — what a "Dataset"
    field in the Studio can actually reference, so the Benchmark form can offer
    a picker instead of a free-text guess.

    Each entry is a name (the ``dataset`` value ``/benchmark`` and
    ``run_benchmark`` resolve), its latest version, all known versions, and the
    latest version's document count. Missing registry file degrades to an
    empty list (``load_registry``'s own contract), never a 500.
    """
    from docie_bench.benchmark.registry import DEFAULT_REGISTRY_PATH, load_registry

    registry = load_registry(DEFAULT_REGISTRY_PATH)
    return [
        {
            "name": name,
            "description": record.description,
            "latest": record.latest,
            "versions": sorted(record.versions),
            "documents": (
                record.versions[record.latest].statistics.get("documents")
                if record.latest and record.latest in record.versions
                else None
            ),
        }
        for name, record in sorted(registry.datasets.items())
    ]


@router.post("/datasets/{name}/validate")
async def validate_dataset_version(
    name: str,
    version: str | None = None,
    near_duplicate_threshold: float = Query(default=0.92, ge=0.0, le=1.0),
) -> dict[str, Any]:
    """Run the CLI's dataset validation (`docie-bench dataset validate`/`inspect`) against
    an already-registered dataset version, reachable from the Studio.

    Resolves the reference through the shared `data/datasets.yaml` registry (this Studio
    API already reads that file directly for `GET /datasets`, so this isn't a new
    filesystem assumption). Unlike either CLI command, this deliberately does NOT use
    `resolve_dataset`'s own `verify_hash` gate (which raises and stops before validation
    runs) -- passing the registry's recorded hash as `expected_hash` instead lets a pure
    drift case (files changed on disk since registration, nothing else wrong) surface
    inside the SAME report as leakage/statistics, rather than a bare error.

    This does NOT mean drift and structural problems always merge into one report:
    `validate_dataset` (registry.py) early-returns as soon as any duplicate-doc_id /
    missing-file / unsupported-suffix error exists, before it ever reaches the hash
    comparison -- same short-circuit it's always had. If a dataset has both a structural
    error and real drift, only the structural error is reported; the hash is never
    checked in that case. See
    test_validate_endpoint_reports_corruption_introduced_after_registration.

    `resolve_dataset` itself still unconditionally hashes every referenced file (even
    with verify_hash=False) -- so a manifest referencing an ENTIRELY missing file raises
    OSError there, before validate_dataset's own graceful per-item existence check ever
    runs (the CLI's `dataset_validate` has this identical gap, see cli.py:363's shared
    `except (OSError, ValueError)`). Surfaced as 422 (the registered dataset itself is
    broken) rather than a raw 500.
    """
    from docie_bench.benchmark.registry import (
        DEFAULT_REGISTRY_PATH,
        load_registry,
        resolve_dataset,
        validate_dataset,
    )

    reference = f"{name}@{version}" if version else name
    try:
        resolved = resolve_dataset(
            reference, registry_path=DEFAULT_REGISTRY_PATH, verify_hash=False
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    expected_hash = None
    if resolved.version is not None:
        record = load_registry(DEFAULT_REGISTRY_PATH).datasets.get(name)
        entry = record.versions.get(resolved.version) if record else None
        expected_hash = entry.dataset_hash if entry else None
    report = validate_dataset(
        resolved.manifest_path,
        near_duplicate_threshold=near_duplicate_threshold,
        expected_hash=expected_hash,
    )
    return {
        # No manifest_path here: an absolute server-side filesystem path, unlike every
        # other artifact this Studio API exposes (see download_artifact's own explicit
        # principle -- resolved via id -> DB row -> shared blob store, never a
        # worker-local path). The CLI's dataset_inspect prints it because it IS the
        # local tool; a network API shouldn't.
        "reference": resolved.reference,
        "version": resolved.version,
        **report,
    }
