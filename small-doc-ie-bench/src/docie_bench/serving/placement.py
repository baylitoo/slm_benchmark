"""Best-effort Postgres placement mirror for deploy/stop/load/remove seams.

Extracted from ``control_plane.py`` so the writers of the ``model_placement``
row are enumerable in one place. The reconciler's ``publish_observed`` is the
authority on OBSERVED liveness (state/endpoint/phase) and carries a monotonic
guard against stale snapshots; these helpers exist for row lifecycle —
creation at deploy, retention on stop/unload, deletion on remove — plus the
fast-path ready refresh after a load. All best-effort by design: a missing
DATABASE_URL or a DB hiccup must never fail an operation that already
succeeded against the local process.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("docie_bench.serving.placement")


def record_placement(model_name: str, record: object) -> None:
    """Upsert the catalog placement of a deployed store model (best-effort).

    Symmetric with ``clear_placement``: recording lives at the supervisor seam
    so CLI and job deploys behave identically (the error hint "Deploy it first
    (... `docie up <name>`)" in the placement resolver depends on this).
    Best-effort by design — a missing DATABASE_URL or a DB hiccup must never
    fail a deploy that already succeeded; the deployment is then just not
    discoverable via ``store:<model_name>``.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    spec = getattr(record, "spec", None)
    state = getattr(record, "state", None)
    # Engine follows the launch runtime: llama-server for GGUF, "encoder" for an
    # analyzer snapshot, "transformers" for the AutoModel fallback snapshot
    # (serve_store_model branches by family).
    launch = getattr(spec, "launch", None)
    runtime = getattr(launch, "runtime", None)
    runtime_label = str(getattr(runtime, "value", runtime))
    engine = (
        runtime_label
        if runtime_label in {"encoder", "transformers"}
        else "llama-server"
    )
    try:
        ModelCatalog().record_placement(
            str(getattr(spec, "name", None) or model_name),
            model_name=model_name,
            engine=engine,
            endpoint=str(getattr(record, "endpoint", None) or ""),
            state=str(getattr(state, "value", state) or "unknown"),
        )
    except CatalogUnavailableError:
        logger.warning(
            "no DATABASE_URL: placement for %r not recorded; store:%s will not resolve",
            model_name,
            model_name,
        )
    except Exception:  # noqa: BLE001 - discoverability must not fail the deploy
        logger.warning("could not record catalog placement for %r", model_name, exc_info=True)


def mark_placement_stopped(name: str, *, phase: str = "cold") -> None:
    """UPDATE the placement of a stopped deployment to ``phase``/"" (best-effort).

    The stop/unload-side sibling of ``record_placement``: keeps the row
    (deletion is ``remove``'s job only, design fix #3) while making it
    non-routable (``endpoint=""``). ``phase="cold"`` for a user Stop,
    ``"evicted"`` for the PR-4 unload path. Best-effort: no DATABASE_URL / a
    DB hiccup must never block stopping a local process.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        ModelCatalog().mark_placement_stopped(name, phase=phase)
    except CatalogUnavailableError:
        pass  # no DATABASE_URL -> nothing was ever recorded; nothing to update
    except Exception:  # noqa: BLE001 - staleness cleanup must not fail the stop
        logger.warning("could not mark catalog placement stopped for %r", name, exc_info=True)


def mark_placement_ready(name: str, *, endpoint: str) -> None:
    """UPDATE an existing placement row to ready/hot after a load (best-effort).

    The load-side sibling of ``mark_placement_stopped``: closes the
    up-to-one-cycle window where a just-reloaded store model's row still says
    ``evicted``/``endpoint=""`` and ``store:<name>`` refuses to route. The
    reconciler's ``publish_observed`` carries a monotonic guard, so a cycle
    snapshot taken BEFORE this write can no longer clobber it back to
    evicted. Never creates a row (the reconciler owns creation); no
    DATABASE_URL / a DB hiccup must never fail a load that already succeeded.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        ModelCatalog().mark_placement_ready(name, endpoint=endpoint)
    except CatalogUnavailableError:
        pass  # no DATABASE_URL -> nothing was ever recorded; nothing to update
    except Exception:  # noqa: BLE001 - discoverability must not fail the load
        logger.warning("could not mark catalog placement ready for %r", name, exc_info=True)


def clear_placement(name: str) -> None:
    """DELETE the catalog placement of a REMOVED deployment (best-effort).

    PR-1: this is the only path that deletes a placement row — ``stop`` now
    UPDATEs via ``mark_placement_stopped`` instead (design fix #3).
    Best-effort by design: a missing DATABASE_URL or a DB hiccup must never
    block tearing down a local process.
    """
    from docie_bench.serving.catalog import CatalogUnavailableError, ModelCatalog

    try:
        ModelCatalog().clear_placement(name)
    except CatalogUnavailableError:
        pass  # no DATABASE_URL -> nothing was ever recorded; nothing to clear
    except Exception:  # noqa: BLE001 - staleness cleanup must not fail the stop
        logger.warning("could not clear catalog placement for %r", name, exc_info=True)


__all__ = [
    "clear_placement",
    "mark_placement_ready",
    "mark_placement_stopped",
    "record_placement",
]
