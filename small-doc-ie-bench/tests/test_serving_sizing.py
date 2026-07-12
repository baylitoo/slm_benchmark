"""PR-3: sizing engine + endpoints.

Stub-tested end to end: a fixed node snapshot + store list drive the pure
engine deterministically; sqlite stands in for Postgres on the endpoint tests.
Honest limits: these stubs cannot verify the live claim that "deploy N
instances until the bar predicts 0 → the (N+1)th deploy fails the fit gate" —
that is the PR's live-verification item.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.resources import (
    DEFAULT_DEPLOY_CONTEXT_LENGTH,
    KV_CACHE_BYTES_PER_TOKEN,
    RUNTIME_OVERHEAD_BYTES,
    FootprintStore,
)
from docie_bench.serving.sizing import (
    UnknownModelError,
    UnpriceableModelError,
    compute_sizing,
    compute_whatif,
    safety_margin_bytes,
)

GIB = 1024**3


def _fp(size_bytes: int, *, context: int = DEFAULT_DEPLOY_CONTEXT_LENGTH, mmproj: int = 0) -> int:
    """The PR-2 footprint formula, restated so fits are asserted from first
    principles: weights + KV(ctx) + runtime overhead (+ mmproj). The default
    context is the DEPLOY default (8192) — what the engine must price when a
    caller specifies none."""
    return size_bytes + KV_CACHE_BYTES_PER_TOKEN * context + RUNTIME_OVERHEAD_BYTES + mmproj


def _model(name: str, size_bytes: int | None, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "family": "openai_chat",
        "size_bytes": size_bytes,
        "model_path": f"/store/{name}/model.gguf",
        "mmproj_path": None,
        **extra,
    }


def _snapshot(
    *, total: int = 16 * GIB, free: int = 6 * GIB, source: str = "cgroup"
) -> dict[str, Any]:
    return {"total_bytes": total, "free_bytes": free, "source": source, "sum_rss_bytes": 0}


def _footprints(tmp_path: Path) -> FootprintStore:
    return FootprintStore(home=tmp_path / "serving-home")


# ------------------------------------------------------------------- fit table


def test_fit_table_is_deterministic_from_snapshot_and_store(tmp_path: Path) -> None:
    """Fixed snapshot + store list => exact fits_now, straight from the design
    formula fits = floor((free - margin) / footprint)."""
    models = [_model("small", 2 * GIB), _model("big", 4 * GIB)]

    report = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )

    assert report.observed_available is True
    assert report.free_effective_bytes == 6 * GIB  # margin 0: budget == measured free
    by_name = {fit.name: fit for fit in report.per_model}
    assert by_name["small"].footprint_bytes == _fp(2 * GIB)
    assert by_name["big"].footprint_bytes == _fp(4 * GIB)
    assert by_name["small"].fits_now == (6 * GIB) // _fp(2 * GIB) == 2
    assert by_name["big"].fits_now == (6 * GIB) // _fp(4 * GIB) == 1
    assert by_name["small"].calibrated_bytes is None  # never run: formula only


def test_sizing_defaults_to_the_deploy_context_not_llama_servers_4096(
    tmp_path: Path,
) -> None:
    """The fit table and what-if price prospective DEPLOYS, and every deploy
    path defaults context_length to 8192 (studio DeployRequest, the deploy
    handler, ControlPlane.up/serve_store_model, the CLI). Pricing at
    llama-server's bare 4096 fallback would under-price every default deploy
    by 65536 x 4096 = 256 MiB of KV per instance and overstate fits_now."""
    from docie_bench.inngest.studio_api import DeployRequest

    # Single source of truth: the deploy surfaces and the sizing engine must
    # share the constant, so they can never drift apart again.
    assert DEFAULT_DEPLOY_CONTEXT_LENGTH == 8192
    assert DeployRequest.model_fields["context_length"].default == DEFAULT_DEPLOY_CONTEXT_LENGTH

    models = [_model("small", 2 * GIB)]
    report = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    assert report.context_length == DEFAULT_DEPLOY_CONTEXT_LENGTH
    assert report.per_model[0].footprint_bytes == _fp(2 * GIB, context=8192)
    # 8192-token KV is 256 MiB MORE than the 4096 default would have priced.
    assert _fp(2 * GIB, context=8192) - _fp(2 * GIB, context=4096) == 256 * 1024**2

    # What-if items with no explicit context price at the same deploy default.
    whatif = compute_whatif(
        models,
        _snapshot(),
        [{"model": "small", "instances": 1}],
        footprints=_footprints(tmp_path),
        margin_fraction=0.0,
    )
    assert whatif.per_item[0].context_length == DEFAULT_DEPLOY_CONTEXT_LENGTH
    assert whatif.per_item[0].footprint_bytes == _fp(2 * GIB, context=8192)


def test_double_count_guard_hot_rss_never_subtracted_again(tmp_path: Path) -> None:
    """THE §3 trap: the snapshot's free is MEASURED, so a HOT deployment's
    steady RSS is already inside 'used'. Passing hot/cold placements must not
    change a single fit number — they inform the display (running_instances),
    never the budget. Subtracting predicted for hot instances here would price
    them twice and halve apparent capacity. (Loading placements are the one
    deliberate exception — see the reservation test below.)"""
    models = [_model("small", 2 * GIB)]
    hot_placements = [
        {"name": "small", "model_name": "small", "phase": "hot", "rss_bytes": 3 * GIB},
        {"name": "old", "model_name": "small", "phase": "cold", "rss_bytes": 0},
    ]

    without = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    with_running = compute_sizing(
        models,
        _snapshot(),
        hot_placements,
        footprints=_footprints(tmp_path),
        margin_fraction=0.0,
    )

    assert with_running.free_effective_bytes == without.free_effective_bytes
    assert with_running.loading_reserved_bytes == 0
    assert with_running.per_model[0].fits_now == without.per_model[0].fits_now == 2
    # Placements surface as display counts only — hot, never cold.
    assert with_running.per_model[0].running_instances == 1
    assert without.per_model[0].running_instances == 0


def test_loading_placement_reserves_its_unfaulted_pages(tmp_path: Path) -> None:
    """The mmap-ramp exception to the double-count guard: a LOADING runtime's
    RSS only covers the pages faulted in so far, so measured free overstates
    capacity by (footprint - rss) until it goes hot. The engine must reserve
    exactly that remainder — no more (the resident part is already in 'used'),
    and nothing at all once the placement is hot."""
    models = [_model("small", 2 * GIB)]
    footprint = _fp(2 * GIB)  # 3 GiB at the deploy-default context
    loading = [
        {"name": "small-1", "model_name": "small", "phase": "loading", "rss_bytes": 1 * GIB},
    ]

    report = compute_sizing(
        models, _snapshot(), loading, footprints=_footprints(tmp_path), margin_fraction=0.0
    )

    reserve = footprint - 1 * GIB  # only the not-yet-resident remainder
    assert report.loading_reserved_bytes == reserve
    assert report.free_effective_bytes == 6 * GIB - reserve
    assert report.per_model[0].fits_now == (6 * GIB - reserve) // footprint == 1
    assert report.per_model[0].running_instances == 1  # display still counts it

    # RSS already past the footprint (calibration drift): reserve clamps at 0,
    # never a negative reservation inflating the budget.
    over = [
        {"name": "small-1", "model_name": "small", "phase": "loading", "rss_bytes": 4 * GIB},
    ]
    clamped = compute_sizing(
        models, _snapshot(), over, footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    assert clamped.loading_reserved_bytes == 0
    assert clamped.free_effective_bytes == 6 * GIB

    # A loading placement whose model is unknown/unpriceable reserves nothing
    # (fail-open, the fit gate's rule for unknowables).
    unknown = [
        {"name": "ghost", "model_name": "gone", "phase": "loading", "rss_bytes": 0},
    ]
    unpriced = compute_sizing(
        models, _snapshot(), unknown, footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    assert unpriced.loading_reserved_bytes == 0


def test_whatif_reserves_loading_placements_like_the_fit_table(tmp_path: Path) -> None:
    """What-if and the fit table must judge against the SAME budget: a plan
    checked mid-load of another model has to see the loader's not-yet-resident
    remainder held back, or Check-fit approves a mix the node cannot hold."""
    models = [_model("small", 2 * GIB), _model("big", 4 * GIB)]
    footprint_big = _fp(4 * GIB)  # 5 GiB
    loading = [
        {"name": "big-1", "model_name": "big", "phase": "loading", "rss_bytes": 1 * GIB},
    ]
    plan = [{"model": "small", "instances": 1}]

    report = compute_whatif(
        models,
        _snapshot(),  # 6 GiB free
        plan,
        loading,
        footprints=_footprints(tmp_path),
        margin_fraction=0.0,
    )

    reserve = footprint_big - 1 * GIB  # 4 GiB still owed to the loader
    assert report.loading_reserved_bytes == reserve
    assert report.free_effective_bytes == 6 * GIB - reserve  # 2 GiB budget
    assert report.ok is False  # small needs 3 GiB: does NOT fit mid-load
    assert report.deficit_bytes == _fp(2 * GIB) - (6 * GIB - reserve)

    # Same plan with the loader hot instead: RSS fully measured, no reserve.
    hot = [{"name": "big-1", "model_name": "big", "phase": "hot", "rss_bytes": 5 * GIB}]
    settled = compute_whatif(
        models, _snapshot(), plan, hot, footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    assert settled.loading_reserved_bytes == 0
    assert settled.ok is True


def test_safety_margin_is_honored_and_surfaced(tmp_path: Path) -> None:
    """The margin is real math (a slice of TOTAL held back before pricing) and
    a visible number — never hidden padding."""
    models = [_model("small", 2 * GIB)]

    report = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.10
    )

    margin = safety_margin_bytes(16 * GIB, 0.10)
    assert report.safety_margin_bytes == margin == int(16 * GIB * 0.10)
    assert report.free_effective_bytes == 6 * GIB - margin
    # 6 GiB fits two 3 GiB instances; 6 - 1.6 GiB fits only one.
    assert report.per_model[0].fits_now == (6 * GIB - margin) // _fp(2 * GIB) == 1
    assert report.margin_fraction == 0.10

    with pytest.raises(ValueError, match="margin_fraction"):
        compute_sizing(models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=1.0)


def test_calibrated_steady_rss_lifts_the_footprint(tmp_path: Path) -> None:
    """max(calibrated, predicted): a model measured to need more than its
    formula must be priced at the measurement (the whole point of PR-2
    calibration), keyed by the store entry's model_path."""
    footprints = _footprints(tmp_path)
    models = [_model("small", 2 * GIB)]
    footprints.record_steady("/store/small/model.gguf", 5 * GIB)

    report = compute_sizing(models, _snapshot(), footprints=footprints, margin_fraction=0.0)

    fit = report.per_model[0]
    assert fit.predicted_bytes == _fp(2 * GIB)
    assert fit.calibrated_bytes == 5 * GIB
    assert fit.footprint_bytes == 5 * GIB  # measurement wins over the formula
    assert fit.fits_now == (6 * GIB) // (5 * GIB) == 1


def test_mmproj_is_priced_into_the_footprint(tmp_path: Path) -> None:
    """Vision families load the projector fully resident: the fit table must
    price it (mmproj-aware, same rule as the reconciler's restart gate)."""
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"p" * 4096)
    models = [_model("vl", 2 * GIB, mmproj_path=str(mmproj))]

    report = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )

    assert report.per_model[0].footprint_bytes == _fp(2 * GIB, mmproj=4096)

    # Unreadable projector degrades to 0 (fail-open), never refuses to price.
    missing = [_model("vl", 2 * GIB, mmproj_path=str(tmp_path / "gone.gguf"))]
    degraded = compute_sizing(
        missing, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )
    assert degraded.per_model[0].footprint_bytes == _fp(2 * GIB)


def test_unpriceable_model_is_honest_not_zero(tmp_path: Path) -> None:
    models = [_model("mystery", None)]  # no size_bytes, model_path unreadable

    report = compute_sizing(
        models, _snapshot(), footprints=_footprints(tmp_path), margin_fraction=0.0
    )

    fit = report.per_model[0]
    assert fit.footprint_bytes is None
    assert fit.fits_now is None
    assert fit.detail is not None
    assert "unpriceable" in fit.detail


def test_no_snapshot_degrades_honestly(tmp_path: Path) -> None:
    """No node snapshot: footprints still price (store-only math) but fits stay
    unknown — never a locally-measured or fabricated free number."""
    models = [_model("small", 2 * GIB)]

    report = compute_sizing(models, None, footprints=_footprints(tmp_path), margin_fraction=0.0)

    assert report.observed_available is False
    assert report.detail == "no node snapshot published"
    assert report.free_effective_bytes is None
    fit = report.per_model[0]
    assert fit.footprint_bytes == _fp(2 * GIB)
    assert fit.fits_now is None


# --------------------------------------------------------------------- what-if


def test_whatif_fits_with_exact_remaining(tmp_path: Path) -> None:
    models = [_model("small", 2 * GIB), _model("big", 4 * GIB)]
    plan = [{"model": "small", "instances": 1}, {"model": "big", "instances": 1}]

    report = compute_whatif(
        models,
        _snapshot(free=10 * GIB),
        plan,
        footprints=_footprints(tmp_path),
        margin_fraction=0.0,
    )

    total = _fp(2 * GIB) + _fp(4 * GIB)
    assert report.total_predicted_bytes == total
    assert report.remaining_bytes == 10 * GIB - total
    assert report.ok is True
    assert report.deficit_bytes == 0
    assert [item.subtotal_bytes for item in report.per_item] == [_fp(2 * GIB), _fp(4 * GIB)]


def test_whatif_deficit_math_is_exact(tmp_path: Path) -> None:
    """An over-committed plan reports HOW MUCH is missing, not just a no."""
    models = [_model("big", 4 * GIB)]
    plan = [{"model": "big", "instances": 2}]

    report = compute_whatif(
        models, _snapshot(free=6 * GIB), plan, footprints=_footprints(tmp_path), margin_fraction=0.0
    )

    total = 2 * _fp(4 * GIB)  # 10 GiB against a 6 GiB budget
    assert report.total_predicted_bytes == total
    assert report.ok is False
    assert report.remaining_bytes == 6 * GIB - total
    assert report.deficit_bytes == total - 6 * GIB


def test_whatif_respects_per_item_context_and_calibration(tmp_path: Path) -> None:
    """The what-if math matches the deploy path: per-item context scales the
    KV term; a calibrated model prices at its measured steady RSS."""
    footprints = _footprints(tmp_path)
    footprints.record_steady("/store/small/model.gguf", 5 * GIB)
    models = [_model("small", 2 * GIB), _model("big", 4 * GIB)]
    plan = [
        {"model": "big", "instances": 1, "context_length": 8192},
        {"model": "small", "instances": 1},
    ]

    report = compute_whatif(
        models, _snapshot(free=12 * GIB), plan, footprints=footprints, margin_fraction=0.0
    )

    big_item, small_item = report.per_item
    assert big_item.footprint_bytes == _fp(4 * GIB, context=8192)
    assert big_item.calibrated is False
    assert small_item.footprint_bytes == 5 * GIB  # calibration wins
    assert small_item.calibrated is True


def test_whatif_rejects_unknown_and_unpriceable_models(tmp_path: Path) -> None:
    models = [_model("small", 2 * GIB), _model("mystery", None)]

    with pytest.raises(UnknownModelError, match="nope"):
        compute_whatif(
            models, _snapshot(), [{"model": "nope"}], footprints=_footprints(tmp_path)
        )
    with pytest.raises(UnpriceableModelError, match="mystery"):
        compute_whatif(
            models, _snapshot(), [{"model": "mystery"}], footprints=_footprints(tmp_path)
        )
    with pytest.raises(ValueError, match="instances"):
        compute_whatif(
            models,
            _snapshot(),
            [{"model": "small", "instances": 0}],
            footprints=_footprints(tmp_path),
        )


def test_whatif_without_snapshot_prices_but_never_judges(tmp_path: Path) -> None:
    models = [_model("small", 2 * GIB)]

    report = compute_whatif(
        models, None, [{"model": "small", "instances": 2}], footprints=_footprints(tmp_path)
    )

    assert report.observed_available is False
    assert report.total_predicted_bytes == 2 * _fp(2 * GIB)
    assert report.ok is None  # honest: no measured budget to judge against
    assert report.remaining_bytes is None
    assert report.deficit_bytes is None


# ------------------------------------------------------------------- endpoints


@pytest.fixture
def _serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate the default FootprintStore the endpoints construct."""
    home = tmp_path / "serving-home"
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    return home


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


def _seed_store(name: str, size_bytes: int) -> None:
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.model_store import StoreEntry

    ModelCatalog().upsert(
        StoreEntry(
            name=name, family="openai_chat", model_path=Path(f"/store/{name}/model.gguf")
        ),
        size_bytes=size_bytes,
    )


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_sizing_endpoint_serves_fit_table_from_observed_surface() -> None:
    from docie_bench.inngest.serving_api import serving_sizing
    from docie_bench.serving.catalog import ModelCatalog

    _seed_store("small", 2 * GIB)
    ModelCatalog().publish_node_snapshot(
        total_bytes=16 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=0
    )

    payload = asyncio.run(serving_sizing())

    assert payload["observed_available"] is True
    assert payload["detail"] is None
    assert payload["node"]["total_bytes"] == 16 * GIB
    assert payload["source"] == "cgroup"
    # Default margin: the configurable settings knob (10% of total).
    margin = int(16 * GIB * 0.10)
    assert payload["safety_margin_bytes"] == margin
    assert payload["assumptions"]["margin_fraction"] == 0.10
    (fit,) = payload["per_model"]
    assert fit["name"] == "small"
    assert fit["footprint_bytes"] == _fp(2 * GIB)
    assert fit["fits_now"] == (6 * GIB - margin) // _fp(2 * GIB) == 1
    assert fit["calibrated"] is False


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_sizing_endpoint_honest_when_snapshot_never_published() -> None:
    from docie_bench.inngest.serving_api import serving_sizing

    _seed_store("small", 2 * GIB)

    payload = asyncio.run(serving_sizing())

    assert payload["observed_available"] is False
    assert "no node snapshot published yet" in payload["detail"]
    assert payload["node"] is None
    (fit,) = payload["per_model"]
    assert fit["footprint_bytes"] == _fp(2 * GIB)  # still priced from the store
    assert fit["fits_now"] is None  # fit honestly unknown


@pytest.mark.usefixtures("_serving_home")
def test_sizing_endpoint_honest_when_database_down() -> None:
    from docie_bench.inngest.serving_api import serving_sizing

    db.dispose_engine()
    payload = asyncio.run(serving_sizing())

    assert payload["observed_available"] is False
    assert "DATABASE_URL" in payload["detail"]
    assert payload["per_model"] == []


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_whatif_endpoint_matches_the_fit_table_math() -> None:
    from docie_bench.inngest.serving_api import (
        WhatIfPlanItem,
        WhatIfRequest,
        serving_sizing_whatif,
    )
    from docie_bench.serving.catalog import ModelCatalog

    _seed_store("small", 2 * GIB)
    ModelCatalog().publish_node_snapshot(
        total_bytes=16 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=0
    )
    margin = int(16 * GIB * 0.10)

    ok_payload = asyncio.run(
        serving_sizing_whatif(WhatIfRequest(plan=[WhatIfPlanItem(model="small")]))
    )
    assert ok_payload["ok"] is True
    assert ok_payload["remaining_bytes"] == (6 * GIB - margin) - _fp(2 * GIB)

    deficit_payload = asyncio.run(
        serving_sizing_whatif(
            WhatIfRequest(plan=[WhatIfPlanItem(model="small", instances=3)])
        )
    )
    assert deficit_payload["ok"] is False
    assert deficit_payload["deficit_bytes"] == 3 * _fp(2 * GIB) - (6 * GIB - margin)


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_whatif_endpoint_422s_an_unknown_model_never_404() -> None:
    """404 is reserved by the frontend's global convention for 'endpoint does
    not exist yet' (api.ts isUnavailableStatus maps it to ApiUnavailable), so a
    store-removal racing the UI poll must come back as a domain error (422)
    carrying the server's detail — not as a bogus 'endpoint unavailable'."""
    from docie_bench.inngest.serving_api import (
        WhatIfPlanItem,
        WhatIfRequest,
        serving_sizing_whatif,
    )

    _seed_store("small", 2 * GIB)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            serving_sizing_whatif(WhatIfRequest(plan=[WhatIfPlanItem(model="nope")]))
        )
    assert excinfo.value.status_code == 422
    assert "unknown store model" in str(excinfo.value.detail)


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_sizing_and_resources_flag_a_stale_snapshot_as_unavailable() -> None:
    """Staleness gate: if the serving reconciler dies, its last snapshot must
    not keep backing observed_available=true forever — /resources promises
    'never a stale number' and /sizing must not compute fits against a dead
    reading. Older than 3x the reconcile interval => the same honest degraded
    state as never-published, with the age in the detail."""
    import datetime as dt

    from docie_bench.inngest.serving_api import (
        serving_resources,
        serving_sizing,
        snapshot_stale_after_s,
    )
    from docie_bench.serving.catalog import SERVING_NODE_ROW_ID, ModelCatalog, ServingNode
    from docie_bench.storage.db import session_scope

    _seed_store("small", 2 * GIB)
    ModelCatalog().publish_node_snapshot(
        total_bytes=16 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=0
    )
    # Age the published stamp past the gate (the reconciler "died" 10x ago).
    aged = dt.datetime.now(dt.UTC) - dt.timedelta(
        seconds=10 * snapshot_stale_after_s()
    )
    with session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, SERVING_NODE_ROW_ID)
        assert row is not None
        row.updated_at = aged

    sizing_payload = asyncio.run(serving_sizing())
    assert sizing_payload["observed_available"] is False
    assert "stale" in sizing_payload["detail"]
    assert sizing_payload["node"] is None  # the dead number is not served
    (fit,) = sizing_payload["per_model"]
    assert fit["footprint_bytes"] == _fp(2 * GIB)  # still priced from the store
    assert fit["fits_now"] is None  # never a fit against a dead reading

    resources_payload = asyncio.run(serving_resources())
    assert resources_payload["observed_available"] is False
    assert resources_payload["node"] is None
    assert "stale" in resources_payload["detail"]


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_sizing_endpoint_reserves_loading_placements() -> None:
    """End to end through the observed surface: a placement the reconciler
    published as 'loading' reduces the deployable budget by its not-yet-
    resident remainder (footprint - rss)."""
    from docie_bench.inngest.serving_api import serving_sizing
    from docie_bench.serving.catalog import ModelCatalog

    _seed_store("small", 2 * GIB)
    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        total_bytes=16 * GIB, free_bytes=6 * GIB, source="cgroup", sum_rss_bytes=GIB
    )
    catalog.publish_observed(
        "small",  # placement name == store name => model_name auto-linked
        engine="llama-server",
        state="starting",
        endpoint="",
        phase="loading",
        pid=42,
        pid_create_time=1.0,
        rss_bytes=GIB,
        health_ok=False,
        last_error=None,
    )

    payload = asyncio.run(serving_sizing())

    margin = int(16 * GIB * 0.10)
    reserve = _fp(2 * GIB) - GIB
    assert payload["loading_reserved_bytes"] == reserve
    assert payload["free_effective_bytes"] == 6 * GIB - margin - reserve
    (fit,) = payload["per_model"]
    assert fit["running_instances"] == 1
    assert fit["fits_now"] == max((6 * GIB - margin - reserve) // _fp(2 * GIB), 0) == 0


@pytest.mark.usefixtures("_sqlite_catalog", "_serving_home")
def test_store_endpoint_strips_container_paths() -> None:
    """model_path/mmproj_path are server-side sizing inputs (calibration key +
    projector pricing) — the unauthenticated /store surface must not echo
    container filesystem paths to the browser."""
    from docie_bench.inngest.serving_api import list_store
    from docie_bench.serving.catalog import ModelCatalog

    _seed_store("small", 2 * GIB)
    # The catalog view itself still carries the paths for the sizing engine.
    (view,) = ModelCatalog().list()
    assert view["model_path"] == "/store/small/model.gguf"

    (entry,) = asyncio.run(list_store())

    assert "model_path" not in entry
    assert "mmproj_path" not in entry
    assert entry["name"] == "small"  # the rest of the view is untouched
    assert entry["size_bytes"] == 2 * GIB


@pytest.mark.usefixtures("_serving_home")
def test_whatif_endpoint_503s_when_database_down() -> None:
    from docie_bench.inngest.serving_api import (
        WhatIfPlanItem,
        WhatIfRequest,
        serving_sizing_whatif,
    )

    db.dispose_engine()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            serving_sizing_whatif(WhatIfRequest(plan=[WhatIfPlanItem(model="small")]))
        )
    assert excinfo.value.status_code == 503
    assert "DATABASE_URL" in str(excinfo.value.detail)
