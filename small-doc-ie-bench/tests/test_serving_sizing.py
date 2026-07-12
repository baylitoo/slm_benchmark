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
KV_4096 = KV_CACHE_BYTES_PER_TOKEN * 4096  # 0.25 GiB at the default context


def _fp(size_bytes: int, *, context: int = 4096, mmproj: int = 0) -> int:
    """The PR-2 footprint formula, restated so fits are asserted from first
    principles: weights + KV(ctx) + runtime overhead (+ mmproj)."""
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


def test_double_count_guard_running_rss_never_subtracted_again(tmp_path: Path) -> None:
    """THE §3 trap: the snapshot's free is MEASURED, so a hot deployment's RSS
    is already inside 'used'. Passing live placements must not change a single
    fit number — they inform the display (running_instances), never the
    budget. Subtracting predicted for running instances here would price them
    twice and halve apparent capacity."""
    models = [_model("small", 2 * GIB)]
    hot_placements = [
        {"name": "small", "model_name": "small", "phase": "hot", "rss_bytes": 3 * GIB},
        {"name": "small-2", "model_name": "small", "phase": "loading", "rss_bytes": GIB},
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
    assert with_running.per_model[0].fits_now == without.per_model[0].fits_now == 2
    # Placements surface as display counts only — hot + loading, never cold.
    assert with_running.per_model[0].running_instances == 2
    assert without.per_model[0].running_instances == 0


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
    # 6 GiB fits two 2.75 GiB instances; 6 - 1.6 GiB fits only one.
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

    total = 2 * _fp(4 * GIB)  # 9.5 GiB against a 6 GiB budget
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
def test_whatif_endpoint_404s_an_unknown_model() -> None:
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
    assert excinfo.value.status_code == 404


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
