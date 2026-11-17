"""Capacity on a fleet (#664 D1-D6, A5, B5): sizing, admission and fit badges
price against the node a deploy lands on, using what THAT node measured."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import StoreEntry
from docie_bench.serving.sizing import compute_sizing
from docie_bench.settings import get_settings

GIB = 1024**3
MODEL = "lfm2.5-350m"
VISION = "nuextract3"


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _catalog_entry(name: str, size_bytes: int = GIB, mmproj: bool = False) -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=name,
            family="openai_chat",
            model_path=Path(f"/models/{name}/model.gguf"),
            mmproj_path=Path(f"/models/{name}/mmproj.gguf") if mmproj else None,
        ),
        size_bytes=size_bytes,
    )


def _node(node_id: str, free: int, **extra: object) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=16 * GIB,
        free_bytes=free,
        source="cgroup",
        sum_rss_bytes=GIB,
        **extra,  # type: ignore[arg-type]
    )


# ------------------------------------------------------------------ B5


def test_published_footprints_round_trip_through_the_node_row(fleet: None) -> None:
    _node(
        "rocky-llm",
        8 * GIB,
        model_footprints={VISION: {"steady_rss_bytes": 3 * GIB, "mmproj_bytes": GIB}},
    )
    (row,) = ModelCatalog().list_node_snapshots()
    assert row["model_footprints"] == {
        VISION: {"steady_rss_bytes": 3 * GIB, "mmproj_bytes": GIB}
    }


def test_sizing_prices_a_remote_nodes_calibration_and_projector() -> None:
    """The api has no sidecar and no mmproj file for a remote node's model; the
    node's published numbers must still reach the fit."""
    models = [
        {
            "name": VISION,
            "family": "openai_chat",
            "model_path": f"/models/{VISION}/model.gguf",
            "mmproj_path": f"/models/{VISION}/mmproj.gguf",
            "size_bytes": GIB,
        }
    ]
    bare = {"total_bytes": 16 * GIB, "free_bytes": 8 * GIB, "source": "cgroup"}
    published = {
        **bare,
        "model_footprints": {
            VISION: {"steady_rss_bytes": 5 * GIB, "mmproj_bytes": 2 * GIB}
        },
    }

    class NoLocalSidecars:
        def get(self, model: str) -> int | None:
            return None

    (before,) = compute_sizing(models, bare, footprints=NoLocalSidecars()).per_model
    (after,) = compute_sizing(models, published, footprints=NoLocalSidecars()).per_model
    assert before.calibrated_bytes is None
    assert after.calibrated_bytes == 5 * GIB
    assert after.predicted_bytes is not None
    assert before.predicted_bytes is not None
    assert after.predicted_bytes - before.predicted_bytes == 2 * GIB


def test_node_publishes_its_own_footprints(
    fleet: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench.serving import resources

    model_dir = tmp_path / "blobs" / VISION
    model_dir.mkdir(parents=True)
    weights = model_dir / "model.gguf"
    weights.write_bytes(b"w")
    projector = model_dir / "mmproj.gguf"
    projector.write_bytes(b"p" * 1234)
    entry = StoreEntry(
        name=VISION, family="openai_chat", model_path=weights, mmproj_path=projector
    )

    class Store:
        def __init__(self, root: object) -> None:
            pass

        def list(self) -> list[StoreEntry]:
            return [entry]

    monkeypatch.setattr("docie_bench.serving.model_store.ModelStore", Store)
    resources.FootprintStore().record_steady(weights.as_posix(), 3 * GIB)

    assert resources._local_model_footprints() == {
        VISION: {"steady_rss_bytes": 3 * GIB, "mmproj_bytes": 1234}
    }


# ------------------------------------------------------- D1 / D2 / D3 / D4


def _place(name: str, node_id: str, *, model: str = MODEL, port: int = 8090) -> None:
    ModelCatalog().record_placement(
        name,
        model_name=model,
        engine="llama-server",
        endpoint=f"http://10.8.0.{2 if node_id == 'rocky-llm' else 3}:{port}/v1",
        state="running",
        node_id=node_id,
    )


def test_sizing_prices_the_chosen_node_not_a_withheld_fleet(fleet: None) -> None:
    import asyncio

    from docie_bench.inngest.serving_api import serving_sizing

    _catalog_entry(MODEL)
    _node("rocky-llm", 8 * GIB)
    _node("ovh-cpu-02", 2 * GIB)

    default = asyncio.run(serving_sizing())
    picked = asyncio.run(serving_sizing(node_id="ovh-cpu-02"))

    assert default["nodes"] == ["ovh-cpu-02", "rocky-llm"]
    assert default["node_id"] == "ovh-cpu-02"
    assert picked["node_id"] == "ovh-cpu-02"
    assert picked["free_bytes"] == 2 * GIB
    rocky = asyncio.run(serving_sizing(node_id="rocky-llm"))
    assert rocky["free_bytes"] == 8 * GIB
    (fit,) = rocky["per_model"]
    assert fit["fits_now"] is not None
    assert fit["fits_now"] > 0


def test_targeted_deploy_is_refused_when_its_node_cannot_fit(fleet: None) -> None:
    from fastapi import HTTPException

    from docie_bench.inngest.studio_api.deploy import DeployRequest, _target_node_deploy

    _catalog_entry(MODEL, size_bytes=6 * GIB)
    _node("rocky-llm", 16 * GIB, store_models=[MODEL])
    _node("ovh-cpu-02", 2 * GIB, store_models=[MODEL])

    with pytest.raises(HTTPException) as refused:
        _target_node_deploy(
            DeployRequest(model=MODEL, node_id="ovh-cpu-02"), object()  # type: ignore[arg-type]
        )
    assert refused.value.status_code == 422
    assert "ovh-cpu-02" in refused.value.detail
    assert ModelCatalog().desired_snapshot("ovh-cpu-02")["deployments"] == []

    _target_node_deploy(
        DeployRequest(model=MODEL, node_id="rocky-llm"), object()  # type: ignore[arg-type]
    )
    assert len(ModelCatalog().desired_snapshot("rocky-llm")["deployments"]) == 1


def test_untargeted_replicas_are_refused_only_when_the_fleet_cannot_fit(
    fleet: None,
) -> None:
    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import admit_replica_ram

    _catalog_entry(MODEL, size_bytes=6 * GIB)
    _node("rocky-llm", 9 * GIB, store_models=[MODEL])
    _node("ovh-cpu-02", 9 * GIB, store_models=[MODEL])

    admit_replica_ram(MODEL, 2, None)  # one per node fits
    with pytest.raises(HTTPException) as refused:
        admit_replica_ram(MODEL, 3, None)
    assert refused.value.status_code == 422
    assert "fleet" in refused.value.detail


def test_fit_badge_uses_the_best_node_and_counts_fitting_nodes(fleet: None) -> None:
    from docie_bench.inngest.studio_api import _annotate_fits
    from docie_bench.inngest.studio_api.deploy import _node_budgets

    _node("rocky-llm", 12 * GIB)
    _node("ovh-cpu-02", 4 * GIB)
    _node("edge-01", 12 * GIB)
    ModelCatalog().set_node_drain("edge-01", drained=True)

    budgets = _node_budgets()
    assert set(budgets) == {"rocky-llm", "ovh-cpu-02"}
    (card,) = _annotate_fits(
        [{"id": "x", "size_est_bytes": 5 * GIB}], max(budgets.values()), budgets=budgets
    )
    assert card["fits_node"] is True
    assert card["fits_node_count"] == 1
    assert card["node_count"] == 2


def test_resources_lists_only_the_shown_nodes_rss(fleet: None) -> None:
    import asyncio

    from docie_bench.inngest.serving_api import serving_resources

    _node("rocky-llm", 8 * GIB)
    _node("ovh-cpu-02", 2 * GIB)
    _place("lfm-a", "rocky-llm")
    _place("lfm-b", "ovh-cpu-02")

    view = asyncio.run(serving_resources(node_id="ovh-cpu-02"))
    assert view["node"]["node_id"] == "ovh-cpu-02"
    assert [(row["name"], row["node_id"]) for row in view["deployments"]] == [
        ("lfm-b", "ovh-cpu-02")
    ]
    assert view["nodes"] == ["ovh-cpu-02", "rocky-llm"]


def test_ports_are_grouped_by_owning_node(fleet: None) -> None:
    from docie_bench.inngest.serving_api import _ports_by_node

    _place("lfm-a", "rocky-llm", port=8090)
    _place("lfm-b", "ovh-cpu-02", port=8090)

    assert _ports_by_node() == {
        "ovh-cpu-02": [{"name": "lfm-b", "port": 8090, "state": "running"}],
        "rocky-llm": [{"name": "lfm-a", "port": 8090, "state": "running"}],
    }


# ------------------------------------------------------------------ A5


def test_targeted_deploy_carries_the_full_launch_payload(fleet: None) -> None:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, _target_node_deploy

    _node("rocky-llm", 8 * GIB, store_models=[MODEL])
    _target_node_deploy(
        DeployRequest(
            model=MODEL,
            node_id="rocky-llm",
            max_tokens=2048,
            n_parallel=4,
            cache_type_k="q8_0",
            cpu_threads=6,
            timeout_seconds=90.0,
        ),
        object(),  # type: ignore[arg-type]
    )
    (row,) = ModelCatalog().desired_snapshot("rocky-llm")["deployments"]
    spec = row["spec"]
    assert spec["max_tokens"] == 2048
    assert spec["n_parallel"] == 4
    assert spec["cache_type_k"] == "q8_0"
    assert spec["cpu_threads"] == 6
    assert spec["timeout_seconds"] == 90.0


def test_owning_node_launches_with_the_desired_spec_knobs() -> None:
    from docie_bench.inngest.worker import _desired_state_deployer

    calls: list[tuple[str, dict[str, object]]] = []

    class Supervisor:
        def serve_store_model(self, model: str, **kwargs: object) -> object:
            calls.append((model, kwargs))
            return {}

    class ControlPlane:
        supervisor = Supervisor()

    deploy = _desired_state_deployer(ControlPlane())
    deploy(
        "lfm-a",
        {"model": MODEL, "context_length": 16384, "n_parallel": 4, "max_tokens": 2048},
    )
    assert calls == [
        (
            MODEL,
            {
                "deployment_name": "lfm-a",
                "context_length": 16384,
                "n_parallel": 4,
                "max_tokens": 2048,
            },
        )
    ]


def test_explicit_runtime_is_refused_on_the_desired_path() -> None:
    import asyncio

    from fastapi import HTTPException

    from docie_bench.inngest.studio_api.deploy import DeployRequest, trigger_deploy

    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            trigger_deploy(
                DeployRequest(model=MODEL, node_id="rocky-llm", runtime="llamacpp"),
                object(),  # type: ignore[arg-type]
            )
        )
    assert refused.value.status_code == 422
