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
