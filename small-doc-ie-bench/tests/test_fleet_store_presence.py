"""Which nodes actually hold a model's blob (#562).

`model_store_entry` is global; the blob store is not -- it lives in
`DOCIE_SERVING_HOME/models`, on a volume #564 documents as unshareable. So the
catalog saying a model exists does not mean the node chosen to serve it has the
file, and the Studio showed one node's disk as the fleet's store.

That combination is the trap this closes: seed events are not node-addressable
(they land on whichever serving node Inngest picks), so an operator cannot
direct a seed at a node -- and before this could not even see which nodes were
missing a model.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.model_store import StoreEntry
from docie_bench.settings import get_settings

GIB = 1024**3
MODEL = "lfm2.5-350m"
OTHER = "nuextract3"


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


def _catalog_entry(name: str) -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=name,
            family="openai_chat",
            model_path=Path(f"/models/{name}/model.gguf"),
        )
    )


def _node(node_id: str, store_models: list[str] | None) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=store_models,
    )


def _store() -> dict[str, dict[str, object]]:
    from docie_bench.inngest.serving_api import list_store

    result = asyncio.run(list_store())
    assert isinstance(result, list)
    return {str(row["name"]): row for row in result}


def test_a_model_seeded_on_another_node_is_listed(fleet: None) -> None:
    """This process has no blob for it, so the on-disk view cannot see it --
    and a deploy of a model the Studio does not show simply fails."""
    _catalog_entry(OTHER)
    _node("node-b", [OTHER])

    rows = _store()

    assert OTHER in rows
    assert rows[OTHER]["store_view"] == "catalog"


def test_presence_separates_missing_from_unreported(fleet: None) -> None:
    """An operator does different things with each: `missing` means re-seed
    that node, `unreported` means wait for its next publish. Collapsing them
    would turn "we do not know" into "it is not there"."""
    _catalog_entry(MODEL)
    _node("node-a", [MODEL])
    _node("node-b", [])  # reported, and does not have it
    _node("node-c", None)  # has not reported its store yet

    nodes = _store()[MODEL]["nodes"]

    assert nodes == {
        "present": ["node-a"],
        "missing": ["node-b"],
        "unreported": ["node-c"],
    }


def test_the_blob_being_global_in_the_catalog_does_not_imply_presence(
    fleet: None,
) -> None:
    """The trap: the catalog says the model exists, every node says it does
    not have it. Before this, nothing in the Studio could show that."""
    _catalog_entry(MODEL)
    _node("node-a", [])
    _node("node-b", [])

    nodes = _store()[MODEL]["nodes"]

    assert nodes["present"] == []
    assert nodes["missing"] == ["node-a", "node-b"]


def test_a_node_reporting_an_empty_store_is_not_unreported(fleet: None) -> None:
    """[] and None are different states and must round-trip as such."""
    _catalog_entry(MODEL)
    _node("node-a", [])

    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, "node-a")
        assert row is not None
        assert json.loads(str(row.store_models)) == []

    assert _store()[MODEL]["nodes"]["missing"] == ["node-a"]


def test_the_store_listing_survives_an_unreachable_catalog(fleet: None) -> None:
    """Degrades to the on-disk view, un-annotated, rather than failing --
    the property the route's docstring already promises."""
    db.dispose_engine()

    rows = _store()

    assert rows == {}  # nothing on this node's disk, and no catalog to add from


def test_presence_is_absent_rather_than_wrong_without_node_reports(
    fleet: None,
) -> None:
    """With no node snapshots at all, every list is empty -- the annotation
    says nothing instead of claiming the model is missing everywhere."""
    _catalog_entry(MODEL)

    nodes = _store()[MODEL]["nodes"]

    assert nodes == {"present": [], "missing": [], "unreported": []}
