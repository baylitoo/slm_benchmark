"""Choosing a node by fit, and spreading replicas across the fleet (#536).

Two of #536's mutations: "choose a target node, or let the platform choose by
free RAM and fit", and "scale a model to N replicas spread over the fleet
rather than stacked on one machine".

The gates carry more risk than the ranking. An auto-placed deploy has no
operator reading a 409, so a node that would have been refused on the targeted
path has to be refused here too -- and the one that bites is a node with a row
and no measurement: its capacity fields are zeroes, which price as a machine
with nothing on it, so a naive ranking picks it every time.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import StoreEntry
from docie_bench.serving.node_selection import rank_nodes, select_nodes
from docie_bench.settings import get_settings

MODEL = "lfm2.5-350m"
GIB = 1024**3
WEIGHTS = GIB // 2


def _models(name: str = MODEL, *, size_bytes: int | None = WEIGHTS) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "family": "openai_chat",
            "size_bytes": size_bytes,
            "model_path": f"/models/{name}/model.gguf",
        }
    ]


def _snapshot(
    node_id: str,
    *,
    free_gib: float = 6,
    total_gib: float = 8,
    source: str = "cgroup",
    store_models: list[str] | None = None,
    drained: bool = False,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "node_id": node_id,
        "source": source,
        "total_bytes": int(total_gib * GIB),
        "free_bytes": int(free_gib * GIB),
        "sum_rss_bytes": GIB,
        "reclaimable_bytes": 0,
        "store_models": [MODEL] if store_models is None else store_models,
    }
    if drained:
        row["drained_at"] = "2026-10-26T00:00:00+00:00"
    return row


def _rank(snapshots: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    ranked = rank_nodes(MODEL, models=_models(), snapshots=snapshots, **kwargs)
    return {c.node_id: c for c in ranked}


# --- the gates ------------------------------------------------------------


def test_a_node_with_no_measurement_is_never_chosen() -> None:
    """THE trap. Its capacity fields are zeroes, which price as a machine with
    nothing running on it -- so a ranking that trusted the numbers would prefer
    a node that has never started over one that is serving fine."""
    ranked = _rank(
        [_snapshot("live"), _snapshot("staged", source="unreported", free_gib=0, total_gib=0)],
        placements=[],
    )

    assert ranked["staged"].eligible is False
    assert "never published" in ranked["staged"].reason
    assert ranked["live"].eligible is True


def test_a_node_that_stopped_reporting_is_not_chosen() -> None:
    ranked = _rank([_snapshot("a"), _snapshot("b")], placements=[], stale={"b"})

    assert ranked["b"].eligible is False
    assert "stopped reporting" in ranked["b"].reason


def test_a_drained_node_is_not_chosen() -> None:
    """An operator took it out of the rotation; placing work on it would be
    undone by its own reconciler."""
    ranked = _rank([_snapshot("a"), _snapshot("b", drained=True)], placements=[])

    assert ranked["b"].eligible is False
    assert "drained" in ranked["b"].reason


def test_a_node_without_the_blob_is_not_chosen() -> None:
    """The catalog is global and the blob store is node-local, so a model being
    listed says nothing about which nodes can serve it."""
    ranked = _rank([_snapshot("a"), _snapshot("b", store_models=[])], placements=[])

    assert ranked["b"].eligible is False
    assert "blob" in ranked["b"].reason


def test_a_node_that_has_not_reported_its_store_stays_eligible() -> None:
    """`[]` and None are different answers. None means "we have not been told",
    and the targeted path accepts it for that reason -- excluding it here would
    silently skip a node that probably does hold the blob."""
    ranked = _rank([_snapshot("b", store_models=[])], placements=[])
    assert ranked["b"].eligible is False

    unreported = _snapshot("c")
    unreported["store_models"] = None

    assert _rank([unreported], placements=[])["c"].eligible is True


def test_a_full_node_is_rejected_with_a_reason_not_silently_ranked_last() -> None:
    ranked = _rank([_snapshot("tiny", free_gib=0.1, total_gib=8)], placements=[])

    assert ranked["tiny"].eligible is False
    assert "no room" in ranked["tiny"].reason


def test_an_unpriceable_model_leaves_the_node_eligible_but_unranked() -> None:
    """Unknown is not full and not roomy. Preferring a node whose fit cannot be
    computed over one that measured fine would be preferring ignorance."""
    ranked = rank_nodes(
        MODEL,
        models=_models(size_bytes=None),
        snapshots=[_snapshot("a")],
        placements=[],
    )

    assert ranked[0].eligible is True
    assert ranked[0].fits_now is None
    assert ranked[0].priced is False


# --- the ranking ----------------------------------------------------------


def _place(name: str, node_id: str, model: str = MODEL) -> dict[str, Any]:
    return {
        "name": name,
        "model_name": model,
        "node_id": node_id,
        "state": "ready",
        "endpoint": f"http://{node_id}:8088/v1",
        "phase": "hot",
    }


def test_the_emptiest_node_wins_so_replicas_do_not_stack() -> None:
    """Spreading is the point of having a fleet: two replicas on one node share
    its memory, its cgroup limit and its failure."""
    chosen, _ = select_nodes(
        MODEL,
        count=1,
        models=_models(),
        snapshots=[_snapshot("busy"), _snapshot("idle")],
        placements=[_place("x", "busy"), _place("y", "busy")],
    )

    assert chosen == ["idle"]


def test_headroom_breaks_the_tie_between_equally_idle_nodes() -> None:
    chosen, _ = select_nodes(
        MODEL,
        count=1,
        models=_models(),
        snapshots=[_snapshot("small", free_gib=2), _snapshot("big", free_gib=7)],
        placements=[],
    )

    assert chosen == ["big"]


def test_replicas_go_to_distinct_nodes() -> None:
    chosen, _ = select_nodes(
        MODEL,
        count=2,
        models=_models(),
        snapshots=[_snapshot("a"), _snapshot("b"), _snapshot("c")],
        placements=[],
    )

    assert len(chosen) == 2
    assert len(set(chosen)) == 2


def test_fewer_nodes_are_returned_rather_than_doubling_up() -> None:
    """A node with room for three instances still only takes one replica. The
    caller decides whether that is a refusal; stacking is never the answer."""
    chosen, _ = select_nodes(
        MODEL, count=3, models=_models(), snapshots=[_snapshot("only")], placements=[]
    )

    assert chosen == ["only"]


def test_the_ranking_keeps_rejected_nodes_so_a_refusal_can_explain_itself() -> None:
    """"No node fits" is not useful. "node-b does not hold the blob" is."""
    _chosen, candidates = select_nodes(
        MODEL,
        count=2,
        models=_models(),
        snapshots=[_snapshot("a"), _snapshot("b", store_models=[])],
        placements=[],
    )

    rejected = {c.node_id: c.reason for c in candidates if not c.eligible}
    assert "b" in rejected
    assert "blob" in rejected["b"]


# --- the endpoint ---------------------------------------------------------


@pytest.fixture
def catalog_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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


def _seed_model() -> None:
    ModelCatalog().upsert(
        StoreEntry(
            name=MODEL,
            family="openai_chat",
            model_path=Path(f"/models/{MODEL}/model.gguf"),
        ),
        size_bytes=WEIGHTS,
    )


def _publish(node_id: str, *, free_gib: float = 6, store: list[str] | None = None) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=int(free_gib * GIB),
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=[MODEL] if store is None else store,
    )


def _deploy(**kwargs: Any) -> Any:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, trigger_deploy

    return asyncio.run(
        trigger_deploy(
            DeployRequest(model=MODEL, **kwargs),
            tenant=None,  # type: ignore[arg-type]
        )
    )


def _desired(node_id: str) -> list[dict[str, Any]]:
    rows = ModelCatalog().desired_snapshot(node_id)["deployments"]
    return list(rows) if isinstance(rows, list) else []


def test_auto_place_records_desired_state_on_the_node_it_picked(
    catalog_db: None,
) -> None:
    _seed_model()
    _publish("roomy", free_gib=7)
    _publish("cramped", free_gib=1.2)

    result = _deploy(auto_place=True)

    # No event and no channel: desired state is converged by the node itself.
    assert result.channel is None
    assert result.event_ids == []
    assert len(_desired("roomy")) == 1
    assert _desired("cramped") == []


def test_auto_place_spreads_replicas_over_distinct_nodes(catalog_db: None) -> None:
    _seed_model()
    _publish("a")
    _publish("b")

    _deploy(auto_place=True, replicas=2)

    assert len(_desired("a")) == 1
    assert len(_desired("b")) == 1


def test_auto_place_refuses_a_partial_spread_and_names_the_gate(
    catalog_db: None,
) -> None:
    """Two replicas on one machine is not what asking for two replicas means,
    and "I asked for 3 and got 2" is a capacity decision only the operator can
    make."""
    from fastapi import HTTPException

    _seed_model()
    _publish("a")
    _publish("no-blob", store=[])

    with pytest.raises(HTTPException) as excinfo:
        _deploy(auto_place=True, replicas=2)

    assert excinfo.value.status_code == 409
    detail = str(excinfo.value.detail)
    assert "no-blob" in detail
    assert "blob" in detail
    # Nothing was placed: the refusal happens before any row is written.
    assert _desired("a") == []


def test_naming_a_node_and_asking_for_one_to_be_chosen_is_refused(
    catalog_db: None,
) -> None:
    from fastapi import HTTPException

    _seed_model()
    _publish("a")

    with pytest.raises(HTTPException) as excinfo:
        _deploy(auto_place=True, node_id="a")

    assert excinfo.value.status_code == 422


def test_auto_place_with_no_reporting_node_says_so(catalog_db: None) -> None:
    """Rather than a confusing "0 of 1 nodes can take it" on a single-node
    deployment that has nothing to choose between."""
    from fastapi import HTTPException

    _seed_model()

    with pytest.raises(HTTPException) as excinfo:
        _deploy(auto_place=True)

    assert excinfo.value.status_code == 409
    assert "no serving node has reported" in str(excinfo.value.detail)


def test_an_explicit_name_cannot_be_combined_with_replicas(catalog_db: None) -> None:
    from fastapi import HTTPException

    _seed_model()
    _publish("a")
    _publish("b")

    with pytest.raises(HTTPException) as excinfo:
        _deploy(auto_place=True, replicas=2, name="pinned")

    assert excinfo.value.status_code == 422


def test_an_untargeted_deploy_is_unchanged_by_any_of_this(catalog_db: None) -> None:
    """The single-node path must not acquire a catalog dependency: without
    auto_place or node_id nothing above runs."""
    from docie_bench.inngest.studio_api.deploy import DeployRequest

    payload = DeployRequest(model=MODEL)

    assert payload.auto_place is False
    assert payload.node_id is None
