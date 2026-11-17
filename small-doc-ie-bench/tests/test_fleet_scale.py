"""Replica counts and placement span the fleet, not the api's own file (#664 A4).

The api holds one node's ``deployments.json``. Counting replicas from it
ignored every other node, and the deploy/delete events it sent reached
whichever node Inngest picked -- so a scale-down could land on a node that did
not own the replica.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog
from docie_bench.serving.model_store import StoreEntry
from docie_bench.settings import get_settings

MODEL = "lfm2.5-350m"
GIB = 1024**3


@pytest.fixture
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[Any]]:
    """An empty local file, a sqlite catalog, and captured Inngest events."""
    from docie_bench.inngest.client import inngest_client

    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    sent: list[Any] = []

    async def fake_send(event: Any) -> list[str]:
        sent.append(event)
        return [f"evt-{len(sent)}"]

    monkeypatch.setattr(inngest_client, "send", fake_send)
    ModelCatalog().upsert(
        StoreEntry(name=MODEL, family="openai_chat", model_path=Path(f"/models/{MODEL}/m.gguf")),
        size_bytes=GIB // 2,
    )
    try:
        yield sent
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _node(node_id: str) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=6 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=[MODEL],
    )


def _replica(name: str, node_id: str) -> None:
    ModelCatalog().record_placement(
        name,
        model_name=MODEL,
        engine="llama-server",
        endpoint=f"http://{node_id}:8088/v1",
        state="ready",
        node_id=node_id,
    )


def _scale(replicas: int) -> dict[str, Any]:
    from docie_bench.inngest.serving_api import ScaleRequest, scale_store_model

    return asyncio.run(
        scale_store_model(MODEL, ScaleRequest(replicas=replicas), tenant=None)  # type: ignore[arg-type]
    )


def _desired(node_id: str) -> list[dict[str, Any]]:
    rows = ModelCatalog().desired_snapshot(node_id)["deployments"]
    return list(rows) if isinstance(rows, list) else []


def test_a_replica_on_another_node_counts_toward_the_target(fleet: list[Any]) -> None:
    """Single node, but not this process's file: already at 1, nothing to add."""
    _node("rocky-llm")
    _replica(f"{MODEL}-a1b2c3", "rocky-llm")

    result = _scale(1)

    assert result["current"] == 1
    assert result["adding"] == []
    assert fleet == []


def test_scale_up_places_the_new_replica_on_a_node_without_one(fleet: list[Any]) -> None:
    _node("rocky-llm")
    _node("gpu-box")
    _replica(f"{MODEL}-a1b2c3", "rocky-llm")

    result = _scale(2)

    assert fleet == []  # no untargeted event
    assert result["channel"] is None
    assert _desired("rocky-llm") == []
    (row,) = _desired("gpu-box")
    assert row["name"] == result["adding"][0]
    assert row["spec"]["model"] == MODEL


def test_scale_down_removes_through_the_owners_desired_state(fleet: list[Any]) -> None:
    _node("rocky-llm")
    _node("gpu-box")
    _replica(f"{MODEL}-a1b2c3", "rocky-llm")
    ModelCatalog().set_desired_deployment(
        "gpu-box", f"{MODEL}-d4e5f6", desired_state="running", spec={"model": MODEL}
    )
    _replica(f"{MODEL}-d4e5f6", "gpu-box")

    result = _scale(1)

    assert fleet == []
    assert result["removing"] == [f"{MODEL}-d4e5f6"]
    assert _desired("gpu-box") == []


def test_scale_down_refuses_a_replica_desired_state_cannot_remove(fleet: list[Any]) -> None:
    """An untargeted delete event could reach the wrong node, so refuse."""
    _node("rocky-llm")
    _node("gpu-box")
    _replica(f"{MODEL}-a1b2c3", "rocky-llm")
    _replica(f"{MODEL}-d4e5f6", "gpu-box")

    with pytest.raises(HTTPException) as excinfo:
        _scale(1)

    assert excinfo.value.status_code == 409
    assert "gpu-box" in str(excinfo.value.detail)
    assert fleet == []


def test_deploy_with_replicas_spreads_over_the_fleet(fleet: list[Any]) -> None:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, trigger_deploy

    _node("rocky-llm")
    _node("gpu-box")

    result = asyncio.run(
        trigger_deploy(DeployRequest(model=MODEL, replicas=2), tenant=None)  # type: ignore[arg-type]
    )

    assert fleet == []
    assert result.channel is None
    assert len(_desired("rocky-llm")) == 1
    assert len(_desired("gpu-box")) == 1


def test_a_leftover_node_row_does_not_switch_off_the_event_path(
    fleet: list[Any],
) -> None:
    """One live node plus the stale row of a replaced one is not a fleet."""
    import datetime as dt

    from docie_bench.serving.catalog import ServingNode
    from docie_bench.storage.db import session_scope

    _node("rocky-llm")
    _node("rocky-llm-old")
    with session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, "rocky-llm-old")
        assert row is not None
        row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)

    result = _scale(1)

    assert len(fleet) == 1  # the ordinary deploy event
    assert result["channel"] is not None


def test_replicas_placed_on_the_fleet_keep_the_deploys_launch_settings(
    fleet: list[Any],
) -> None:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, trigger_deploy

    _node("rocky-llm")
    _node("gpu-box")

    asyncio.run(
        trigger_deploy(
            DeployRequest(model=MODEL, replicas=2, max_tokens=512, cpu_threads=6),
            tenant=None,  # type: ignore[arg-type]
        )
    )

    for node in ("rocky-llm", "gpu-box"):
        (row,) = _desired(node)
        assert row["spec"]["max_tokens"] == 512
        assert row["spec"]["cpu_threads"] == 6
