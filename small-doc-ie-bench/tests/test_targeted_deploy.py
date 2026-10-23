"""Deploying to a chosen node (#535).

Until now a deploy was an event on the shared `docie-serving` app, so it landed
on whichever serving node Inngest picked and a specific node could only be
reached by running the CLI on it (#576). A deploy with `node_id` is recorded as
desired state owned by that node, and only that node's reconciler reads it.

Most of these tests are about the pre-flight refusals, because both mistakes an
operator will actually make are silent otherwise: a row nothing will ever read,
or a deploy that fails on the node for a reason the Studio does not show.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.node_liveness import snapshot_stale_after_s

NODE = "ovh-cpu-02"
MODEL = "lfm2.5-350m"
GIB = 1024**3


class _Tenant:
    tenant_id = "tests"


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[ModelCatalog]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield ModelCatalog()
    finally:
        db.dispose_engine()


def _node(
    catalog: ModelCatalog,
    node_id: str = NODE,
    *,
    store_models: list[str] | None = None,
    age_s: float = 0.0,
) -> None:
    catalog.publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
        store_models=store_models,
    )
    if age_s:
        with db.session_scope() as session:
            assert session is not None
            row = session.get(ServingNode, node_id)
            assert row is not None
            row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s)


def _deploy(**overrides: object) -> object:
    from docie_bench.inngest.studio_api.deploy import DeployRequest, _target_node_deploy

    payload = DeployRequest(model=MODEL, node_id=NODE, **overrides)  # type: ignore[arg-type]
    return _target_node_deploy(payload, _Tenant())  # type: ignore[arg-type]


# --- the happy path ------------------------------------------------------


def test_a_targeted_deploy_is_recorded_as_desired_state(
    catalog: ModelCatalog,
) -> None:
    _node(catalog, store_models=[MODEL])

    response = _deploy()

    snapshot = catalog.desired_snapshot(NODE)
    assert len(snapshot["deployments"]) == 1
    (row,) = snapshot["deployments"]
    assert row["desired_state"] == "running"
    assert row["spec"]["model"] == MODEL
    # No progress stream: that node's reconciler converges on its own schedule.
    assert response.event_ids == []
    assert response.channel is None
    assert "desired state" in (response.detail or "")


def test_the_record_name_is_minted_so_two_nodes_cannot_collide(
    catalog: ModelCatalog,
) -> None:
    """The reason #572 mints: `model_placement.name` is the primary key, so two
    nodes deploying the same model must not choose the same record name."""
    _node(catalog, store_models=[MODEL])
    _node(catalog, "other-node", store_models=[MODEL])

    _deploy()
    from docie_bench.inngest.studio_api.deploy import DeployRequest, _target_node_deploy

    _target_node_deploy(
        DeployRequest(model=MODEL, node_id="other-node"),  # type: ignore[arg-type]
        _Tenant(),  # type: ignore[arg-type]
    )

    here = catalog.desired_snapshot(NODE)["deployments"][0]["name"]
    there = catalog.desired_snapshot("other-node")["deployments"][0]["name"]
    assert here != there
    assert here.startswith(f"{MODEL}-")


def test_an_explicit_name_is_honoured(catalog: ModelCatalog) -> None:
    _node(catalog, store_models=[MODEL])

    _deploy(name="chosen-by-hand")

    (row,) = catalog.desired_snapshot(NODE)["deployments"]
    assert row["name"] == "chosen-by-hand"


# --- the refusals --------------------------------------------------------


def test_an_unknown_node_is_refused(catalog: ModelCatalog) -> None:
    """Otherwise the row is written and nothing ever reads it -- the deploy
    simply never happens, with a 200 to say it did."""
    _node(catalog, "some-other-node", store_models=[MODEL])

    with pytest.raises(HTTPException) as caught:
        _deploy()

    assert caught.value.status_code == 404
    assert "some-other-node" in str(caught.value.detail)  # names what IS known


def test_a_node_that_stopped_reporting_is_refused(catalog: ModelCatalog) -> None:
    _node(catalog, store_models=[MODEL], age_s=snapshot_stale_after_s() * 10)

    with pytest.raises(HTTPException) as caught:
        _deploy()

    assert caught.value.status_code == 409
    assert "stopped reporting" in str(caught.value.detail)


def test_a_node_without_the_blob_is_refused(catalog: ModelCatalog) -> None:
    """The #573 trap, turned into a pre-flight refusal: the catalog knows the
    model globally, the blob store is node-local, so the deploy would fail on
    that node for a reason the Studio does not show."""
    _node(catalog, store_models=["something-else"])

    with pytest.raises(HTTPException) as caught:
        _deploy()

    assert caught.value.status_code == 409
    detail = str(caught.value.detail)
    assert "does not hold" in detail
    assert "seed" in detail  # tells the operator what to do


def test_a_node_that_has_not_reported_its_store_is_allowed(
    catalog: ModelCatalog,
) -> None:
    """`unreported` is not `missing` (#573): a node that has not published its
    store yet must not be refused as though it lacked the model."""
    _node(catalog, store_models=None)

    _deploy()

    assert len(catalog.desired_snapshot(NODE)["deployments"]) == 1


def test_targeting_a_node_with_replicas_is_refused(catalog: ModelCatalog) -> None:
    """node_id names ONE node; replicas > 1 would have to invent a placement
    policy, which is a separate decision (#539)."""
    _node(catalog, store_models=[MODEL])

    with pytest.raises(HTTPException) as caught:
        _deploy(replicas=3)

    assert caught.value.status_code == 422


def test_an_unconfigured_catalog_is_refused_with_the_untargeted_hint(
    tmp_path: Path,
) -> None:
    """Targeting needs the catalog; say so and point at the path that does
    not, rather than failing opaquely."""
    db.dispose_engine()

    with pytest.raises(HTTPException) as caught:
        _deploy()

    assert caught.value.status_code == 503
    assert "node_id" in str(caught.value.detail)
