"""The Studio's deployment listing covers the fleet, not one node's disk (#562).

`serving_api.py` says of itself that it is *"a read-only viewer over the shared
on-disk state (DOCIE_SERVING_HOME)"*. With a second serving node that is no
longer a viewer over the fleet: every deployment on another machine is simply
absent, and the listing presents one node's file as though it were everything.
A missing deployment is worse than a partial one, so catalog-derived records
fill the gap and say what they are.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.node_liveness import snapshot_stale_after_s
from docie_bench.serving.runtime import RuntimeKind, RuntimeLaunchSpec
from docie_bench.serving.supervisor import DeploymentSpec, PersistentSupervisor
from docie_bench.settings import get_settings

GIB = 1024**3
LOCAL = "lfm2.5-350m"
REMOTE = "nuextract3"


class _FakeAdapter:
    def start(self, spec: object) -> object:  # pragma: no cover - never started
        raise AssertionError("tests never launch a runtime")


@pytest.fixture
def serving_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "serving"
    home.mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield home
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _seed_local(home: Path, name: str = LOCAL, *, port: int = 8088) -> None:
    """A deployment this process has a file for — node A's own record."""
    supervisor = PersistentSupervisor(
        home / "deployments.json", adapters={RuntimeKind.LLAMACPP: _FakeAdapter()}
    )
    supervisor.deploy(
        DeploymentSpec(
            name=name,
            launch=RuntimeLaunchSpec(
                runtime=RuntimeKind.LLAMACPP,
                model=f"/models/{name}.gguf",
                alias=name,
                port=port,
            ),
        )
    )


def _place(name: str, *, node_id: str, state: str = "ready") -> None:
    ModelCatalog().record_placement(
        name,
        model_name=name,
        engine="llama-server",
        endpoint=f"http://{node_id}:8088/v1",
        state=state,
        node_id=node_id,
    )


def _node_snapshot(node_id: str, *, age_s: float = 0.0) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=node_id,
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    if age_s:
        with db.session_scope() as session:
            assert session is not None
            row = session.get(ServingNode, node_id)
            assert row is not None
            row.updated_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=age_s)


def _listing() -> list[dict[str, object]]:
    from docie_bench.inngest.serving_api import list_deployments

    result = asyncio.run(list_deployments())
    assert isinstance(result, list)
    return result


def test_a_remote_nodes_deployment_is_listed_and_attributed(serving_home: Path) -> None:
    """Acceptance: a listing shows deployments from both nodes, each
    attributed to its node."""
    _seed_local(serving_home)
    _place(LOCAL, node_id="node-a")
    _place(REMOTE, node_id="node-b")  # no local file for this one
    _node_snapshot("node-a")
    _node_snapshot("node-b")

    by_name = {str((r.get("spec") or {}).get("name")): r for r in _listing()}

    assert set(by_name) == {LOCAL, REMOTE}
    assert by_name[LOCAL]["node_id"] == "node-a"
    assert by_name[REMOTE]["node_id"] == "node-b"


def test_a_catalog_derived_record_says_so(serving_home: Path) -> None:
    """It is deliberately partial -- desired state, restart policy and pins
    live in the owning node's file -- so the UI must be able to tell."""
    _seed_local(serving_home)
    _place(LOCAL, node_id="node-a")
    _place(REMOTE, node_id="node-b")

    by_name = {str((r.get("spec") or {}).get("name")): r for r in _listing()}

    assert by_name[LOCAL]["record_source"] == "node"
    assert by_name[REMOTE]["record_source"] == "catalog"
    assert by_name[REMOTE]["endpoint"] == "http://node-b:8088/v1"


def test_an_unreachable_node_is_reported_not_omitted(serving_home: Path) -> None:
    """Acceptance: a node that is unreachable is reported as unreachable
    rather than dropped from the listing."""
    _seed_local(serving_home)
    _place(LOCAL, node_id="node-a")
    _place(REMOTE, node_id="node-b")
    _node_snapshot("node-a")
    _node_snapshot("node-b", age_s=snapshot_stale_after_s() * 10)

    by_name = {str((r.get("spec") or {}).get("name")): r for r in _listing()}

    assert REMOTE in by_name  # still listed
    assert by_name[REMOTE]["node_reporting"] is False
    assert by_name[LOCAL]["node_reporting"] is True


def test_replica_counts_span_the_fleet(serving_home: Path) -> None:
    """A replica on another node counts towards the group, or the Deploy tab
    would report a scaled model as having fewer replicas than it has."""
    _seed_local(serving_home, LOCAL)
    _place(LOCAL, node_id="node-a")
    ModelCatalog().record_placement(
        f"{LOCAL}-2",
        model_name=LOCAL,  # the shared launch alias
        engine="llama-server",
        endpoint="http://node-b:8088/v1",
        state="ready",
        node_id="node-b",
    )

    by_name = {str((r.get("spec") or {}).get("name")): r for r in _listing()}

    assert by_name[LOCAL]["replicas"] == 2
    assert by_name[f"{LOCAL}-2"]["replicas"] == 2
    assert by_name[f"{LOCAL}-2"]["replica_group"] == LOCAL


def test_a_single_node_listing_is_unchanged(serving_home: Path) -> None:
    """Acceptance: with one node, behaviour is unchanged -- every placement
    row matches a local record, so nothing is added."""
    _seed_local(serving_home)
    _place(LOCAL, node_id="node-a")
    _node_snapshot("node-a")

    listing = _listing()

    assert len(listing) == 1
    assert listing[0]["record_source"] == "node"
    assert listing[0]["node_reporting"] is True


def test_the_listing_survives_an_unreachable_catalog(serving_home: Path) -> None:
    """The degraded path from the original docstring: with no catalog the Board
    falls back to the local file, which the reconciler keeps de-staled. It must
    not become empty or raise."""
    _seed_local(serving_home)
    db.dispose_engine()

    listing = _listing()

    assert [str((r.get("spec") or {}).get("name")) for r in listing] == [LOCAL]
    assert listing[0]["observed_available"] is False


# --- lifecycle on a remote deployment (#562) ----------------------------


def test_a_lifecycle_action_on_a_remote_deployment_names_its_node(
    serving_home: Path,
) -> None:
    """These handlers read THIS node's deployments.json, so a live deployment
    on another node reaches the not-found path. A 404 would say it does not
    exist -- wrong, and the kind of wrong that sends an operator looking in the
    right place last.
    """
    import asyncio

    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import _fire_lifecycle_event

    _place(REMOTE, node_id="node-b")  # catalogued, no local record

    with pytest.raises(HTTPException) as caught:
        asyncio.run(_fire_lifecycle_event(REMOTE, event="serving/x", prefix="x"))

    assert caught.value.status_code == 409
    detail = str(caught.value.detail)
    assert "node-b" in detail
    assert "docie-serving stop" in detail  # tells them what to do instead


def test_a_genuinely_unknown_deployment_is_still_a_404(serving_home: Path) -> None:
    """The new branch must not swallow the real not-found case."""
    import asyncio

    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import _fire_lifecycle_event

    with pytest.raises(HTTPException) as caught:
        asyncio.run(_fire_lifecycle_event("never-existed", event="serving/x", prefix="x"))

    assert caught.value.status_code == 404


def test_a_placement_with_no_owner_is_still_a_404(serving_home: Path) -> None:
    """A pre-#530 row has an unknown owner, not a remote one, so there is no
    node to point at and the honest answer stays 404."""
    import asyncio

    from fastapi import HTTPException

    from docie_bench.inngest.serving_api import _fire_lifecycle_event

    ModelCatalog().record_placement(
        "legacy",
        model_name="legacy",
        engine="llama-server",
        endpoint="http://legacy:8088/v1",
        state="ready",
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(_fire_lifecycle_event("legacy", event="serving/x", prefix="x"))

    assert caught.value.status_code == 404
