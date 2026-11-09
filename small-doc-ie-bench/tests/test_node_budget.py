"""A per-node memory budget an operator edits live (#539).

These nodes are shared hosts. Free host RAM is not DocIE's to take: memory
that is free now belongs to whichever co-tenant asks for it next, and the OOM
killer picks the loser. So the budget is a correctness input, and the direction
that matters is that nothing ever WIDENS it by accident -- not a catalog
outage, not a restart, not an unreadable cache file.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.security import TenantContext
from docie_bench.serving.catalog import (
    CatalogUnavailableError,
    ModelCatalog,
    NodeBudgetConflictError,
    ServingNode,
)
from docie_bench.serving.node_budget import (
    BudgetResolution,
    budget_cache_path,
    read_budget_cache,
    reset_budget_cache,
    resolve_budget,
    write_budget_cache,
)
from docie_bench.settings import get_settings

NODE = "ovh-cpu-02"
GIB = 1024**3


@pytest.fixture
def node_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", NODE)
    monkeypatch.delenv("DOCIE_SERVING_MEM_BUDGET", raising=False)
    from docie_bench.serving.node_identity import resolve_node_id

    resolve_node_id.cache_clear()
    get_settings.cache_clear()
    reset_budget_cache()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        resolve_node_id.cache_clear()
        get_settings.cache_clear()
        reset_budget_cache()


def _setting(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv("DOCIE_SERVING_MEM_BUDGET", raising=False)
    else:
        monkeypatch.setenv("DOCIE_SERVING_MEM_BUDGET", value)
    get_settings.cache_clear()
    reset_budget_cache()


# --- precedence ------------------------------------------------------------


def test_a_catalog_override_wins_over_the_bootstrap_setting(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point: the live number beats the one baked into the container
    at creation."""
    _setting(monkeypatch, "8g")
    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)
    reset_budget_cache()

    resolution = resolve_budget()

    assert resolution == BudgetResolution(budget_bytes=6 * GIB, source="catalog")


def test_no_override_falls_back_to_the_setting(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setting(monkeypatch, "8g")

    assert resolve_budget() == BudgetResolution(
        budget_bytes=8 * GIB, source="setting"
    )


def test_no_override_and_no_setting_is_no_budget(node_home: None) -> None:
    """The cgroup ceiling alone, which is already a real limit where one
    exists. A budget of zero would refuse every load."""
    assert resolve_budget() == BudgetResolution(budget_bytes=None, source="none")


def test_clearing_the_override_returns_to_the_setting(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setting(monkeypatch, "8g")
    first = ModelCatalog().set_node_budget(
        NODE, budget_bytes=6 * GIB, expected_version=None
    )
    ModelCatalog().set_node_budget(
        NODE, budget_bytes=None, expected_version=first["mem_budget_version"]
    )
    reset_budget_cache()

    assert resolve_budget().budget_bytes == 8 * GIB


# --- the outage direction --------------------------------------------------


def test_an_unreachable_catalog_never_widens_the_budget(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE case this exists for. Cached 6g, setting 8g, catalog gone: 6g.

    Taking the setting here would hand a co-tenant's 2GiB to a model load,
    because the operator's narrower number is the one that was validated.
    """
    _setting(monkeypatch, "8g")
    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)
    reset_budget_cache()
    assert resolve_budget().budget_bytes == 6 * GIB  # populates the cache

    db.dispose_engine()  # the catalog goes away
    reset_budget_cache()

    resolution = resolve_budget()
    assert resolution.budget_bytes == 6 * GIB
    assert resolution.source == "catalog_cache"
    assert resolution.stale


def test_an_outage_with_a_validated_no_override_uses_the_setting_but_says_stale(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog did answer, and said there is no override. The setting is
    right -- and one may have been set since, so it is flagged."""
    _setting(monkeypatch, "8g")
    assert resolve_budget().source == "setting"  # caches the validated NULL
    assert read_budget_cache(NODE) is None

    db.dispose_engine()
    reset_budget_cache()

    resolution = resolve_budget()
    assert resolution.budget_bytes == 8 * GIB
    assert resolution.source == "setting"
    assert resolution.stale


def test_a_node_that_never_read_its_row_uses_the_setting_unflagged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-node stack with no DATABASE_URL at all. Not an outage: there is
    nothing it is out of touch with."""
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", NODE)
    monkeypatch.setenv("DOCIE_SERVING_MEM_BUDGET", "8g")
    get_settings.cache_clear()
    reset_budget_cache()
    db.dispose_engine()
    try:
        resolution = resolve_budget()
    finally:
        get_settings.cache_clear()
        reset_budget_cache()

    assert resolution == BudgetResolution(budget_bytes=8 * GIB, source="setting")
    assert not resolution.stale


def test_the_error_type_does_not_decide_the_outage_path(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disposed engine raises CatalogUnavailableError, the same error as "no
    DATABASE_URL" -- so reading the error as "single-node stack" would widen a
    fleet node's budget during an outage. The CACHE decides instead."""
    _setting(monkeypatch, "8g")
    write_budget_cache(6 * GIB, node_id=NODE)

    def unavailable(self: ModelCatalog, node_id: str) -> dict[str, object] | None:
        raise CatalogUnavailableError("DATABASE_URL is not configured")

    monkeypatch.setattr(ModelCatalog, "read_node_budget", unavailable)
    reset_budget_cache()

    assert resolve_budget().budget_bytes == 6 * GIB


def test_an_unreadable_cache_file_does_not_invent_a_budget(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setting(monkeypatch, "8g")
    path = budget_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    assert read_budget_cache() == "unknown"

    db.dispose_engine()
    reset_budget_cache()
    assert resolve_budget().budget_bytes == 8 * GIB


def test_the_cache_survives_a_restart(node_home: None) -> None:
    """Durable beside the node's other state, so outage behaviour after a
    restart is the same as before one."""
    write_budget_cache(6 * GIB, node_id=NODE)
    reset_budget_cache()  # the in-process TTL is what a restart clears

    assert read_budget_cache(NODE) == 6 * GIB


# --- the read cost ---------------------------------------------------------


def test_the_catalog_is_not_read_on_every_sizing_decision(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`read_node_memory` runs fresh at every decision by design, so a round
    trip per call is not affordable."""
    reads: list[str] = []
    original = ModelCatalog.read_node_budget

    def counting(self: ModelCatalog, node_id: str) -> dict[str, object] | None:
        reads.append(node_id)
        return original(self, node_id)

    monkeypatch.setattr(ModelCatalog, "read_node_budget", counting)
    reset_budget_cache()

    for _ in range(5):
        resolve_budget(clock=100.0)

    assert len(reads) == 1


def test_a_live_edit_lands_once_the_ttl_passes(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    reset_budget_cache()
    assert resolve_budget(clock=100.0).budget_bytes is None

    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)

    assert resolve_budget(clock=101.0).budget_bytes is None  # inside the TTL
    assert resolve_budget(clock=200.0).budget_bytes == 6 * GIB


def test_changing_the_setting_is_not_served_from_a_stale_entry(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The setting is one of the inputs, so an entry computed under a different
    one answers a different question."""
    _setting(monkeypatch, "8g")
    assert resolve_budget(clock=100.0).budget_bytes == 8 * GIB

    monkeypatch.setenv("DOCIE_SERVING_MEM_BUDGET", "4g")
    get_settings.cache_clear()

    assert resolve_budget(clock=100.0).budget_bytes == 4 * GIB


# --- the catalog write -----------------------------------------------------


def test_a_budget_write_round_trips_and_versions(node_home: None) -> None:
    first = ModelCatalog().set_node_budget(
        NODE, budget_bytes=6 * GIB, expected_version=None, actor="ops@example.test"
    )

    assert first["mem_budget_bytes"] == 6 * GIB
    assert first["mem_budget_version"] == 1
    assert first["mem_budget_actor"] == "ops@example.test"
    assert first["mem_budget_updated_at"]

    second = ModelCatalog().set_node_budget(
        NODE, budget_bytes=5 * GIB, expected_version=1
    )
    assert second["mem_budget_version"] == 2


def test_a_stale_version_loses_rather_than_overwriting(node_home: None) -> None:
    """Two operators editing one node's share of a shared host must not
    interleave: the loser re-reads and decides against what actually won."""
    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)

    with pytest.raises(NodeBudgetConflictError):
        ModelCatalog().set_node_budget(
            NODE, budget_bytes=4 * GIB, expected_version=None
        )

    assert ModelCatalog().read_node_budget(NODE)["mem_budget_bytes"] == 6 * GIB


def test_a_budget_of_zero_is_refused(node_home: None) -> None:
    """Nobody means "refuse every load" by typing zero. None clears."""
    with pytest.raises(ValueError, match="must be positive"):
        ModelCatalog().set_node_budget(NODE, budget_bytes=0, expected_version=None)


def test_a_budget_can_be_set_before_the_node_ever_reports(node_home: None) -> None:
    """The normal order when adding a machine to the fleet."""
    result = ModelCatalog().set_node_budget(
        "ovh-cpu-07", budget_bytes=6 * GIB, expected_version=None
    )

    assert result["mem_budget_bytes"] == 6 * GIB
    rows = {row["node_id"]: row for row in ModelCatalog().list_node_snapshots()}
    assert rows["ovh-cpu-07"]["source"] == "unreported"


def test_a_budget_write_does_not_make_the_node_reconverge(node_home: None) -> None:
    """A budget edit is not a change to what this node should be RUNNING.
    Bumping the desired generation would make it re-plan its deployments
    because somebody typed a number."""
    ModelCatalog().set_desired_deployment(
        NODE, "lfm2.5-350m", desired_state="running", spec={}
    )
    before = ModelCatalog().desired_snapshot(NODE)["generation"]

    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)

    assert ModelCatalog().desired_snapshot(NODE)["generation"] == before


def test_the_fleet_view_reports_the_requested_budget(node_home: None) -> None:
    """What was ASKED for, which is not what the node sizes against -- it
    reports that in total_bytes, having already taken min(budget, ceiling) and
    subtracted its own working set."""
    ModelCatalog().set_node_budget(
        NODE, budget_bytes=6 * GIB, expected_version=None, actor="ops@example.test"
    )

    row = next(
        r for r in ModelCatalog().list_node_snapshots() if r["node_id"] == NODE
    )
    assert row["mem_budget_bytes"] == 6 * GIB
    assert row["mem_budget_version"] == 1
    assert row["mem_budget_actor"] == "ops@example.test"


def test_reading_a_node_with_no_row_is_distinct_from_no_override(
    node_home: None,
) -> None:
    """Two different facts: the catalog has nothing for this machine, versus
    the catalog says this machine has no override."""
    assert ModelCatalog().read_node_budget("never-seen") is None

    with db.session_scope() as session:
        assert session is not None
        session.add(
            ServingNode(
                id="seen",
                total_bytes=0,
                free_bytes=0,
                source="unreported",
                sum_rss_bytes=0,
                reclaimable_bytes=0,
            )
        )

    row = ModelCatalog().read_node_budget("seen")
    assert row is not None
    assert row["mem_budget_bytes"] is None


# --- the seam every sizing decision already goes through -------------------


def test_the_sizing_denominator_follows_the_live_budget(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`configured_budget_bytes` is the one place the budget enters sizing, so
    the fit check, gated restart, eviction and the published snapshot all
    inherit a live edit without being touched."""
    from docie_bench.serving.resources import configured_budget_bytes

    _setting(monkeypatch, "8g")
    assert configured_budget_bytes() == 8 * GIB

    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)
    reset_budget_cache()

    assert configured_budget_bytes() == 6 * GIB


# --- the endpoint ----------------------------------------------------------


def _tenant(*, operator: bool = True) -> TenantContext:
    return TenantContext(
        tenant_id="ops@example.test", authenticated=True, operator=operator
    )


def _set_budget(
    node_id: str = NODE,
    *,
    budget_bytes: int | None = 6 * GIB,
    expected_version: int | None = None,
    operator: bool = True,
) -> dict[str, object]:
    from docie_bench.inngest.serving_api import NodeBudgetRequest, set_node_budget

    return asyncio.run(
        set_node_budget(
            node_id,
            NodeBudgetRequest(
                budget_bytes=budget_bytes, expected_version=expected_version
            ),
            tenant=_tenant(operator=operator),
        )
    )


def test_the_endpoint_sets_a_budget_the_node_then_sizes_against(
    node_home: None,
) -> None:
    result = _set_budget()

    assert result["mem_budget_bytes"] == 6 * GIB
    assert result["mem_budget_actor"] == "ops@example.test"
    reset_budget_cache()
    assert resolve_budget().budget_bytes == 6 * GIB


def test_a_studio_issued_key_cannot_shrink_a_nodes_share(node_home: None) -> None:
    """A client key calls the serving surfaces. Letting it shrink a node's
    budget would let it refuse everybody else's loads."""
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _set_budget(operator=False)

    assert excinfo.value.status_code == 403
    # Nothing was written at all -- not even the minimal row the write path
    # creates for a node that has never reported.
    assert ModelCatalog().read_node_budget(NODE) is None


def test_a_stale_version_is_a_409_carrying_the_version_that_won(
    node_home: None,
) -> None:
    from fastapi import HTTPException

    _set_budget()

    with pytest.raises(HTTPException) as excinfo:
        _set_budget(budget_bytes=4 * GIB, expected_version=None)

    assert excinfo.value.status_code == 409
    assert "version 1" in str(excinfo.value.detail)


def test_a_budget_of_zero_is_a_400_that_says_what_to_do_instead(
    node_home: None,
) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as excinfo:
        _set_budget(budget_bytes=0)

    assert excinfo.value.status_code == 400
    assert "None to clear" in str(excinfo.value.detail)


def test_clearing_through_the_endpoint_says_what_takes_over(
    node_home: None,
) -> None:
    first = _set_budget()

    result = _set_budget(
        budget_bytes=None, expected_version=int(first["mem_budget_version"])  # type: ignore[arg-type]
    )

    assert result["mem_budget_bytes"] is None
    assert "DOCIE_SERVING_MEM_BUDGET" in str(result["detail"])


def test_without_a_catalog_the_endpoint_points_at_the_node_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-node stack has nowhere to store a per-node budget, and the
    answer is the setting on that node rather than a 500."""
    from fastapi import HTTPException

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    db.dispose_engine()

    with pytest.raises(HTTPException) as excinfo:
        _set_budget()

    assert excinfo.value.status_code == 503
    assert "DOCIE_SERVING_MEM_BUDGET" in str(excinfo.value.detail)


# --- what the node publishes for the Studio --------------------------------


def _cgroup(root: Path, *, limit: int, current: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "memory.max").write_text(str(limit), encoding="utf-8")
    (root / "memory.current").write_text(str(current), encoding="utf-8")
    (root / "memory.stat").write_text("inactive_file 0\n", encoding="utf-8")
    return root


def test_a_smaller_budget_keeps_the_kernel_ceiling_visible(tmp_path: Path) -> None:
    """`total_bytes` becomes the budget, so the ceiling is carried separately."""
    from docie_bench.serving.resources import read_node_memory

    root = _cgroup(tmp_path / "cg", limit=8 * GIB, current=5 * GIB)

    memory = read_node_memory(cgroup_root=root, budget_bytes=4 * GIB)

    assert memory.total_bytes == 4 * GIB
    assert memory.ceiling_bytes == 8 * GIB
    assert memory.working_set_bytes == 5 * GIB
    assert memory.over_budget_bytes == 1 * GIB


def test_a_node_with_no_cgroup_limit_reports_no_ceiling(tmp_path: Path) -> None:
    from docie_bench.serving.resources import NodeMemory, read_node_memory

    memory = read_node_memory(
        cgroup_root=tmp_path / "absent",
        vm_reader=lambda: NodeMemory(total_bytes=64 * GIB, free_bytes=32 * GIB, source="vm"),
        budget_bytes=6 * GIB,
    )

    assert memory.ceiling_bytes is None
    assert memory.total_bytes == 6 * GIB


def test_the_snapshot_carries_the_budget_and_where_it_came_from(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    from docie_bench.serving.resources import NodeMemory, ResourceTracker

    _setting(monkeypatch, "6g")
    reading = NodeMemory(
        total_bytes=6 * GIB,
        free_bytes=1 * GIB,
        source="cgroup",
        budget_bytes=6 * GIB,
        working_set_bytes=5 * GIB,
        ceiling_bytes=8 * GIB,
    )
    tracker = ResourceTracker(memory_reader=lambda: reading)

    snapshot = tracker.observe_cycle([])

    assert snapshot.ceiling_bytes == 8 * GIB
    assert snapshot.working_set_bytes == 5 * GIB
    assert snapshot.budget_bytes == 6 * GIB
    assert snapshot.budget_source == "setting"
    assert snapshot.budget_stale is False


def test_published_budget_fields_reach_the_fleet_view(node_home: None) -> None:
    ModelCatalog().publish_node_snapshot(
        node_id=NODE,
        total_bytes=6 * GIB,
        free_bytes=GIB,
        source="cgroup",
        sum_rss_bytes=0,
        ceiling_bytes=8 * GIB,
        working_set_bytes=5 * GIB,
        effective_budget_bytes=6 * GIB,
        budget_source="catalog_cache",
        budget_stale=True,
    )

    row = next(r for r in ModelCatalog().list_node_snapshots() if r["node_id"] == NODE)

    assert row["ceiling_bytes"] == 8 * GIB
    assert row["working_set_bytes"] == 5 * GIB
    assert row["effective_budget_bytes"] == 6 * GIB
    assert row["budget_source"] == "catalog_cache"
    assert row["budget_stale"] is True


def test_a_node_publish_does_not_overwrite_the_operators_budget(node_home: None) -> None:
    """The node owns the measurements, the central API owns the override."""
    ModelCatalog().set_node_budget(NODE, budget_bytes=6 * GIB, expected_version=None)

    ModelCatalog().publish_node_snapshot(
        node_id=NODE, total_bytes=1, free_bytes=1, source="cgroup", sum_rss_bytes=0
    )

    assert ModelCatalog().read_node_budget(NODE)["mem_budget_bytes"] == 6 * GIB


# --- identity --------------------------------------------------------------


def test_resolving_a_budget_never_mints_a_node_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The api and the doctor size too, and only the publisher may mint."""
    from docie_bench.serving.node_identity import node_id_path

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.delenv("DOCIE_SERVING_NODE_ID", raising=False)
    _setting(monkeypatch, "8g")
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        resolution = resolve_budget()
    finally:
        db.dispose_engine()

    assert resolution == BudgetResolution(budget_bytes=8 * GIB, source="setting")
    assert not node_id_path().exists()


def test_another_nodes_cached_answer_is_not_this_nodes(
    node_home: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache sits in a volume the api and serving containers share. An
    answer written under another id must not stand in for this node's during
    an outage."""
    _setting(monkeypatch, "8g")
    write_budget_cache(2 * GIB, node_id="some-other-node")

    assert read_budget_cache(NODE) == "unknown"

    db.dispose_engine()
    reset_budget_cache()
    assert resolve_budget().budget_bytes == 8 * GIB
