"""Node identity and per-node capacity rows (#529, phase 1 of #527).

Covers the issue's acceptance criteria directly: two nodes reconciling produce
two rows and neither overwrites the other, an id survives a restart on the same
volume, the pre-fleet single row migrates without losing its snapshot, and a
capacity number is never priced against an unidentified node.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import select

import docie_bench.storage.db as db
from docie_bench.serving.catalog import (
    LEGACY_SERVING_NODE_ROW_ID,
    ModelCatalog,
    ServingNode,
)
from docie_bench.serving.node_identity import (
    NODE_ID_ENV_VAR,
    InvalidNodeIdError,
    node_id_path,
    resolve_node_id,
    validate_node_id,
)

GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A private serving home per test, and no leaked cached id between tests."""
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.delenv(NODE_ID_ENV_VAR, raising=False)
    resolve_node_id.cache_clear()
    try:
        yield
    finally:
        resolve_node_id.cache_clear()


@pytest.fixture
def _sqlite_catalog(tmp_path: Path) -> Iterator[None]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()


# --- identity ------------------------------------------------------------


def test_a_generated_id_survives_a_restart_on_the_same_volume() -> None:
    """Acceptance: the id is stable across container restarts.

    A restart is a fresh process against the same serving home, which is a
    cleared cache and the same file — so the second resolve must READ, not mint.
    """
    first = resolve_node_id()
    assert node_id_path().read_text(encoding="utf-8").strip() == first

    resolve_node_id.cache_clear()  # the restart
    assert resolve_node_id() == first


def test_the_id_is_resolved_once_per_process() -> None:
    """The file is not re-read every reconcile cycle."""
    first = resolve_node_id()
    node_id_path().unlink()
    assert resolve_node_id() == first  # cached, no second read


def test_two_nodes_generate_different_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = resolve_node_id()

    # A second machine: its own serving home, its own process.
    resolve_node_id.cache_clear()
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "other-node"))
    assert resolve_node_id() != first


def test_an_operator_override_wins_and_is_not_persisted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(NODE_ID_ENV_VAR, "paris-1")

    assert resolve_node_id() == "paris-1"
    # Nothing written: the env var is the source of truth while it is set, so a
    # stale file can never shadow a changed override.
    assert not node_id_path().exists()


@pytest.mark.parametrize(
    "bad",
    [
        "x" * 33,  # longer than the String(32) primary key
        "-leading-dash",  # must start alphanumeric
        "has space",
        "has/slash",
        "",
    ],
)
def test_an_unusable_override_is_rejected_not_truncated(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently truncated id is a collision between two nodes — the bug this
    module exists to prevent — so the value is refused instead."""
    monkeypatch.setenv(NODE_ID_ENV_VAR, bad)
    if not bad:
        # An empty override is indistinguishable from unset; it falls through to
        # the generated id rather than failing the node.
        assert resolve_node_id()
        return
    with pytest.raises(InvalidNodeIdError):
        resolve_node_id()


def test_a_corrupted_id_file_regenerates_instead_of_failing_the_node() -> None:
    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a valid id!!", encoding="utf-8")

    regenerated = resolve_node_id()

    assert validate_node_id(regenerated)
    assert path.read_text(encoding="utf-8").strip() == regenerated


def test_a_racing_second_writer_adopts_the_winners_id() -> None:
    """Two serving processes overlapping on one volume (a replaced container
    inside the lease window) must agree, not overwrite each other."""
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    winner = _persist(path, "aaaa1111")

    assert _persist(path, "bbbb2222") == winner
    assert path.read_text(encoding="utf-8").strip() == winner


# --- per-node rows -------------------------------------------------------


@pytest.mark.usefixtures("_sqlite_catalog")
def test_two_nodes_produce_two_rows_and_neither_overwrites_the_other() -> None:
    """Acceptance: the headline criterion of #529."""
    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        node_id="node-a",
        total_bytes=8 * GIB,
        free_bytes=6 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    catalog.publish_node_snapshot(
        node_id="node-b",
        total_bytes=64 * GIB,
        free_bytes=48 * GIB,
        source="cgroup",
        sum_rss_bytes=2 * GIB,
    )

    by_id = {row["node_id"]: row for row in catalog.list_node_snapshots()}

    assert set(by_id) == {"node-a", "node-b"}
    assert by_id["node-a"]["free_bytes"] == 6 * GIB
    assert by_id["node-b"]["free_bytes"] == 48 * GIB  # the big node is not flattened


@pytest.mark.usefixtures("_sqlite_catalog")
def test_the_prefleet_row_is_adopted_so_its_history_is_not_lost() -> None:
    """Acceptance: the existing single-node deployment migrates without losing
    its capacity snapshot."""
    catalog = ModelCatalog()
    with db.session_scope() as session:
        assert session is not None
        session.add(
            ServingNode(
                id=LEGACY_SERVING_NODE_ROW_ID,
                total_bytes=8 * GIB,
                free_bytes=3 * GIB,
                source="vm",
                sum_rss_bytes=GIB,
                reclaimable_bytes=0,
            )
        )

    catalog.publish_node_snapshot(
        node_id="node-a",
        total_bytes=8 * GIB,
        free_bytes=5 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )

    rows = catalog.list_node_snapshots()
    # Re-keyed in place: one row, under the real node id — not a fresh row
    # beside an orphaned "node".
    assert [row["node_id"] for row in rows] == ["node-a"]
    assert rows[0]["free_bytes"] == 5 * GIB


@pytest.mark.usefixtures("_sqlite_catalog")
def test_adoption_is_skipped_when_this_node_already_has_a_row() -> None:
    """A pre-fleet container still publishing during a rolling restart keeps
    re-creating the legacy row; re-keying it onto an id that already exists
    would violate the primary key and fail every publish."""
    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        node_id="node-a",
        total_bytes=8 * GIB,
        free_bytes=6 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    with db.session_scope() as session:  # the outgoing old container
        assert session is not None
        session.add(
            ServingNode(
                id=LEGACY_SERVING_NODE_ROW_ID,
                total_bytes=8 * GIB,
                free_bytes=1 * GIB,
                source="vm",
                sum_rss_bytes=GIB,
                reclaimable_bytes=0,
            )
        )

    catalog.publish_node_snapshot(  # must not raise
        node_id="node-a",
        total_bytes=8 * GIB,
        free_bytes=4 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )

    with db.session_scope() as session:
        assert session is not None
        assert {row.id for row in session.scalars(select(ServingNode)).all()} == {
            "node-a",
            LEGACY_SERVING_NODE_ROW_ID,
        }


@pytest.mark.usefixtures("_sqlite_catalog")
def test_the_unqualified_read_refuses_to_pick_one_node_of_several() -> None:
    """``get_node_snapshot()`` without an id answers only while there is a
    single answer. Its callers PRICE with the result (RAM admission), and
    choosing one of N nodes' free RAM is the "whichever reconciled last" guess
    this issue removes — it can refuse a deploy that fits on another node."""
    catalog = ModelCatalog()
    catalog.publish_node_snapshot(
        node_id="node-a",
        total_bytes=8 * GIB,
        free_bytes=6 * GIB,
        source="cgroup",
        sum_rss_bytes=GIB,
    )
    assert catalog.get_node_snapshot() is not None  # one node: unambiguous

    catalog.publish_node_snapshot(
        node_id="node-b",
        total_bytes=64 * GIB,
        free_bytes=48 * GIB,
        source="cgroup",
        sum_rss_bytes=0,
    )

    assert catalog.get_node_snapshot() is None
    # Acceptance: admission can still be made against a NAMED node.
    named = catalog.get_node_snapshot("node-b")
    assert named is not None
    assert named["free_bytes"] == 48 * GIB


# --- what the api reports -----------------------------------------------


@pytest.mark.usefixtures("_sqlite_catalog")
def test_the_resources_view_says_which_node_it_is_showing() -> None:
    from docie_bench.inngest.serving_api import serving_resources

    catalog = ModelCatalog()
    for node_id, free in (("node-a", 6 * GIB), ("node-b", 48 * GIB)):
        catalog.publish_node_snapshot(
            node_id=node_id,
            total_bytes=64 * GIB,
            free_bytes=free,
            source="cgroup",
            sum_rss_bytes=0,
        )

    payload = asyncio.run(serving_resources())

    assert payload["observed_available"] is True
    assert payload["node_count"] == 2
    assert payload["node"]["node_id"] in {"node-a", "node-b"}
    # The number describes one machine, and the view says so rather than
    # implying it is the fleet's capacity.
    assert payload["node"]["node_id"] in payload["detail"]
    assert "not the fleet total" in payload["detail"]


@pytest.mark.usefixtures("_sqlite_catalog")
def test_sizing_withholds_a_number_it_cannot_attribute_to_a_node() -> None:
    from docie_bench.inngest.serving_api import _sizing_inputs

    catalog = ModelCatalog()
    for node_id in ("node-a", "node-b"):
        catalog.publish_node_snapshot(
            node_id=node_id,
            total_bytes=64 * GIB,
            free_bytes=48 * GIB,
            source="cgroup",
            sum_rss_bytes=0,
        )

    _models, snapshot, _placements, detail = _sizing_inputs()

    assert snapshot is None  # fail-open, per admit_replica_ram's contract
    assert detail is not None
    assert "2 serving nodes" in detail
