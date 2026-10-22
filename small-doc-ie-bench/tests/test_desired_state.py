"""Desired state in Postgres: generations and snapshots (#534, phase 2).

Two generation counters doing different jobs, which is the part worth testing
rather than assuming:

* **per row** — optimistic concurrency between *writers*, so two central API
  calls racing on one deployment cannot interleave;
* **per node** — lets a *reader* decide in one read whether the catalog is
  provably current for it, which is what `convergence.decide_convergence`
  consumes.

Nothing reads this in production yet; the reconciler side is separate.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import docie_bench.storage.db as db
from docie_bench.serving.catalog import (
    DesiredStateConflictError,
    ModelCatalog,
    ServingNode,
)

NODE = "ovh-cpu-02"
MODEL = "lfm2.5-350m"


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[ModelCatalog]:
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield ModelCatalog()
    finally:
        db.dispose_engine()


def _write(
    catalog: ModelCatalog, *, expected: int | None = None, state: str = "running"
) -> dict[str, object]:
    return catalog.set_desired_deployment(
        NODE, MODEL, desired_state=state, spec={"port": 8088}, expected_generation=expected
    )


# --- per-row generation: writer concurrency ------------------------------


def test_a_first_write_starts_at_generation_one(catalog: ModelCatalog) -> None:
    assert _write(catalog)["generation"] == 1


def test_a_second_write_must_carry_the_generation_it_read(
    catalog: ModelCatalog,
) -> None:
    _write(catalog)

    assert _write(catalog, expected=1)["generation"] == 2


def test_a_stale_write_loses_instead_of_overwriting(catalog: ModelCatalog) -> None:
    """Two central API calls read generation 1; the second to land must lose.

    Overwriting would let the loser's intent silently replace the winner's --
    the interleaving this counter exists to prevent.
    """
    _write(catalog)
    _write(catalog, expected=1)  # now at 2

    with pytest.raises(DesiredStateConflictError, match="generation 2"):
        _write(catalog, expected=1)


def test_claiming_a_row_does_not_exist_when_it_does_loses(
    catalog: ModelCatalog,
) -> None:
    """`expected_generation=None` means "this should not exist yet", so it must
    not silently become an overwrite."""
    _write(catalog)

    with pytest.raises(DesiredStateConflictError):
        _write(catalog, expected=None)


def test_a_stale_delete_loses(catalog: ModelCatalog) -> None:
    _write(catalog)
    _write(catalog, expected=1)  # now at 2

    with pytest.raises(DesiredStateConflictError):
        catalog.delete_desired_deployment(NODE, MODEL, expected_generation=1)


def test_deleting_an_absent_row_loses_rather_than_succeeding_quietly(
    catalog: ModelCatalog,
) -> None:
    with pytest.raises(DesiredStateConflictError):
        catalog.delete_desired_deployment(NODE, MODEL, expected_generation=1)


# --- per-node generation: reader staleness -------------------------------


def test_the_node_generation_advances_on_every_write(catalog: ModelCatalog) -> None:
    assert _write(catalog)["node_desired_generation"] == 1
    assert _write(catalog, expected=1)["node_desired_generation"] == 2


def test_the_node_generation_advances_on_delete_too(catalog: ModelCatalog) -> None:
    """A removal is a change of intent, so a reader must see the generation
    move -- otherwise a node would never learn to stop something."""
    _write(catalog)

    assert catalog.delete_desired_deployment(NODE, MODEL, expected_generation=1) == 2


def test_the_two_counters_cannot_disagree(catalog: ModelCatalog) -> None:
    """Bumped in the same transaction, so a reader seeing a newer node
    generation is guaranteed to see the row that caused it."""
    _write(catalog)

    snapshot = catalog.desired_snapshot(NODE)

    assert snapshot["generation"] == 1
    assert snapshot["deployments"][0]["generation"] == 1


def test_a_node_with_no_desired_state_reports_generation_none(
    catalog: ModelCatalog,
) -> None:
    """None, not 0: `decide_convergence` distinguishes "never written" from a
    node that had rows and lost them."""
    snapshot = catalog.desired_snapshot("never-written")

    assert snapshot["generation"] is None
    assert snapshot["deployments"] == []


def test_desired_state_can_be_written_before_a_node_ever_reports(
    catalog: ModelCatalog,
) -> None:
    """An operator may stage a deployment for a node that has not started. The
    capacity row is created minimally rather than failing, and must not pretend
    to carry measurements."""
    _write(catalog)

    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, NODE)
        assert row is not None
        assert row.source == "unreported"
        assert row.total_bytes == 0


def test_a_real_publish_overwrites_the_placeholder_capacity(
    catalog: ModelCatalog,
) -> None:
    """The placeholder must not survive as a zero-capacity node, which the
    sizing engine would read as "no room"."""
    _write(catalog)

    catalog.publish_node_snapshot(
        node_id=NODE,
        total_bytes=8 * 1024**3,
        free_bytes=4 * 1024**3,
        source="cgroup",
        sum_rss_bytes=1024**3,
    )

    (node,) = [row for row in catalog.list_node_snapshots() if row["node_id"] == NODE]
    assert node["source"] == "cgroup"
    assert node["total_bytes"] == 8 * 1024**3
    # And the desired generation is untouched by a capacity publish.
    assert node["desired_generation"] == 1


# --- the snapshot --------------------------------------------------------


def test_a_snapshot_is_scoped_to_one_node(catalog: ModelCatalog) -> None:
    """A node converges only its own rows; seeing another node's would make it
    tear down or start the wrong things."""
    _write(catalog)
    catalog.set_desired_deployment(
        "other-node", "nuextract3", desired_state="running", spec={}
    )

    snapshot = catalog.desired_snapshot(NODE)

    assert [row["name"] for row in snapshot["deployments"]] == [MODEL]


def test_the_spec_round_trips(catalog: ModelCatalog) -> None:
    catalog.set_desired_deployment(
        NODE, MODEL, desired_state="running", spec={"port": 8088, "ctx": 4096}
    )

    (row,) = catalog.desired_snapshot(NODE)["deployments"]

    assert row["spec"] == {"port": 8088, "ctx": 4096}


def test_stopped_is_a_state_but_removed_is_a_deletion(catalog: ModelCatalog) -> None:
    """"Should not be running" and "no longer desired at all" call for
    different actions, so they are represented differently."""
    _write(catalog, state="stopped")

    (row,) = catalog.desired_snapshot(NODE)["deployments"]
    assert row["desired_state"] == "stopped"

    catalog.delete_desired_deployment(NODE, MODEL, expected_generation=1)
    assert catalog.desired_snapshot(NODE)["deployments"] == []
