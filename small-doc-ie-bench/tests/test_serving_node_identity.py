"""Node identity and per-node capacity rows (#529, phase 1 of #527).

Covers the issue's acceptance criteria directly: two nodes reconciling produce
two rows and neither overwrites the other, an id survives a restart on the same
volume, the pre-fleet single row migrates without losing its snapshot, and a
capacity number is never priced against an unidentified node.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

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
    NodeIdUnreadableError,
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


def _unreadable_node_id(failures: list[int] | None = None):
    """Make reads of ``node_id`` raise EIO, leaving every other read alone.

    chmod is not a usable lever here: the suite runs on Windows too.
    """
    real = Path.read_text
    budget = failures if failures is not None else [-1]

    def flaky(self: Path, *args: object, **kwargs: object) -> str:
        if self.name == "node_id" and budget[0] != 0:
            budget[0] -= 1
            raise OSError(5, "Input/output error")
        return real(self, *args, **kwargs)  # type: ignore[arg-type]

    return mock.patch.object(Path, "read_text", flaky)


def test_an_unreadable_id_file_is_not_mistaken_for_an_absent_one() -> None:
    """The bug: ``_read_persisted`` returned None for BOTH absent and
    unreadable, so ``os.link`` raising FileExistsError plus a None read read as
    "a file is there but it is garbage" -- and the node overwrote a perfectly
    good identity with a fresh one, orphaning the ``serving_node`` and placement
    rows it had already published.
    """
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = "a" * 32
    path.write_text(original, encoding="utf-8")

    with _unreadable_node_id(), pytest.raises(NodeIdUnreadableError, match="could not be read"):
        _persist(path, "b" * 32)

    assert path.read_text(encoding="utf-8").strip() == original


def test_resolving_under_an_unreadable_file_refuses_rather_than_re_minting() -> None:
    """Every caller resolves inside a broad ``except Exception`` that logs and
    continues, so refusing costs a publish cycle, not the container."""
    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = "a" * 32
    path.write_text(original, encoding="utf-8")

    with _unreadable_node_id(), pytest.raises(NodeIdUnreadableError):
        resolve_node_id()

    assert path.read_text(encoding="utf-8").strip() == original


def test_a_transient_read_error_self_heals_under_the_real_id() -> None:
    """Why refusing beats re-minting: ``lru_cache`` does not cache exceptions,
    so the next cycle retries and gets the id that was always on disk. Re-minting
    would have been permanent."""
    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = "a" * 32
    path.write_text(original, encoding="utf-8")

    with _unreadable_node_id(failures=[2]):
        for _ in range(2):
            with pytest.raises(NodeIdUnreadableError):
                resolve_node_id()
        assert resolve_node_id() == original

    assert resolve_node_id() == original  # and it is cached from then on


def test_an_invalid_id_file_is_still_regenerated_not_refused() -> None:
    """The state that must NOT change: readable-but-corrupt content is a
    documented regenerate, since refusing to serve over it would be worse."""
    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not a valid id!!", encoding="utf-8")

    regenerated = resolve_node_id()

    assert validate_node_id(regenerated)
    assert path.read_text(encoding="utf-8").strip() == regenerated


def test_the_id_is_flushed_to_disk_before_it_is_linked() -> None:
    """The id is published to Postgres as soon as it is minted, so a power loss
    that takes the buffered write but not the placement rows leaves the node
    re-minting against rows the old id still owns."""
    from docie_bench.serving.node_identity import _persist

    calls: list[str] = []
    real_fsync, real_link = os.fsync, os.link

    def fsync(fd: int) -> None:
        calls.append("fsync")
        real_fsync(fd)

    def link(src: object, dst: object) -> None:
        calls.append("link")
        real_link(src, dst)  # type: ignore[arg-type]

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with mock.patch.object(os, "fsync", fsync), mock.patch.object(os, "link", link):
        _persist(path, "a" * 32)

    assert calls[: calls.index("link")] == ["fsync"]


def test_a_filesystem_without_hard_links_still_adopts_the_winners_id() -> None:
    """Some bind mounts reject ``os.link``. That fallback used to rename
    unconditionally, so on those filesystems it clobbered the very id the
    FileExistsError branch beside it was written to adopt."""
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    winner = "a" * 32
    path.write_text(winner, encoding="utf-8")

    with mock.patch.object(os, "link", side_effect=OSError(1, "operation not permitted")):
        assert _persist(path, "b" * 32) == winner

    assert path.read_text(encoding="utf-8").strip() == winner


def test_a_failed_persist_leaves_no_staging_file() -> None:
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a" * 32, encoding="utf-8")

    with _unreadable_node_id(), pytest.raises(NodeIdUnreadableError):
        _persist(path, "b" * 32)

    assert [p.name for p in path.parent.iterdir() if p.name != "node_id"] == []


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

def test_simultaneous_creators_agree_when_hardlinks_are_unavailable() -> None:
    """The race the earlier fallback left open.

    Some bind mounts refuse os.link, and the fallback read the file, found
    nothing and renamed its own in. Two creators could both observe absence
    and both write: each returned its OWN id while the disk kept only the
    loser's, and because resolution is cached per process the two then
    disagreed about who this node is for the rest of their lives.

    The existing coverage only had a second writer ADOPT an id already on
    disk, which is the easy half -- the winner is already decided there.
    """
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    start = threading.Barrier(8)
    returned: list[str] = []
    errors: list[BaseException] = []

    def create(worker: int) -> None:
        start.wait()
        try:
            returned.append(_persist(path, f"{worker:032d}"))
        except BaseException as exc:  # noqa: BLE001 - asserted below
            errors.append(exc)

    with mock.patch.object(os, "link", side_effect=OSError(1, "no hardlinks here")):
        threads = [threading.Thread(target=create, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == [], f"creation failed: {errors[:3]}"
    on_disk = path.read_text(encoding="utf-8").strip()
    assert len(set(returned)) == 1, f"creators disagreed about this node: {set(returned)}"
    assert returned[0] == on_disk, "a creator returned an id the disk does not hold"


def test_the_winner_is_whoever_the_filesystem_let_create_the_file() -> None:
    """And the losers adopt it rather than minting their own, with hardlinks
    unavailable so the exclusive-creation path is the one under test."""
    from docie_bench.serving.node_identity import _persist

    path = node_id_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    with mock.patch.object(os, "link", side_effect=OSError(1, "no hardlinks here")):
        winner = _persist(path, "a" * 32)
        loser = _persist(path, "b" * 32)

    assert loser == winner
    assert path.read_text(encoding="utf-8").strip() == winner

