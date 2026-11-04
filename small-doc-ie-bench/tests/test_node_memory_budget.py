"""A per-node soft memory budget (#539, part 1).

The serving nodes are SHARED hosts: each also runs other applications, so
DocIE is a tenant rather than the owner. Free host RAM is therefore not
DocIE's to take -- whatever is free now belongs to whichever co-tenant asks
for it next, and a denominator built from it admits a model into memory
another application is about to need, leaving the OOM killer to pick the
loser.

Two numbers, not one: the hard ceiling stays cgroup ``memory.max``, which
stops a DocIE bug from eating a neighbour's memory, and the soft budget is
DocIE's declared share, which admission sizes against. The reading layer is
what these cover.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.serving.resources import (
    NodeMemory,
    read_cgroup_working_set,
    read_node_memory,
)
from docie_bench.settings import _parse_size_bytes

GIB = 1024**3


def _write_cgroup(
    root: Path, *, limit: str | None, current: str | None, stat: str | None = None
) -> None:
    if limit is not None:
        (root / "memory.max").write_text(limit, encoding="ascii")
    if current is not None:
        (root / "memory.current").write_text(current, encoding="ascii")
    if stat is not None:
        (root / "memory.stat").write_text(stat, encoding="ascii")


def _vm_reader() -> NodeMemory:
    return NodeMemory(total_bytes=32 * GIB, free_bytes=20 * GIB, source="vm")


def _read(root: Path, budget: int | None) -> NodeMemory:
    return read_node_memory(cgroup_root=root, vm_reader=_vm_reader, budget_bytes=budget)


# --- with a kernel-enforced ceiling ---------------------------------------


def test_a_budget_below_the_hard_ceiling_becomes_the_denominator(tmp_path: Path) -> None:
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=str(2 * GIB))

    reading = _read(tmp_path, 6 * GIB)

    assert reading.total_bytes == 6 * GIB
    assert reading.free_bytes == 4 * GIB  # 6 budget - 2 working set
    assert reading.source == "cgroup"  # the ceiling is still kernel-enforced
    assert reading.budget_bytes == 6 * GIB


def test_a_budget_above_the_hard_ceiling_changes_no_number(tmp_path: Path) -> None:
    """The kernel limit is the real guardrail; a larger budget cannot raise it.
    It is still recorded, so the Studio can show what was asked for."""
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=str(2 * GIB))

    reading = _read(tmp_path, 16 * GIB)

    assert reading.total_bytes == 8 * GIB
    assert reading.free_bytes == 6 * GIB
    assert reading.budget_bytes == 16 * GIB


def test_no_budget_leaves_the_reading_exactly_as_it_was(tmp_path: Path) -> None:
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=str(2 * GIB))

    reading = _read(tmp_path, None)

    assert (reading.total_bytes, reading.free_bytes, reading.source) == (
        8 * GIB,
        6 * GIB,
        "cgroup",
    )
    assert reading.budget_bytes is None
    assert reading.over_budget_bytes == 0


# --- the case the issue exists for: no cgroup limit ------------------------


def test_with_no_cgroup_limit_a_budget_is_exact_not_a_cap(tmp_path: Path) -> None:
    """The real deployment case, and the unsafe one before this.

    ``memory.max`` holding the ``max`` sentinel used to discard the WHOLE
    reading, falling back to a host-wide VM view that -- in the existing
    design's own words -- "says nothing about what THIS container may use". But
    only the ceiling was missing: ``memory.current`` was readable the entire
    time. With the budget supplying the ceiling, the node gets a real
    denominator.
    """
    _write_cgroup(tmp_path, limit="max", current=str(2 * GIB))

    reading = _read(tmp_path, 6 * GIB)

    assert reading.total_bytes == 6 * GIB
    assert reading.free_bytes == 4 * GIB  # exact: budget - this container's usage
    assert reading.working_set_bytes == 2 * GIB
    # Still "vm": nothing but DocIE enforces this ceiling.
    assert reading.source == "vm"


def test_without_a_budget_the_sentinel_still_falls_back_to_the_vm_view(
    tmp_path: Path,
) -> None:
    """Unchanged behaviour where no budget is configured."""
    _write_cgroup(tmp_path, limit="max", current=str(2 * GIB))

    reading = _read(tmp_path, None)

    assert reading.total_bytes == 32 * GIB  # the VM reader's numbers
    assert reading.source == "vm"
    assert reading.budget_bytes is None


def test_the_working_set_excludes_reclaimable_page_cache_under_a_budget(
    tmp_path: Path,
) -> None:
    """llama.cpp mmaps multi-GB GGUFs and those pages linger as cache long
    after an unload, so the budget path must use the same working-set
    accounting the ceiling path already does."""
    _write_cgroup(
        tmp_path, limit="max", current=str(5 * GIB), stat=f"inactive_file {3 * GIB}\n"
    )

    reading = _read(tmp_path, 6 * GIB)

    assert reading.working_set_bytes == 2 * GIB
    assert reading.free_bytes == 4 * GIB
    assert reading.reclaimable_bytes == 3 * GIB


# --- over budget -----------------------------------------------------------


def test_over_budget_reports_free_zero_and_the_overage(tmp_path: Path) -> None:
    """``free_bytes`` clamps at 0, so the overage has to be carried separately
    or the node can only say "full" and not "full by how much"."""
    _write_cgroup(tmp_path, limit=str(16 * GIB), current=str(9 * GIB))

    reading = _read(tmp_path, 6 * GIB)

    assert reading.free_bytes == 0
    assert reading.over_budget_bytes == 3 * GIB
    assert reading.working_set_bytes == 9 * GIB


def test_a_reading_with_no_budget_is_never_over_budget(tmp_path: Path) -> None:
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=str(7 * GIB))

    assert _read(tmp_path, None).over_budget_bytes == 0


# --- the inexact case, named as such --------------------------------------


def test_with_no_cgroup_files_at_all_the_budget_only_caps(tmp_path: Path) -> None:
    """A bare dev host. A host-wide "used" counts co-tenants and cannot be
    subtracted from DocIE's own allowance, so the budget bounds the
    denominator without pretending to know DocIE's share."""
    reading = _read(tmp_path, 6 * GIB)

    assert reading.total_bytes == 6 * GIB
    assert reading.free_bytes == 6 * GIB  # capped from the VM reader's 20 GiB
    assert reading.source == "vm"
    assert reading.budget_bytes == 6 * GIB
    assert reading.working_set_bytes is None  # unknown, and not invented


def test_an_authoritative_limit_with_unreadable_usage_keeps_its_contract(
    tmp_path: Path,
) -> None:
    """``read_cgroup_memory`` degrades an unreadable ``memory.current`` to
    0 used rather than discarding the limit. The budget must not change that --
    and the budget-only path is deliberately stricter, since with no ceiling an
    unmeasurable working set cannot be exact."""
    _write_cgroup(tmp_path, limit=str(8 * GIB), current=None)

    reading = _read(tmp_path, 6 * GIB)

    assert reading.total_bytes == 6 * GIB
    assert reading.free_bytes == 6 * GIB
    assert reading.source == "cgroup"

    _write_cgroup(tmp_path, limit="max", current=None)
    assert read_cgroup_working_set(tmp_path) is None
    assert _read(tmp_path, 6 * GIB).source == "vm"


# --- the knob --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("6g", 6 * GIB),
        ("6144m", 6 * GIB),
        ("6gib", 6 * GIB),
        ("8G", 8 * GIB),
        ("2.5g", int(2.5 * GIB)),
        ("1024", 1024),
        ("", 0),
        ("  ", 0),
        ("nonsense", 0),
        ("0", 0),
        ("-4g", 0),
    ],
)
def test_the_budget_accepts_the_units_an_operator_already_writes(
    text: str, expected: int
) -> None:
    """``mem_limit: 8g`` is how the hard ceiling is already written in the
    compose file, so the soft budget beside it takes the same units. An
    unparseable value reads as unset: a typo in an optional knob must not stop
    a node from serving."""
    assert _parse_size_bytes(text) == expected


def test_six_gigabytes_is_the_same_number_however_it_is_spelled() -> None:
    assert _parse_size_bytes("6g") == _parse_size_bytes("6144m") == 6 * GIB
