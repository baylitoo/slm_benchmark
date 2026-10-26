"""Catching a misconfigured fleet node before it takes traffic (#538).

Every check here exists because the failure it catches is otherwise invisible.
The node starts, publishes capacity, accepts deploys — and then answers
requests for its models from another machine, or ignores every targeted deploy,
or holds its working set while logging a warning nobody reads.

Table-driven over the pure half, per ``preflight``'s own split: the facts are
constructed, so there is no node, no database and no socket to simulate.
"""

from __future__ import annotations

import pytest

from docie_bench.serving.preflight import (
    NodeFacts,
    Severity,
    collect_facts,
    evaluate,
    worst,
)


def _check(facts: NodeFacts, name: str) -> object:
    for check in evaluate(facts):
        if check.name == name:
            return check
    raise AssertionError(f"no check named {name!r}")


def _healthy() -> NodeFacts:
    """A node that should pass everything."""
    return NodeFacts(
        worker_role="serving",
        serving_home="/app/.serving",
        serving_home_writable=True,
        node_id="ovh-cpu-02",
        advertise_host="10.8.0.3",
        advertise_resolved=("10.8.0.3",),
        local_addresses=frozenset({"10.8.0.3", "127.0.0.1"}),
        port_range=(8088, 8107),
        ports_in_use=(),
        desired_apply=True,
        db_reachable=True,
        db_detail="reachable",
        db_clock_skew_s=0.2,
        inngest_base_url="http://10.8.0.1:8288",
        inngest_reachable=True,
        catalog_models=("lfm2.5-350m",),
        store_models=("lfm2.5-350m",),
        other_node_ids=("ovh-cpu-01",),
    )


def test_a_correctly_configured_node_raises_nothing_worse_than_a_note() -> None:
    checks = evaluate(_healthy())

    assert worst(checks) in {Severity.PASS, Severity.INFO}
    assert [c.name for c in checks if c.severity in {Severity.WARN, Severity.FAIL}] == []


# --- the failures that present as wrong answers ---------------------------


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "0.0.0.0", "::1", ""],  # noqa: S104 - rejected, not bound
)
def test_an_unroutable_advertise_host_fails(host: str) -> None:
    """THE #538 failure. The central api resolves the recorded endpoint in its
    OWN network, where every one of these is the api's own container -- so
    requests for this node's models are answered by the wrong machine, with a
    200."""
    facts = NodeFacts(**{**_healthy().__dict__, "advertise_host": host})

    check = _check(facts, "advertise host")

    assert check.severity is Severity.FAIL  # type: ignore[attr-defined]


def test_an_advertise_host_pointing_at_another_machine_warns() -> None:
    """Resolution is the fact, not the spelling: a perfectly dotted name that
    resolves somewhere else is exactly the trap."""
    facts = NodeFacts(
        **{
            **_healthy().__dict__,
            "advertise_host": "serving.internal",
            "advertise_resolved": ("10.8.0.9",),
        }
    )

    check = _check(facts, "advertise host")

    assert check.severity is Severity.WARN  # type: ignore[attr-defined]
    assert "OTHER machine" in check.consequence  # type: ignore[attr-defined]


def test_an_unresolvable_advertise_host_warns_rather_than_fails() -> None:
    """It may still resolve in the api's network, which is the one that
    matters -- so this is a suspicion, not a verdict."""
    facts = NodeFacts(**{**_healthy().__dict__, "advertise_resolved": None})

    assert _check(facts, "advertise host").severity is Severity.WARN  # type: ignore[attr-defined]


def test_a_duplicate_node_id_fails() -> None:
    """ModelPlacement.name is a primary key, so two nodes sharing an id
    overwrite each other: one node's replica goes unroutable while still
    holding its RAM."""
    facts = NodeFacts(
        **{**_healthy().__dict__, "other_node_ids": ("ovh-cpu-01", "ovh-cpu-02")}
    )

    check = _check(facts, "node id")

    assert check.severity is Severity.FAIL  # type: ignore[attr-defined]
    assert "ALSO published" in check.detail  # type: ignore[attr-defined]


def test_desired_apply_off_warns_and_names_what_is_inert() -> None:
    """The gap between "the code shipped" and "the fleet works". A node without
    it joins, reports healthy, and ignores every targeted deploy."""
    facts = NodeFacts(**{**_healthy().__dict__, "desired_apply": False})

    check = _check(facts, "desired-state apply")

    assert check.severity is Severity.WARN  # type: ignore[attr-defined]
    consequence = check.consequence  # type: ignore[attr-defined]
    assert "targeted at it" in consequence
    assert "auto-placed" in consequence
    assert "drain" in consequence
    # And what still works, so the operator does not over-read it.
    assert "Routing exclusion" in consequence


def test_a_role_without_serving_fails() -> None:
    facts = NodeFacts(**{**_healthy().__dict__, "worker_role": "worker"})

    assert _check(facts, "worker role").severity is Severity.FAIL  # type: ignore[attr-defined]


@pytest.mark.parametrize("role", ["serving", "all", "SERVING", " all "])
def test_the_roles_that_do_serve_pass(role: str) -> None:
    facts = NodeFacts(**{**_healthy().__dict__, "worker_role": role})

    assert _check(facts, "worker role").severity is Severity.PASS  # type: ignore[attr-defined]


def test_an_unwritable_serving_home_fails() -> None:
    facts = NodeFacts(**{**_healthy().__dict__, "serving_home_writable": False})

    assert _check(facts, "serving home").severity is Severity.FAIL  # type: ignore[attr-defined]


def test_an_unreachable_catalog_fails_but_says_serving_continues() -> None:
    """#527's non-negotiable: an unreachable control plane must not stop a
    loaded model from serving. The check has to fail without overstating it."""
    facts = NodeFacts(**{**_healthy().__dict__, "db_reachable": False, "db_detail": "nope"})

    check = _check(facts, "model catalog")

    assert check.severity is Severity.FAIL  # type: ignore[attr-defined]
    assert "already loaded" in check.consequence  # type: ignore[attr-defined]


def test_an_inverted_port_window_fails() -> None:
    facts = NodeFacts(**{**_healthy().__dict__, "port_range": (8107, 8088)})

    assert _check(facts, "port window").severity is Severity.FAIL  # type: ignore[attr-defined]


def test_ports_held_by_something_else_warn() -> None:
    facts = NodeFacts(**{**_healthy().__dict__, "ports_in_use": (8088, 8089)})

    check = _check(facts, "port window")

    assert check.severity is Severity.WARN  # type: ignore[attr-defined]
    assert "8088" in check.detail  # type: ignore[attr-defined]


def test_unreachable_inngest_warns_without_claiming_total_failure() -> None:
    """Desired state is read from the catalog, so a node with no Inngest still
    converges -- it just receives no events."""
    facts = NodeFacts(**{**_healthy().__dict__, "inngest_reachable": False})

    check = _check(facts, "inngest")

    assert check.severity is Severity.WARN  # type: ignore[attr-defined]
    assert "desired state" in check.consequence  # type: ignore[attr-defined]


# --- the three-state and no-blame checks ----------------------------------


def test_missing_blobs_are_a_note_not_a_warning() -> None:
    """The catalog is global and the blob store is node-local, so a node not
    holding every model is the normal state of a fleet."""
    facts = NodeFacts(
        **{**_healthy().__dict__, "catalog_models": ("lfm2.5-350m", "nuextract3")}
    )

    check = _check(facts, "blob store")

    assert check.severity is Severity.INFO  # type: ignore[attr-defined]
    assert "nuextract3" in check.detail  # type: ignore[attr-defined]


def test_an_unreadable_store_reports_unknown_rather_than_none_missing() -> None:
    """Absence of a comparison is not an absence of gaps."""
    facts = NodeFacts(**{**_healthy().__dict__, "store_models": None})

    check = _check(facts, "blob store")

    assert check.severity is Severity.INFO  # type: ignore[attr-defined]
    assert "cannot compare" in check.detail  # type: ignore[attr-defined]


def test_clock_skew_is_reported_without_being_graded() -> None:
    """Deliberately not a WARN against the staleness window. Staleness compares
    a timestamp this node WRITES against the clock of the process READING it --
    the central api -- so the database is only a third reference, and grading
    it would be chasing the wrong pair of machines."""
    facts = NodeFacts(**{**_healthy().__dict__, "db_clock_skew_s": 120.0})

    check = _check(facts, "clocks")

    assert check.severity is Severity.INFO  # type: ignore[attr-defined]
    assert "central api" in check.consequence  # type: ignore[attr-defined]


def test_worst_orders_failure_above_everything() -> None:
    assert worst(evaluate(NodeFacts())) is Severity.FAIL
    assert worst([]) is Severity.PASS


# --- the impure half ------------------------------------------------------


def test_collect_facts_survives_a_node_with_nothing_configured(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case it will actually be run in first: a fresh machine with no
    database. Every probe is contained, so this must return facts rather than
    raise -- a doctor that cannot run on a broken node is useless."""
    import docie_bench.storage.db as db

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    monkeypatch.delenv("INNGEST_BASE_URL", raising=False)
    monkeypatch.delenv("DOCIE_SERVING_DESIRED_APPLY", raising=False)
    db.dispose_engine()

    facts = collect_facts(probe_ports=False, probe_inngest=False)

    assert facts.db_reachable is False
    assert facts.db_detail  # a reason, not just a flag
    assert facts.db_clock_skew_s is None
    assert facts.catalog_models is None
    assert facts.desired_apply is False
    # And it still produced a full report rather than stopping at the failure.
    assert len(evaluate(facts)) == len(evaluate(_healthy()))


def test_collect_facts_reads_the_desired_apply_flag(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")

    assert collect_facts(probe_ports=False, probe_inngest=False).desired_apply is True


def test_a_free_port_window_is_detected_by_binding_it(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe binds rather than guesses, so it notices a process outside this
    container squatting the published range."""
    import socket

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_START", str(port))
    monkeypatch.setenv("DOCIE_SERVING_PORT_RANGE_END", str(port))
    monkeypatch.setenv("DOCIE_SERVING_BIND_HOST", "127.0.0.1")
    from docie_bench.settings import get_settings

    get_settings.cache_clear()
    try:
        facts = collect_facts(probe_ports=True, probe_inngest=False)
    finally:
        holder.close()
        get_settings.cache_clear()

    assert facts.ports_in_use == (port,)


def test_the_report_is_stable_so_a_bringup_script_can_diff_it() -> None:
    """Order is part of the contract: what the node is, then whether the fleet
    can see it, then what it will actually do."""
    names = [check.name for check in evaluate(_healthy())]

    assert names == [
        "worker role",
        "serving home",
        "node id",
        "advertise host",
        "model catalog",
        "inngest",
        "desired-state apply",
        "port window",
        "blob store",
        "clocks",
    ]


def test_the_skew_is_signed_so_the_direction_is_visible() -> None:
    ahead = NodeFacts(**{**_healthy().__dict__, "db_clock_skew_s": 5.0})
    behind = NodeFacts(**{**_healthy().__dict__, "db_clock_skew_s": -5.0})

    assert "+5.0s" in _check(ahead, "clocks").detail  # type: ignore[attr-defined]
    assert "-5.0s" in _check(behind, "clocks").detail  # type: ignore[attr-defined]


def test_a_reachable_catalog_is_reported_with_a_usable_skew(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Against a real database, including the naive-timestamp case: SQLite
    returns CURRENT_TIMESTAMP without an offset, and subtracting that from an
    aware `now` raises -- which the collector must absorb rather than losing
    every other check to it."""
    import docie_bench.storage.db as db

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path}/catalog.db")
    try:
        facts = collect_facts(probe_ports=False, probe_inngest=False)
    finally:
        db.dispose_engine()

    assert facts.db_reachable is True
    assert facts.catalog_models == ()
    assert facts.other_node_ids == ()
    assert isinstance(facts.db_clock_skew_s, float)
    # A naive stamp read as UTC can be hours off a local clock; the point is
    # that it is a NUMBER and nothing raised.
    assert _check(facts, "clocks").severity is Severity.INFO  # type: ignore[attr-defined]
