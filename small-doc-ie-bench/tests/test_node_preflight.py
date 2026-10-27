"""Catching a misconfigured fleet node before it takes traffic (#538).

Every check here exists because the failure it catches is otherwise invisible.
The node starts, publishes capacity, accepts deploys — and then answers
requests for its models from another machine, or ignores every targeted deploy,
or holds its working set while logging a warning nobody reads.

Table-driven over the pure half, per ``preflight``'s own split: the facts are
constructed, so there is no node, no database and no socket to simulate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docie_bench.serving.preflight import (
    NodeFacts,
    Severity,
    collect_facts,
    evaluate,
    worst,
)

MODEL = "lfm2.5-350m"


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


def test_the_blob_comparison_actually_runs_against_this_nodes_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression this file missed the first time.

    ``collect_facts`` built its ModelStore without the required root, which
    raised a TypeError that the handler turned into ``store_models=None`` --
    the same value an unreadable store produces. So the blob check reported
    "cannot compare" on every node, and the existing tests agreed with it:
    the pure ones set ``store_models`` by hand, and the impure one asserted the
    degraded path. A three-state value makes a broken probe look like an honest
    "we do not know", which is why this asserts the WORKING path.
    """
    import docie_bench.storage.db as db
    from docie_bench.serving.model_store import ModelStore

    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    gguf = tmp_path / "weights.gguf"
    gguf.write_bytes(b"GGUF-model-weights")
    ModelStore(home / "models").add_gguf(
        name=MODEL, family="openai_chat", model_gguf=gguf
    )
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path}/catalog.db")
    try:
        facts = collect_facts(probe_ports=False, probe_inngest=False)
    finally:
        db.dispose_engine()

    assert facts.store_models == (MODEL,)
    # An empty catalog and a held blob: nothing is missing, so this must be a
    # PASS rather than the "cannot compare" note.
    assert _check(facts, "blob store").severity is Severity.PASS  # type: ignore[attr-defined]


def test_an_empty_store_is_empty_rather_than_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """() and None are different answers, and the whole blob check turns on
    which one a fresh node reports."""
    import docie_bench.storage.db as db

    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path}/catalog.db")
    try:
        facts = collect_facts(probe_ports=False, probe_inngest=False)
    finally:
        db.dispose_engine()

    assert facts.store_models == ()
    assert facts.store_models is not None


def test_a_node_missing_a_catalogued_model_is_told_which_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check's actual job on a fleet: say what to seed here."""
    import docie_bench.storage.db as db
    from docie_bench.serving.catalog import ModelCatalog
    from docie_bench.serving.model_store import StoreEntry

    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path}/catalog.db")
    try:
        ModelCatalog().upsert(
            StoreEntry(
                name="nuextract3",
                family="openai_chat",
                model_path=Path("/models/nuextract3/model.gguf"),
            )
        )
        facts = collect_facts(probe_ports=False, probe_inngest=False)
    finally:
        db.dispose_engine()

    check = _check(facts, "blob store")
    assert check.severity is Severity.INFO  # type: ignore[attr-defined]
    assert "nuextract3" in check.detail  # type: ignore[attr-defined]


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


def test_the_port_probe_is_off_by_default_because_it_is_container_only() -> None:
    """Binding the deploy window only means something in the serving
    container's own network namespace. Run from the host it reports ports held
    on unrelated interfaces as conflicts the runtime will never see, and a
    check that cries wolf on first use costs the credibility of every check
    beside it."""
    import inspect

    default = inspect.signature(collect_facts).parameters["probe_ports"].default

    assert default is False


def test_an_unprobed_port_window_says_how_to_probe_it(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))

    facts = collect_facts(probe_inngest=False)

    assert facts.ports_in_use is None
    assert "--probe-ports" in _check(facts, "port window").detail  # type: ignore[attr-defined]


def test_a_generated_node_id_says_that_this_call_minted_it(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The doctor persists an id on a fresh node, exactly as the reconciler's
    first publish would. An operator should see which of the three it got
    rather than reading a hex string of unknown origin."""
    from docie_bench.serving.node_identity import resolve_node_id

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    monkeypatch.delenv("DOCIE_SERVING_NODE_ID", raising=False)
    resolve_node_id.cache_clear()
    try:
        first = collect_facts(probe_inngest=False)
        resolve_node_id.cache_clear()
        second = collect_facts(probe_inngest=False)
    finally:
        resolve_node_id.cache_clear()

    assert first.node_id_source == "generated"
    # The id is now on disk, so the next run reports it as persisted -- and it
    # is the SAME id, which is the property that matters.
    assert second.node_id_source == "persisted"
    assert second.node_id == first.node_id
    assert "generated and persisted just now" in _check(first, "node id").detail  # type: ignore[attr-defined]


def test_an_explicit_node_id_is_reported_as_coming_from_the_environment(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env var wins over the persisted file, so an operator who sets it
    after a first doctor run is not silently stuck with a generated id."""
    from docie_bench.serving.node_identity import resolve_node_id

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path))
    monkeypatch.delenv("DOCIE_SERVING_NODE_ID", raising=False)
    resolve_node_id.cache_clear()
    collect_facts(probe_inngest=False)  # mints and persists one

    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", "ovh-cpu-07")
    resolve_node_id.cache_clear()
    try:
        facts = collect_facts(probe_inngest=False)
    finally:
        resolve_node_id.cache_clear()

    assert facts.node_id == "ovh-cpu-07"
    assert facts.node_id_source == "env"


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
