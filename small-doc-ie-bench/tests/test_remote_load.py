"""Loading a deployment that another node owns (#590)."""

from __future__ import annotations

import asyncio
import datetime as dt
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

import docie_bench.storage.db as db
from docie_bench.serving.applied_state import read_applied_state
from docie_bench.serving.catalog import ModelCatalog, ServingNode
from docie_bench.serving.convergence import plan_convergence
from docie_bench.serving.drain import DRAIN_GRACE_CEILING_S
from docie_bench.serving.supervisor import DesiredState
from docie_bench.settings import get_settings

MODEL = "lfm2.5-350m"
RECORD = "lfm2.5-350m-ab12"
OWNER = "node-b"


class _Launch:
    timeout_seconds = None


class _Spec:
    def __init__(self, desired_state: DesiredState) -> None:
        self.launch = _Launch()
        self.desired_state = desired_state


class _Record:
    def __init__(self, desired_state: DesiredState) -> None:
        self.spec = _Spec(desired_state)


class _Lock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


class _Supervisor:
    lock = _Lock()

    def __init__(self, desired_state: DesiredState = DesiredState.STOPPED) -> None:
        self._records = {RECORD: _Record(desired_state)}
        self.loaded: list[str] = []
        self.stopped: list[str] = []

    def records(self) -> dict[str, _Record]:
        return self._records

    def load(self, name: str) -> None:
        self.loaded.append(name)
        self._records[name].spec.desired_state = DesiredState.RUNNING

    def stop(self, name: str) -> None:
        self.stopped.append(name)
        self._records[name].spec.desired_state = DesiredState.STOPPED


@pytest.fixture
def owner_node(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This process is the node that owns the deployment."""
    from docie_bench.serving.node_identity import resolve_node_id

    monkeypatch.setenv("DOCIE_SERVING_HOME", str(tmp_path / "serving"))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", OWNER)
    monkeypatch.setenv("DOCIE_SERVING_DESIRED_APPLY", "1")
    resolve_node_id.cache_clear()
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    try:
        yield
    finally:
        db.dispose_engine()
        resolve_node_id.cache_clear()
        get_settings.cache_clear()


@pytest.fixture
def central_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """This process has no local record, and the catalog names another owner."""
    home = tmp_path / "serving"
    (home / "models").mkdir(parents=True)
    monkeypatch.setenv("DOCIE_SERVING_HOME", str(home))
    monkeypatch.setenv("DOCIE_SERVING_NODE_ID", "node-a")
    get_settings.cache_clear()
    db.dispose_engine()
    db.init_engine(f"sqlite:///{tmp_path / 'catalog.db'}")
    ModelCatalog().publish_node_snapshot(
        node_id=OWNER, total_bytes=1, free_bytes=1, source="cgroup", sum_rss_bytes=0
    )
    try:
        yield
    finally:
        db.dispose_engine()
        get_settings.cache_clear()


def _place(*, live: bool) -> None:
    ModelCatalog().record_placement(
        RECORD,
        model_name=MODEL,
        engine="llama-server",
        endpoint=f"http://{OWNER}:8088/v1" if live else "",
        state="ready" if live else "stopped",
        node_id=OWNER,
    )


def _row() -> dict[str, Any]:
    rows = ModelCatalog().desired_snapshot(OWNER)["deployments"]
    assert isinstance(rows, list)
    return rows[0]


def _write(state: str, *, request: str | None = None) -> None:
    rows = ModelCatalog().desired_snapshot(OWNER)["deployments"]
    existing = rows[0] if isinstance(rows, list) and rows else None
    ModelCatalog().set_desired_deployment(
        OWNER,
        RECORD,
        desired_state=state,
        spec={"model": MODEL, **({"load_request": request} if request else {})},
        expected_generation=int(existing["generation"]) if existing else None,
    )


def _converge(
    supervisor: _Supervisor, *, loader: Callable[[str], object] | None = None
) -> None:
    from docie_bench.serving.reconciler import ServingReconciler

    ServingReconciler(
        supervisor,  # type: ignore[arg-type]
        snapshot_publisher=lambda _snapshot: None,
        loader=loader if loader is not None else supervisor.load,
    )._converge_desired_state()


def _load(name: str = RECORD) -> dict[str, Any]:
    from docie_bench.inngest.serving_api import load_deployment

    return asyncio.run(load_deployment(name, tenant=None))  # type: ignore[arg-type]


def _unload(name: str = RECORD) -> dict[str, Any]:
    from docie_bench.inngest.serving_api import unload_deployment

    return asyncio.run(unload_deployment(name, tenant=None))  # type: ignore[arg-type]


# --- the plan --------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "local", "applied", "expected"),
    [
        ("running", [RECORD], {}, (RECORD,)),
        ("running", [RECORD], {RECORD: "old"}, (RECORD,)),
        ("running", [RECORD], {RECORD: "r1"}, ()),
        ("stopped", [RECORD], {}, ()),
        ("running", [], {}, ()),
    ],
)
def test_only_an_unapplied_request_on_a_local_running_row_plans_a_load(
    state: str, local: list[str], applied: dict[str, str], expected: tuple[str, ...]
) -> None:
    plan = plan_convergence(
        desired={RECORD: state},
        local_names=local,
        previously_applied=[RECORD],
        load_requests={RECORD: "r1"},
        applied_loads=applied,
    )

    assert plan.to_load == expected


def test_a_stopped_record_desired_running_without_a_request_stays_stopped() -> None:
    """The operator-stopped and evicted cases: a generation bump is not a load."""
    plan = plan_convergence(
        desired={RECORD: "running"}, local_names=[RECORD], previously_applied=[RECORD]
    )

    assert plan.is_empty


# --- the owning node -------------------------------------------------------


def test_the_owner_loads_a_stopped_deployment_on_request(owner_node: None) -> None:
    supervisor = _Supervisor()
    _write("running", request="r1")

    _converge(supervisor)

    assert supervisor.loaded == [RECORD]
    assert read_applied_state().loads == {RECORD: "r1"}


def test_replaying_a_carried_out_request_does_not_load_again(owner_node: None) -> None:
    """An unrelated write bumps the generation and repeats the row's request."""
    supervisor = _Supervisor()
    _write("running", request="r1")
    _converge(supervisor)

    ModelCatalog().set_desired_deployment(
        OWNER, "other", desired_state="stopped", spec={}, expected_generation=None
    )
    supervisor._records[RECORD].spec.desired_state = DesiredState.STOPPED
    _converge(supervisor)

    assert supervisor.loaded == [RECORD]


def test_a_failed_load_is_retried_on_the_next_cycle(owner_node: None) -> None:
    supervisor = _Supervisor()
    _write("running", request="r1")

    def refuse(_name: str) -> None:
        raise RuntimeError("no fit")

    _converge(supervisor, loader=refuse)
    assert read_applied_state().loads == {}

    _converge(supervisor)
    assert supervisor.loaded == [RECORD]


def test_a_load_requested_during_a_drain_runs_after_the_resume(owner_node: None) -> None:
    supervisor = _Supervisor()
    _write("running", request="r1")
    ModelCatalog().set_node_drain(OWNER, drained=True)
    with db.session_scope() as session:
        assert session is not None
        row = session.get(ServingNode, OWNER)
        assert row is not None
        row.drained_at = dt.datetime.now(dt.UTC) - dt.timedelta(
            seconds=DRAIN_GRACE_CEILING_S * 2
        )

    _converge(supervisor)
    assert supervisor.loaded == []

    ModelCatalog().set_node_drain(OWNER, drained=False)
    _converge(supervisor)

    assert supervisor.loaded == [RECORD]


# --- the endpoint ----------------------------------------------------------


def test_a_remote_load_records_a_request_and_keeps_the_spec(central_api: None) -> None:
    _place(live=False)
    ModelCatalog().set_desired_deployment(
        OWNER,
        RECORD,
        desired_state="stopped",
        spec={"model": MODEL, "context_length": 8192},
        expected_generation=None,
    )

    result = _load()

    row = _row()
    assert row["desired_state"] == "running"
    assert row["spec"]["context_length"] == 8192
    assert row["spec"]["load_request"]
    assert result["channel"] is None
    assert OWNER in str(result["detail"])


def test_a_load_with_no_desired_row_creates_one(central_api: None) -> None:
    """An evicted deployment made with `docie up` has no row yet."""
    _place(live=False)

    _load()

    assert _row()["spec"]["load_request"]


def test_each_remote_load_is_a_new_request(central_api: None) -> None:
    _place(live=False)
    _load()
    first = _row()["spec"]["load_request"]

    _load()

    assert _row()["spec"]["load_request"] != first


def test_loading_a_live_deployment_writes_nothing(central_api: None) -> None:
    _place(live=True)

    result = _load()

    assert "already serving" in str(result["detail"])
    assert ModelCatalog().desired_snapshot(OWNER)["deployments"] == []


def test_a_load_overrides_a_pending_unload(central_api: None) -> None:
    """Still live because the owner has not acted on the unload yet."""
    _place(live=True)
    _unload()

    _load()

    assert _row()["desired_state"] == "running"
    assert _row()["spec"]["load_request"]


def test_an_unknown_deployment_is_still_a_404(central_api: None) -> None:
    with pytest.raises(HTTPException) as excinfo:
        _load("not-a-deployment")

    assert excinfo.value.status_code == 404
