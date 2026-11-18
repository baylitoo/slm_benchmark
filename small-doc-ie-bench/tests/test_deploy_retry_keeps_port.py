"""A repeated deploy of a running model must not restart it.

The model's own process holds its port, so the allocator's socket probe sees
it as busy. Without preferring the record's own port, a retried deploy (an
Inngest retry after a long wait behind the per-node deploy lock) picked a new
port, which changed the launch and restarted a loaded model.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from docie_bench.serving.control_plane import _DefaultSupervisor
from docie_bench.serving.runtime import LifecycleState


def _fake_supervisor(existing_port: int | None) -> tuple[Any, list[int]]:
    records = (
        [
            SimpleNamespace(
                spec=SimpleNamespace(name="nuextract3", launch=SimpleNamespace(port=existing_port))
            )
        ]
        if existing_port is not None
        else []
    )
    fake = SimpleNamespace(
        backend=SimpleNamespace(list=lambda: records),
        # The running model holds its port, so the allocator never offers it.
        _port_allocator=lambda: SimpleNamespace(allocate=lambda bind_host, reserved: 8095),
        _reserved_ports=lambda exclude_name=None: set(),
    )
    return fake, []


def _deploy(fake: Any, chosen: list[int]) -> None:
    def launch_and_await(port: int) -> Any:
        chosen.append(port)
        return SimpleNamespace(state=LifecycleState.READY)

    _DefaultSupervisor._deploy_with_port_reallocation(
        fake,
        launch_and_await=launch_and_await,
        bind_host="127.0.0.1",
        exclude_name="nuextract3",
        port_override=None,
    )


def test_a_repeated_deploy_reuses_the_running_deployments_port() -> None:
    fake, chosen = _fake_supervisor(existing_port=8089)
    _deploy(fake, chosen)
    assert chosen == [8089]


def test_a_first_deploy_still_allocates() -> None:
    fake, chosen = _fake_supervisor(existing_port=None)
    _deploy(fake, chosen)
    assert chosen == [8095]
