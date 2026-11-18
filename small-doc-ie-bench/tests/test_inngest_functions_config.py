"""Config-level tests for Inngest function registration in functions.py.

Covers issue #317: serving-deploy has no RAM admission against a concurrent
deploy on the same node (unlike serving-load's LoadCoordinator), so deploys on
one node are serialized. That lives in the serving process, not in an Inngest
limit, so a deploy on one node never waits on another node's.
"""

from __future__ import annotations

import threading
import time

import pytest

from docie_bench.inngest import functions
from docie_bench.serving import control_plane


def test_serving_deploy_has_no_global_concurrency_limit() -> None:
    """A global limit of 1 made a long load on one node block every node."""
    config = functions.deploy_model_job.get_config("http://example.invalid").main
    assert config.concurrency is None


def test_a_deploy_waits_for_one_already_running_on_this_node() -> None:
    supervisor = control_plane._DefaultSupervisor.__new__(control_plane._DefaultSupervisor)
    supervisor.model_store_root = None
    holding = threading.Event()
    release = threading.Event()
    finished: list[float] = []

    def running_deploy() -> None:
        with control_plane._DEPLOY_LOCK:
            holding.set()
            release.wait(5)

    def second_deploy() -> None:
        with pytest.raises(ValueError, match="model store is not configured"):
            supervisor.serve_store_model("lfm2.5-350m")
        finished.append(time.monotonic())

    holder = threading.Thread(target=running_deploy)
    holder.start()
    assert holding.wait(5)
    caller = threading.Thread(target=second_deploy)
    caller.start()
    time.sleep(0.2)
    assert finished == []
    released_at = time.monotonic()
    release.set()
    holder.join(5)
    caller.join(5)
    assert len(finished) == 1
    assert finished[0] >= released_at


def test_serving_load_has_no_platform_concurrency_config() -> None:
    """serving-load already has in-code admission control (LoadCoordinator's
    per-deployment lock + admission lock + in-flight RAM reservations in
    serving/lifecycle.py), so it deliberately carries no redundant
    Inngest-level concurrency limit."""
    config = functions.load_deployment_job.get_config("http://example.invalid").main
    assert config.concurrency is None
