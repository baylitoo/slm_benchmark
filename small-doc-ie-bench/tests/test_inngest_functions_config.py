"""Config-level tests for Inngest function registration in functions.py.

Covers issue #317: serving-deploy has no in-code admission control against a
concurrent deploy on the shared single-replica host (unlike serving-load's
LoadCoordinator, which already reserves RAM per in-flight load), so it needs a
platform-level concurrency limit instead.
"""

from __future__ import annotations

from docie_bench.inngest import functions


def test_serving_deploy_has_global_concurrency_limit_one() -> None:
    """serving-deploy serializes ALL deploys (no per-model/per-host key).

    The deploy path never checks RAM fit against other in-flight deploys
    before spawning, and every deploy lands on the same single-replica node
    regardless of which model it names -- so the limit must be global, not
    scoped by a key.
    """
    config = functions.deploy_model_job.get_config("http://example.invalid").main
    assert config.concurrency is not None
    assert len(config.concurrency) == 1
    limit = config.concurrency[0]
    assert limit.limit == 1
    assert limit.key is None


def test_serving_load_has_no_platform_concurrency_config() -> None:
    """serving-load already has in-code admission control (LoadCoordinator's
    per-deployment lock + admission lock + in-flight RAM reservations in
    serving/lifecycle.py), so it deliberately carries no redundant
    Inngest-level concurrency limit."""
    config = functions.load_deployment_job.get_config("http://example.invalid").main
    assert config.concurrency is None
